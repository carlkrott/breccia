# breccia — project brief

**Status as of 2026-10-27.** This is the orientation document: what the engine is, what it
measures, what has been tried (including what failed and why), the rules that keep
measurements honest, and what is planned. It is written so a new session — or a new agent —
can start without re-deriving anything. `AGENTS.md` holds the binding rules;
`docs/plan/BRECCIA-PLAN.md` holds the gated scope; `docs/plan/PDF-GAP-ANALYSIS.md` maps the
original design conversation onto what is built.

---

## 1. What this is

**breccia** is a from-scratch, Vulkan-first inference engine in Rust for
**Qwen3.8-Flash-Next** (75 GB, IQ3_XXS, GSQ-RCO) running on a **heterogeneous dual-GPU
desktop**: an **RX 6950 XT** (RDNA2, gfx1030, Wave32, 576 GB/s GDDR6 + 128 MB Infinity
Cache) and a **Radeon VII / MI50** (GCN5.1, gfx906, Wave64, ~1 TB/s HBM2). 58 GB RAM, 32 GB
VRAM total, model larger than RAM.

The model is a 48-block MoE hybrid: **36 GDN (gated delta-net) + 12 QSA (attention)** blocks,
`n_embd` 2560, hyper-connections `hc` 4×2560 (lr 320), **512 experts, top-10, n_ff 640**,
vocab 248,320, plus a **27 GB PLE n-gram table** and SSM state `S=128, hv=48`.

**Goal:** beat the production llama.cpp/HIP baseline of **20.79 tok/s** on this exact
hardware, with **both cards doing real work**, scaling to **4–6 concurrent agent streams** —
and do it with a backend that is not tied to CUDA, ROCm, or one vendor's tensor-core layout.

**Success shape** (from the plan): single-stream ≥ 20.79 tok/s, 4–6 agents at usable speed
within 32 GB VRAM + ~30 GB host RAM. Tailnet/mesh work is **deferred by user decision**.

Source: https://github.com/carlkrott/breccia (public; history scrubbed of build artifacts and
hostname before publication).

---

## 2. Non-negotiables

| Rule | Why |
|---|---|
| **Vulkan is the primary backend** | Works across mismatched AMD generations (gfx906 + gfx1030) where ROCm target-matching fails. HIP is secondary. |
| **The production HIP launch flags must keep working** | `qwen-dual-serve.sh` verbatim, including `ROCm0`/`ROCm1` device names — resolved to devices by **PCI identity**, because Vulkan enumeration ordinals are unstable on this box. |
| **Both cards must do real work** | A single-card number is a diagnostic, never a target or a gate. |
| **Every claim carries a measurement** | No milestone is "done" without a number recorded in `docs/phase-N/`. |
| **Model/hardware agnosticism** | Shapes come from GGUF metadata; placement is a rule derived from measured capabilities, not a host-specific block list. |
| **No secrets in the repo** | Public. No tokens, keys, or host-identifying details. |

---

## 3. Architecture

| Crate | Role |
|---|---|
| `breccia-vk` | Device/queue/timeline primitives, bounded waits, `dispatch_batch`, transfer ring, `VK_EXT_memory_budget` probe, host-memory import probe |
| `breccia-kernels` | SPIR-V compute kernels (dequant GEMV/GEMM, ops, GDN, QSA, fan-in, MoE), CPU expert lane, conformance tests |
| `breccia-engine` | The forward pass: layer graph, chunk planner, expert placement/arena, PLE, timers/census, GPU-time probe |
| `breccia-format` | Zero-copy GGUF loader and the pack format |
| `breccia-profile` | Hardware discovery, calibration micro-benchmarks, `hw_profile.json` + BLAKE3 fingerprint |
| `breccia-flags` | The llama.cpp/HIP flag surface, including device-alias resolution |
| `breccia-cache` | Expert residency/caching primitives |
| `breccia-draft`, `breccia-server` | MTP/speculative and the agent server — **stubs today** |
| `breccia-cli` | `breccia profile`, `breccia bench …`, `breccia serve --dry-run-plan` |

~53,700 lines of Rust, 107 shader sources, 10 integration test files plus inline unit tests
(29 test suites total), `scripts/verify.sh` as the gate (build, test, clippy `-D warnings`,
fmt, reference-clone hygiene, no-weights-tracked).

---

## 4. Where it stands (measured, 2026-10-27)

| Metric | Value |
|---|---|
| **Single stream, both cards** | **18.59 tok/s · 53.78 ms/token** (89 % of the 20.79 baseline) |
| **2 streams, both cards** | **22.8–23.7 tok/s aggregate** |
| 2 streams at a 32K context | **23.74 tok/s aggregate** (11.87/stream, 84.2 ms/token) |
| Agents at 32K | **4** with the q8_0 V cache (764 MiB/agent, implemented); 6 with K+V |
| Parity | **32/32 greedy tokens, top-1 13.170510** at chunk 8/16/32/128 |
| Per-card single (diagnostic) | RX 13.3–15.4 tok/s, MI50 9.3–10.4 |
| Expert residency | **30 of 48 blocks** in a **26.7 GiB** arena; the other 18 stream through the host lane at **301 MiB/token** |
| Prefill | ~6.0–6.45 tok/s/agent in a quiet window (chunk 64) |

Single-stream progression this session: 11.78 → 15.33 (expert coverage) → 16.94 → 17.19
(sync2) → 18.26 (head) → 18.59 (IQ4_XS words).

### The per-token budget (GPU-time probe, dual, 30 resident blocks)

```
sum 30,319 µs   vs   wait-gpu 30,691 µs   (residual ~1 %, so the table explains the fence)
```

| Stage | µs/token | Note |
|---|---|---|
| `hc-read` | ~5,300 | 96 dependent reads; batched across blocks is **impossible** (see §7) |
| `gdn-proj` | ~3,240 | IQ4_XS, now word-level |
| `gdn-out` / `gdn-scan` | ~2,790 / ~2,860 | scan is inherently sequential |
| `moe-gate/up/down` | ~3,700 | 366 M el/ms — the best-measured stage |
| `hc-write`, `qkv`, `attn`, `router`, `shared`, `head` | ~1,300–1,800 each | |
| lane (host experts) | ~13 ms | CPU-side, 18 blocks × experts |
| `ple`, `xfer`, `embed` | ~1–2 ms each | |

---

## 5. The rules we now trust (all measured)

1. **Cost is per submit/command-buffer boundary, not per dispatch.** Fence-waited submit
   **33.2 µs** (8.6 submit + 22–24 wait) versus **8.6 µs queued**. Recording a dispatch is
   only **0.28 µs** (and 165 ns of that was our own profiler timers).
   **And the engine is fully synchronous at the submission level: 64 submits/token, 100 %
   fence-waited, 34.0 ms/token spent inside the wait.** Deferring the 30 placed-block waits
   was implemented and **closed the question**: `wait-gpu` fell 30.8 → 10.2 ms/token (−67 %)
   and **the token did not move**, because the waits simply relocated to the next
   fence-bearing operation. So ~30.4 ms of the 34.0 ms is *real GPU execution*, ~3.5 ms is
   latency, and **single-stream `lane + GPU` is structural** — it cannot be fixed by fence
   bookkeeping. The only remaining lever on a single stream is making the GPU work smaller.
2. **A barrier costs `1.1 µs + 0.45 µs × bpr`** — flat in grid size and in bytes, i.e. it is
   the next leg's cold dependent-load chain, not a dirty-L2 flush.
3. **Element rate = `K / I_unit`** with `K = 7.8e12 slots/s` (MI50) and `1.59e13` (RX).
   Quantised GEMV is **ALU-bound on the MI50 by 4–5×** and **DRAM-bound on the RX**. Law:
   ALU-bound iff `I_unit > 8·bpw` (MI50) / `53·bpw` (RX).
4. **But dependency depth beats opcode count.** IQ4_XS word-level extraction *increased*
   instructions/unit 123 → 144 and made the kernel **2.02× faster** (MI50) by removing the
   dependent per-element codebook chain. Steer by dependency depth, not static counts.
5. **BF16/f16 GEMV reaches 98 % of DRAM** (810 GB/s); quantised types do not, which is where
   the remaining headroom is.
6. **PCIe: one stream saturates the link** (25–26.5 GB/s on the RX); only duplex helps
   (17.9 up + 17.9 down). The MI50 is gen3×4 (~2.9 GB/s, ~0 % overlap) and is therefore
   **never** used for streaming.
7. **The GPU can execute host-resident experts directly** via `VK_EXT_external_memory_host`:
   22–26 GB/s dequant GEMV straight out of host RAM (86–92 % of copy-engine rate), no
   staging, no CPU compute. Not yet used by the engine — see §12.
8. **Clock state changes the answer by 4.4×.** The RX idles at 500 MHz; `power_dpm_force_
   performance_level=high` takes it from 3.54 → 15.41 tok/s. Both cards are pinned by
   `gpu-clocks-high.service` + a 60 s timer (a ring reset silently reverts it).

---

## 6. Hardware facts worth knowing

| Fact | Detail |
|---|---|
| **The RX drives the displays** | `boot_vga=1`, DP-1/DP-2/HDMI-A-1 connected. The **MI50 is headless**. Risky kernels go to the MI50 first. Determine this from sysfs, never from an Xorg log (a stale log cost us a day). |
| A reset can reboot the box | The 2026-10-26 23:44 BACO reset happened on the *headless* card and still took the machine down. Bounded waits + `VK_EXT_robustness2` are the real mitigation. |
| Btrfs read-ahead | Defaults to 8 MiB → 9× read amplification on expert/PLE records. Pinned to 128 KiB by `breccia-io-tune.timer` (it silently regresses). |
| `ulimit -l` | 8 MiB hard limit — a pinned host pool needs `LimitMEMLOCK=infinity` in a unit. Pageable import measured the same speed, so this is a correctness-of-pinning issue, not a speed one. |
| `/mnt/agent-storage-a` is a SATA SSD | 0.39 GB/s — wrong place for an expert tier. The system NVMe is the real Tier-3 (~5.9 GB/s at QD8). |
| Missing hardware features | `VK_KHR_cooperative_matrix` absent on both cards; `VK_KHR_performance_query` advertises 17 counters then **segfaults**; DGC compute creation works but **execution faults**. |

---

## 7. What has been tried — outcomes, including the failures

Rounds are grouped by theme; every number lives in `docs/phase-4/01-forward-pass.md` (§8a–§8bg)
or the matching research doc.

### Worked (kept)

| Change | Effect |
|---|---|
| Expert coverage 9 → 30 blocks (config) | single stream 11.82 → **16.94 tok/s**; lane 47.5 → 13.5 ms; expert bytes 799 → 301 MiB/token |
| `sync2` `COMPUTE→COMPUTE` instead of v1 `ALL_COMMANDS` | **1.45×**, −26.2 ms (settled a disagreement between two research runs) |
| Batched PLE (one flush per chunk) | submits/block-chunk 13.9 → 4.65; `ple` 0.94 → 0.42 ms/token (−34 ms/chunk) |
| Prefill chunk planner | 132 tokens → 66+66 instead of 64+64+4 |
| `down` GEMV + `Act` epilogue fusion | −97 dispatches/token, **bit-identical**, parity clean |
| `hc` gated-mean fused into the `up` GEMV epilogue | `hc-read` 7,142 → **5,311 µs**; −96 dispatches |
| GDN `beta|alpha` merged into one 96-row dispatch | −36 dispatches; 827 → 440 µs |
| Head Q5_K "wide" decode (contiguous unit walk) | head 4,254 → **1,772 µs**; 96 → 433 GB/s |
| IQ4_XS word-level nibbles + LDS codebook | **2.02×** (MI50) / 1.38× (RX), bit-identical, +5.1 % single-stream |
| Adaptive expert arena from `VK_EXT_memory_budget` | matches the hand-tuned list within 3 %, with a sysfs-driven display reserve and refuse-not-OOM |
| **q8_0 V cache** (quantise on write, dequantise in registers) | 932 → **764 MiB/agent** at 32K (−168) ⇒ **4 agents**; top-1 stable, greedy 32/32 unchanged; conformance 2.8e-6 vs a 1e-5 bar. **Capacity only** — attention is latency-bound at ~13 GB/s, so it buys no speed (−1.0 %) |
| `--head-device` (flagged, then removed) | head 6.95 → 3.82 ms, but see below |
| GPU-time probe with one query pool per ring slot | the per-stage table above; `sum ≈ wait-gpu` |

### Tried and rejected (recorded so nobody repeats them)

| Idea | Why it failed |
|---|---|
| **Per-dispatch cost model** | cost is per submit/boundary; removing 97 dispatches bought 30 µs |
| **Barrier pruning** (per-resource last-write tracking) | −19.6 % barriers, **zero** `wait-gpu` gain: the graph is a chain through the residual `r`, so the barriers are real dependencies |
| **Persistent descriptor sets** | −0.3 ms, inside noise: the 3.2 µs/dispatch was the whole record path, and the real record cost is 0.28 µs |
| **Pre-recorded CB replay** | worth only 0.8–1.1 ms *after* fusion, and every fusion invalidates the frozen graph |
| **Batching `hc_read` across blocks** | impossible: `hc_read(attn)_{b+1}` consumes `r_{b+1}`, which only exists after block *b*'s ffn `hc_write` |
| **Tiling attention to shrink scratch** | would save **0.4 %** — the per-agent cost is the KV pools, not scratch |
| **Fusing `RmsNorm` into a GEMV prologue** | not bit-identically fusable (separable two-pass norm recomputes per workgroup) |
| **fp16 accumulate** | no speed, worse accuracy |
| **Mega-kernels** | no cooperative launch / forward-progress guarantee; a deadlock is a ring timeout → BACO reset |
| **`VK_EXT_device_generated_commands`** | execution page-faults and ring-resets; and as a recorded command it can never remove a submit |
| **Device-crossing head handoff** | 2 of 16 processes produced a wrong sequence; the crossing path was **removed** |
| **int8 (dp4a) weights+activations** | 1.18× (MI50) / 1.11× (RX) but **accuracy fails**: 6.2e-3 per element, logits 0.51 off, greedy changes. Cause is the `q8_1` activation, not the dot. Flag stays off with a recorded ledger |
| **Pre-dequantised f16 weight cache** | DRAM-bound at 405–413 M el/ms (slower than the int8 path) and needs 17 GiB against 16 GiB/card |
| **Expecting a speedup from KV quantisation** | attention is **latency-bound at ~13 GB/s** at T=1, so cutting V bytes to 0.56× cannot be spent: 32.5 → 32.3 ms (−1.0 %). q8_0 KV is a *capacity* feature only |
| **Push descriptors** | 150 ns vs 39 ns for a bind — worse here |
| **Layer-pipeline split across the cards** | MI50 is gen3×4 (~2.9 GB/s): activation shipping dominates |
| **Pipeline-"storm" theory** for the crash | pipeline creation is host-side ACO compilation with zero GPU submissions |

---

## 8. Measurement discipline (this is why the numbers can be trusted)

Three separate conclusions were wrong before these rules existed, all from dividing a rate by
the wrong denominator. They are now documented in `docs/REPRODUCE.md`:

- **Mixed windows**: a timer reset per window divided by a counter that never resets.
- **Run totals vs per-token**: `--warm 1` adds a whole generation, so a token count taken
  from `runs_n` was 3× wrong (5,167 vs the true **1,843** dispatches/token).
- **Per chunk vs per token**: a "34 ms" saving was really 0.29 ms.

Invariants every claim must show: a `sum · wall · residual` line that is ≈0, and a
**dispatch census whose total equals the run's count** (now `prelude 543 · lane+ffn 1,176 ·
ple 78 · handoff 39 · final-hc 7 = 1,843`).

**Quiet windows are load-bearing.** A co-tenant process on the cards produces both ~15×
slowdowns *and wrong values* (RADV evicting device-local pages under a running kernel). Hence:
`gpu-clocks-high` + 60 s timer, `breccia-io-tune` timer, `VK_EXT_memory_budget` co-tenant
warning, and `BRECCIA_REQUIRE_QUIET=1` which makes a co-tenant **fatal** for gate runs.

**Parity is a gate, and one green run is not evidence.** `scripts/head-handoff-race.sh` runs
N exclusive processes and requires every one to produce the same sequence; the required bar
after a correctness change is **32 generations, zero wrong**.

---

## 9. Safety rules (binding — see `AGENTS.md`)

1. Every GPU run in `timeout 90`; one GPU process at a time, and that process uses **both** cards
   (default run shape in `AGENTS.md` "Dual-GPU first"; single-device only for a named reason).
2. **A new shader's first conformance run on the MI50 (headless)**; everything after it runs dual. Determine the
   display card from sysfs.
3. Validation layer on for first runs of new shader code; `spirv-val` at build.
4. **One quant type per invocation; ≤4 fresh dispatched shapes per process**; pipeline
   pre-warm before sweeps.
5. **Never `vkWaitForFences(u64::MAX)`** — bounded waits + `fault_abort` leave the device
   healthy; `VK_EXT_robustness2` is enabled so OOB reads return zeros instead of faulting.
6. A reset is **stop-the-line**: stop, capture `journalctl -b -1`, report.
7. `VK_KHR_performance_query` and `VK_EXT_device_generated_commands` are **forbidden** here.

---

## 10. Reference engines

18 clones in `Reference Projects/` (git-ignored, read-only, never modified, never committed —
manifest and pinned commits in `docs/reference-engines.md`), from llama.cpp and Strata through
to vLLM, SGLang, mistral.rs, KTransformers, MoE-Infinity, ncnn, DeepSpeed, Ollama, MLC-LLM,
exllamav2 and ggml-vulkan.

**12 paired research rounds** have run alongside implementation, each producing a doc plus a
measured recommendation: `docs/research/01`–`12`. The most consequential:

| Doc | Finding |
|---|---|
| 07 | Attribution tooling; **cost is per submit/boundary**; the 500 MHz clock discovery |
| 08 | Submit/boundary re-measurement; DeepSpeed audit |
| 09 | **Zero-copy host experts work at 22–26 GB/s**; PCIe duplex; NVMe tier pricing; KV-quant price |
| 10 | Dispatch-reduction design; barrier cost model; exllamav2 bit-allocation method |
| 11 | **Recording floor 266 ns** (202 of it our own timers); MLC-Vulkan audit |
| 12 | **Element rate = K/I_unit**; the instruction-mix decomposition; MMQ structure; accuracy contract |

---

## 11. GitHub workflow

- `main` is protected by `scripts/verify.sh`: build, `cargo test --workspace`, clippy
  `-D warnings`, `fmt --check`, reference clones untouched, no weights tracked.
- Phase work commits to `main` once verify is green; the repo is public and pushed after each
  green milestone (156 commits at time of writing).
- Parallel agents work in **git worktrees** under
  `~/.local/share/opencode/worktree/afb070/<name>` on their own branches
  (`gpu-time-table`, `head-device-arena`, `kv-q8v`, `hc-epilogue-fusion`), and the coordinator
  merges (`--no-ff`, conflicts resolved by keeping both sides' new sections).
- `wip/four-gemms-crash-20261026` preserves the work interrupted by the 2026-10-26 reboot.
- Untracked/ignored: `Reference Projects/`, `target/`, model weights, scratch harnesses
  (those live in `/var/tmp/users/korphaus/tmp/opencode/research-harnesses/`, never the repo).

---

## 12. Planned next — ranked by measured value

| # | Item | Expected | State |
|---|---|---|---|
| 1 | ~~Defer the placed-block waits~~ | **DONE, and it closed the question**: `wait-gpu` −67 %, token unchanged ⇒ the engine is GPU-execution-bound; fences were never the problem | done |
| 1b | **`gdn-scan` — DIAGNOSED, fix open.** The state is **3.15 MB per layer** (`S×h_v×S` = 786,432 floats), so 36 blocks × 9.44 MB = **340 MB/token**; at the measured 2,768 µs that is **119 GB/s = 14 % of the MI50's 825**, against a **275 µs one-pass floor**. Cause: 48 workgroups = **1.6 wave64 per CU**, i.e. **MLP-starved**, not arithmetic-bound (the update itself is ~13 µs of FMA). The structural fix is 32 lanes per column with the column in registers, one read + one write, 1,536 workgroups — measured **1.4–1.8× faster (2,768 → 1,547–1,920 µs)** but **it produced wrong logits and was reverted**. It is **not bit-identical** (reduction tree changes summation order), needs an accuracy contract + parity + the 8-process gate, and the prime suspect for the wrong result is the 32-lane xor reduce crossing a **Wave64 boundary**. §8bi. | 275 µs floor ⇒ up to ~2.3 ms | open, with a known-wrong fast version to debug |
| 1c | Then the other small-but-expensive kernels: `hc-norm`/`hc-gated` elementwise, `final-hc`, `moe-swiglu`, `rope`, `kv-append`, `router` | each needs the arithmetic-first diagnosis the head got | queued |
| 2 | **IQ3_S / Q5_K word-level** dequant (dependency depth, not opcodes) | stage-dependent, bit-identical | planned |
| 3 | ~~**V-q8_0 KV**~~ | **landed**: 932 → 764 MiB/agent ⇒ 4 agents (capacity, not speed) | done |
| 4 | **K-q8_0** on top | 728 MiB/agent ⇒ 5–6 agents | next |
| 4b | **32K-context prefill** — a real 32K run is **NVMe-streaming-bound** (~13 tok/s, ~9 s per 128-token chunk; two attempts died at the 900 s / 600 s timeouts) | this, not decode, gates long-context agents; makes radix prefix sharing essential rather than optional | newly found |
| 5 | **Zero-copy host-resident experts** (`external_memory_host`) | removes the CPU lane's serial section | priced, not wired |
| 6 | **Radix/prompt prefix sharing** | agent workloads share a large system prompt | open |
| 7 | **MTP / speculative drafting** | needs the cache fixed first (earlier: 0.60×) | open |
| 8 | **Agent server + continuous batching** (`/v1/chat/completions`) | the product surface | stubs |
| 9 | Tailnet mesh | deferred by user decision | deferred |

**Open risks / unknowns**

- **The step-0 wrong-sequence bug is parked, not fixed.** Rate ≤ 6.1 %/process (96
  generations, 95 % bound); every state/ordering/coverage/weights-content instrument is clean;
  the deliberate fault does not reproduce it. `scripts/head-race-capture.sh` is armed so the
  next occurrence captures layout + weights + logits + a control automatically.
- KV quantisation is lossy and **not yet characterised** — no quality numbers exist for it.
- The engine has never been exercised as a server under real agent load; all numbers are
  benchmark-harness numbers.
- Prefill (~6.0–6.45 tok/s/agent) has had far less attention than decode.

---

## 13. Quick reference

```bash
# gate everything
./scripts/verify.sh

# profile hardware / see the placement plan (no inference)
cargo run -r -p breccia-cli -- profile --dry-run-plan

# canonical single-stream measurement (clocks must read 'high' on both cards)
./target/release/breccia bench token --streams 1 \
  --devices 0000:0e:00.0 --devices 0000:03:00.0 \
  --gpu-experts auto --prompt-ids 760,13017,1433 --gen 32 --runs 1 --warm 1

# cross-process stability / correctness gate
BRECCIA_REQUIRE_QUIET=1 RACE_PROCS=8 RACE_RUNS=2 RACE_GEN=32 RACE_WARM=1 \
  RACE_PROMPT=parity scripts/head-handoff-race.sh

# per-stage GPU-time table and the dispatch census
BRECCIA_GPU_TIME=1 BRECCIA_TRACE_DISPATCH=1 <bench command above>

# capture a divergence automatically (layout, weights, logits, control process)
scripts/head-race-capture.sh
```

Diagnostic env switches: `BRECCIA_REQUIRE_QUIET`, `BRECCIA_DUMP_LAYOUT`,
`BRECCIA_VERIFY_WEIGHTS=all`, `BRECCIA_FAULT_POISON_LOGITS`, `BRECCIA_FAULT_TRUNCATE_KB`,
`BRECCIA_FAULT_DIVERGE_STEP`, `BRECCIA_NO_HC_EPILOGUE`, `BRECCIA_NO_HC_GATED`,
`BRECCIA_NO_GDN_MERGE`, `BRECCIA_NO_GEMV_WORDS`, `BRECCIA_GEMV_DP4A`,
`BRECCIA_NO_PLE_BATCH`, `BRECCIA_NO_PREFILL_PLAN`, `BRECCIA_NO_BARRIER_TRACK`.


---

## 14. Handover state (work stopped here, 2026-10-27)

- `main` is **clean, green and pushed** at `808920a`; `./scripts/verify.sh` passes.
- **Nothing is in flight and nothing is loose.** The `gdn-scan` round was halted and tidied:
  its notes are merged (§8bi), its **wrong-but-fast** 32-lane kernel was **reverted and the
  shader deleted** (it was flag-on by default and produced wrong logits — correctly not left
  in the tree), and its branch `gpu-time-table` (`dff848e`) is trees-clean with nothing
  outstanding.
- Worktrees whose commits are all on main and can be pruned: `gpu-time-probe`
  (`gpu-time-table`, `dff848e`), `hc-epilogue-fusion` (detached, `1bca85e`),
  `head-device-arena` (`3ec0c95`), `kv-q8v` (detached, `d855841`).
- `wip/four-gemms-crash-20261026` is historical (the 2026-10-26 reboot) — keep, don't merge.
- The brief, the plan and the coverage audit are the three docs to read first:
  `docs/BRIEF.md` (this), `docs/plan/BRECCIA-PLAN.md`, `docs/plan/PDF-GAP-ANALYSIS.md`.
  Live work log: `docs/phase-4/01-forward-pass.md` §8a–§8bh.
