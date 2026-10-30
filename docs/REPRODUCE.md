# Reproducing the measurements

Every number in `docs/` came from a command in this file or its phase report. All GPU runs follow the safety rules in `AGENTS.md` (timeout, one process at a time, and **new/risky kernels on the Radeon VII/MI50 first — it is the headless card (`0000:0e:00.0`); the RX 6950 XT (`0000:03:00.0`, `boot_vga=1`) drives DP-1/DP-2/HDMI-A-1**).

Reference hardware (measurements do not transfer verbatim to other systems): Ryzen 9 7950X, 58 GB DDR5, RX 6950 XT (gfx1030, 16 GB GDDR6, Infinity Cache) + Radeon VII/MI50 (gfx906, 16 GB HBM2), Mesa RADV 26.2.4, ROCm 7.2, Linux 6.18.

Model files are **not** in this repository. The target is `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` (75 GB, 2 shards) plus the vision projector, Strata pack and MTP artifacts; place them anywhere and adjust `-m`/`--mmproj`.

## 0. Machine prerequisite — btrfs BDI read-ahead (found 2026-10-26)

The root filesystem is btrfs and the per-BDI read-ahead was **8 MiB**, which causes ~9× read
amplification on random small records (expert blobs and the 90-byte PLE records) and caps cold
reads at 0.7–1.1 GB/s. The block-device `queue/read_ahead_kb` is a **no-op for btrfs** — the knob
that matters is the BDI:

```bash
sudo sh -c 'for b in /sys/class/bdi/btrfs-*; do echo 128 > "$b/read_ahead_kb"; done'
# verify
for b in /sys/class/bdi/btrfs-*/; do echo "$(basename $b)=$(cat $b/read_ahead_kb)"; done
```

Measured effect (`docs/research/03-io-and-exllamav2.md`): cold expert reads **0.7–1.1 → 4.7–5.4
GB/s**, read amplification **9× → 1.05×**, cold model load 28 s → 4 s. Do **not** use `O_DIRECT`
(5.9 GB/s cold but throws away the 58 GB/s warm page cache), `FADV_RANDOM` (ignored by btrfs) or
`MADV_RANDOM` (1.4–2× worse). **It regresses silently** (found back at 8192 KiB on 10-27), so it is
re-applied every 5 minutes by `breccia-io-tune.timer` — check the value before any cold-read claim.

## 0b. Units discipline — read this before quoting any derived cost

Three separate conclusions were wrong between 2026-10-26 and 10-27 because a **rate was divided by
the wrong denominator**. All three are cheap to avoid:

| Trap | Symptom | Rule |
|---|---|---|
| **Mixed windows** | `record` printed 6.9 µs/dispatch when the true cost is **0.28 µs** | a counter reset per window (`reset_record_seconds`, the decode window) must be divided by a counter that resets with it. `Session::dispatches` is a **whole-session** total and is never reset. |
| **Run total vs per token** | dispatch count read 5 167/token; the true figure is **1 843** | `--warm 1` adds a whole generation, so a run holds more token-graphs than `--gen`. Divide by token-graphs, not by `runs_n`. |
| **Per chunk vs per token** | a "34 ms" submit saving was really **0.29 ms** | state the unit in the label: `ms/token`, `ms/block-chunk`, `dispatches/token`. |

Two invariants the engine now prints, and which every claim should show: the **`sum · wall · residual`**
line for a prefill chunk (must be ≈0), and the **per-stage dispatch census** whose total must equal the
run's dispatch count. If a table doesn't sum, the table is wrong — not the model.

Clock state belongs in every run header as well: `power_dpm_force_performance_level` must read `high` on
**both** cards (`gpu-clocks-high.service` + `gpu-clocks-high.timer` re-assert it every 60 s, because a
ring reset silently drops the RX back to `auto`, which measures **4.4× slower**). `breccia-io-tune.timer`
re-applies the btrfs read-ahead below on the same cadence.

## 1. Baselines (llama.cpp HIP, the number to beat)

```bash
# production-equivalent server (build with qwen4exp support; see docs/PROJECT-SETUP.md)
llama-server -m <shard1.gguf> --device ROCm0,ROCm1 -ngl 99 -ts 9,15 \
  -cmoe --moe-cache-mib 0 -lzm on -lm none --no-host \
  -c 200000 -kvu -np 1 --kv-unified-per-slot 200000 -ctk q4_0 -ctv q4_0 -fa on \
  -t 16 -tb 16 --host 127.0.0.1 --port 8160 --jinja --metrics --fit off --no-op-offload -ub 256

python3 harnesses/bench_decode.py --url http://127.0.0.1:8160 --out /tmp/baseline.json \
  --runs 3 --gen-tokens 256
```

Reported baseline (`docs/phase-0/01`): **20.79 tok/s** decode (server `eval time`), **112.72 tok/s** prefill @1k tokens. Caveat: this build's HIP device access needs `/dev/kfd` (root:gpu-compute) — run under `sudo -n` or a unit with `SupplementaryGroups=gpu-compute`.

## 2. Backend probe (raw Vulkan, both cards)

```bash
cargo run -p breccia-cli -- profile --quick --json /tmp/hw.json      # calibration
cargo run -p breccia-cli -- profile --dry-run-plan --json /tmp/hw.json
harnesses/vk-probe/target/release/vk-probe --json /tmp/raw.json      # bandwidth/PCIe/overlap
```

Expected (`docs/phase-0/02`): RX 462 GB/s DRAM (1811 GB/s in-cache), 26.7 TF FP32, 28.7 GB/s PCIe, ~100% overlap; MI50 826 GB/s HBM, **~2.9 GB/s PCIe (gen3×4)**, ~0% overlap (historical: the link is now PCIe 4.0 ×4, expect 5.4–6.6 GB/s).

**Environment trap**: `MESA_VK_DEVICE_SELECT=1002:164e!` (set by some desktop sessions) hides both discrete GPUs from every Vulkan process. `breccia` unsets it with a warning; manual runs need `env -u MESA_VK_DEVICE_SELECT`.

## 3. Transfer ring + expert churn (M3 shape)

```bash
cargo run --release -p breccia-cli -- bench transfer --json /tmp/transfer.json
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  timeout 90 cargo run -p breccia-cli -- bench transfer --quick
```

Expected (`docs/phase-2/01`): RX ring 28.25–28.71 GB/s, churn overlap 90.8–91.2%, consumer stall 0.003 ms/MB; MI50 2.71–2.84 GB/s, 58%.

## 4. Kernels

```bash
# GEMV throughput + real mmap'd expert blob (no model copy)
cargo run --release -p breccia-cli -- bench gemv --type IQ3_S --rows 640 --cols 2560 --json /tmp/gemv.json
cargo run --release -p breccia-cli -- bench gemv --real --role down --json /tmp/gemv-real.json

# conformance — one type per process, MI50 first (safety rule)
timeout 90 env VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
  BRECCIA_KERNEL_DEVICE=0000:0e:00.0 BRECCIA_QTYPE=IQ3_S \
  cargo test -p breccia-kernels --test conformance

# elementwise ops
cargo run --release -p breccia-cli -- bench ops --op rmsnorm --json /tmp/ops.json
timeout 90 env BRECCIA_KERNEL_DEVICE=0000:0e:00.0 BRECCIA_OP=rmsnorm \
  cargo test -p breccia-kernels --test ops_conformance
```

Expected (`docs/phase-3/01–04`): 271–470 GB/s per IQ type on RX (83–106% of 445 GB/s), expert batching 3.4×, ops 1.6–9.2 µs at T=1.

## 5. Launcher compatibility (HIP flags → Vulkan)

```bash
# the production qwen-dual-serve.sh argv parses unchanged; devices resolve by PCI identity
cargo run -p breccia-cli -- serve --dry-run-plan --device ROCm0,ROCm1 -ngl 99 -c 32768 \
  -ctk q4_0 -ctv q4_0 -fa on -ot 'blk\.(0|1)\.ffn_exps\.=ROCm0'
```

Expect: `ROCm0 -> AMD Radeon RX 6950 XT [pci 0000:03:00.0]`, `ROCm1 -> Radeon VII [0000:0e:00.0]`, one compiled override, zero writes. (Ordinals are deliberately ignored — Vulkan enumeration order is unstable on this host; `docs/phase-1/profile-report.md`.)

## 6. Full test gate

```bash
./scripts/verify.sh          # build + test + clippy + fmt + reference-clone/weights checks
```
