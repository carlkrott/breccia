# Phase 5 results: coord-lanes (2026-10-29 to 10-31)

Status: merged to `main` on 2026-10-31 (fast-forward from `claude/coord-lanes`; `scripts/verify.sh` passes). Numbers are single runs or short A/B/A/B series on the reference box (`docs/REPRODUCE.md`); 0 GPU hangs in every arm listed.

Measurement caveats:

- The out-of-tree timing harness gated on GPU faults and running builds, **not on load average**. The box idles at a load of about 3.8 (kswapd and ttm kworkers), so absolute tok/s carries a few percent of noise. Ratios come from interleaved arms on the same binary.
- Rep 0 of every MTP bench has a cold plain baseline and is excluded from ratios.
- "Bytes ==" means the dumped logits are byte-identical to the reference run. "Tokens ==" means greedy output is identical.

## Current numbers (RX 6950 XT unless stated, greedy, defaults on unless stated)

| Metric | Value | Notes |
|---|---|---|
| Plain decode, single stream | 21.5-22.4 tok/s (about 46-49 ms/token) | ctx 256 code/prose prompts, top-8 default; exact-routing opt-out 21.3 tok/s (4dce67e build) |
| Decode at 4k context | 48.76 ms/token (was 63.83) | 4k prompt; split-K QSA apply + rank top-k |
| Prefill, 1k prompt (plx: merged prefill, chunk 1024, MI50 + RX) | 314 tok/s top-8 default; 300-308 tok/s exact routing | exact routing (7df7b6f..35300b3, opt-out 307 on 4dce67e): logits bytes == reference. Top-8 default: top-1 identical at all 9039 positions, KL 5.1e-3 vs exact, not byte-identical |
| Two streams (RX + MI50), 1k prompt, gen 32 | 27-28 tok/s aggregate | exact routing, measured before the top-8 default (35300b3); tokens == reference; not re-run under top-8 |
| MTP spec, fixed k=3, code prompt | 1.17x plain | ctx 256, 64 tokens; greedy identical to plain |
| MTP spec, gate gC (default), code / prose | 1.04x / 0.89-0.93x | ctx 256, 64 tokens; see "MTP gate" |

Phase 4 (README, 10-27, exact routing) was 15.8-16.5 tok/s aggregate and 8-10 tok/s single stream.

Spec decoding runs only in the bench path (`BRECCIA_MTP_SPEC_BENCH`); serving does not use it yet.

## Defaults now on (and how to opt out)

| Change | Commit | Effect | Opt out |
|---|---|---|---|
| Top-8 expert routing + 50 % host/GPU verify split | 4dce67e | owner's call 10-31; approximate (routes 8 of 10 experts) | `BRECCIA_EXPERTS_USED=10 BRECCIA_PF_SPLIT_HOST=0` (reproduces the old exact logits) |
| Split-K QSA attention apply (decode/verify, T < 16) | f506856 | 4k decode -24 %; tokens identical on/off | `BRECCIA_QSA_APPLY_SPLIT=0` |
| Parallel rank top-k for QSA block selection | f506856 | same ids/order/count as the serial kernel | `BRECCIA_QSA_TOPK_RANK=0` |
| Tiled GEMM for m >= 64 (4x4) | §12.1c | prefill | `BRECCIA_GEMM_TILE=0` |
| Resident blocks' experts on the RX in prefill | §12.4 | prefill | `BRECCIA_PF_EXPERT_ALL=0` |
| Shared expert under the RX's routed GEMMs | §12.5 | prefill | `BRECCIA_PF_SHARED_UNDER=0` |
| hc fusion | 7df7b6f | prefill, bytes == reference | `BRECCIA_FUSE_HC=0` |
| MTP gate gC | 702d591 | see below | none (gate only used by the MTP bench) |

Opt-in only: `BRECCIA_KV_Q8V=1` (q8_0 V cache, saves 168 MiB, lossy, no speed gain).

## MTP gate gC (default, 702d591)

The gate picks the draft depth k (0..MAX_K) that maximises expected tokens per ms. It uses the live round time of each k and an acceptance estimate per draft position. The old gate had a weak acceptance prior and probed only one depth deeper than its best. One early miss could pin it at k=0 (plain), and a k=1 probe never beats plain, so positions 2+ were never re-learned.

gC changes two things:

- **Stronger prior.** The prior is now 6.4/8.0 (was 1.6/2.0).
- **Deeper probes from plain.** From plain, it probes the best spec depth every 4 picks (was every 16).

64 tokens, warm reps, greedy identical in every rep:

| Prompt | Old gate | gC |
|---|---|---|
| code | 0.90-0.91x | 1.04x |
| prose | 0.95x | 0.89-0.93x |
| code, mid-generation (cx64) | 1.09-1.33x | 1.33x |
| prose, mid-generation (px64) | 0.94-0.95x | 0.93-0.96x |

The decision was to take the code gain at the cost of about -4 % on prose.

## Kept vs dropped (§ numbers refer to the out-of-tree plan record)

**Kept**

- Tiled dense GEMM, LDS-staged X window.
- kv-head-group QSA prefill attention.
- RX-side expert placement in prefill, shared-under, routed gate/up pair for streamed IQ3_S blocks.
- Small-T spec-verify attention, verify split.
- Split-K QSA decode attention + rank top-k.
- MTP draft head with the content-adaptive gate (gC).

**Dropped** (measured, no gain or parity failure)

- Approximate top-k beyond the top-8 default (parity FAIL at every k).
- RX 8x4 GEMM tile.
- MI50 dense-chain fusions.
- GPU-built dispatch lists (`claude/gpu-dispatch` 1de5fdf).
- GDN chunkwise WY scan (`claude/gdn-chunkwise` 8613eaf).
- Multi-device spec with a cross-device expert helper: verify 6.9-7.8x decode vs 2.2-3.0x on RX alone. The cost is the per-block cross-device hop, not expert streaming.
- DSpark/DFlash2 instead of MTP: similar tokens/round, and verify is the wall either way.
- Lean 4 toolchain: none of its claimed techniques touches a measured bottleneck.

**Not built** (candidates for the next round)

- int8 dot4 GEMM. Dense prefill GEMMs run at 23-43 % of MI50 fp32 peak, so this is the largest untapped prefill lever. It is approximate.
- Multi-device block split. Gate first: plain decode on `--devices 03,0e --split-blocks 38,10` must beat RX alone.
- Static command-buffer replay (about 1 ms/token ceiling).

## Reproduce

All runs use `breccia bench token` (release build) with `--arena-display-reserve-mib 2560`, `--threads 30` and `--gpu-experts auto`. Model files and prompt-id files are outside the repo (see `docs/REPRODUCE.md`). Per-arm settings are passed as env vars. The model defaults to a path on the reference box; elsewhere add `--gguf '<model-dir>/*-0000?-of-00002.gguf'` to every command.

```bash
B=target/release/breccia
# plain + MTP spec bench on the RX (arms: fixed k=3 and the gate), 64 tokens
BRECCIA_MTP_SPEC_BENCH=3,gate BRECCIA_MTP_REPS=3 BRECCIA_MTP_GEN=64 \
  $B bench token --streams 1 --threads 30 --gpu-experts auto --prefill-merged --prefill-chunk 1024 \
  --runs 1 --ctx 256 --gen 4 --devices 0000:03:00.0 --arena-display-reserve-mib 2560 \
  --prompt-ids-file <prompt.ids>
# spec verify cost at T=4
BRECCIA_SPEC_VERIFY_BENCH=4 $B bench token <same args>
# prefill (plx), MI50 + RX, 1k prompt
BRECCIA_PREFILL_ONLY=1 BRECCIA_PF_EXPERT_DEVICE=1 $B bench token --streams 1 --threads 30 \
  --gpu-experts auto --prefill-merged --prefill-chunk 1024 --runs 1 --ctx 1100 --gen 1 \
  --devices 0000:0e:00.0,0000:03:00.0 --split-blocks 48,0 --arena-display-reserve-mib 2560 \
  --prompt-ids-file <prompt-1k.ids> --dump-logits <out.bin>
# two streams
$B bench token --streams 2 --devices 0000:03:00.0,0000:0e:00.0 --gpu-experts auto --threads 30 \
  --ctx 1100 --gen 32 --runs 1 --arena-display-reserve-mib 2560 --prefill-merged \
  --prompt-ids-file <prompt-1k.ids>
```

The fault/load gate, the systemd memory limits (MemoryMax 30G, no swap, LimitMEMLOCK=infinity) and the logit-parity script were run out of tree. They are not in this repo.

## Merge state

Merged to `main` 2026-10-31 after `scripts/verify.sh` passed. To get there:

- `cargo fmt --all` over the branch (one style-only commit, graph.rs included).
- Clippy `-D warnings` fixes across breccia-cache, -engine, -kernels and -vk (div_ceil, is_multiple_of, iterator loops, type aliases, test module order); no behaviour change intended.
- Env-dependent tests: `breccia-flags` launch_surface now uses placeholder roots when `BRECCIA_MODEL_ROOT` / `BRECCIA_SLOT_ROOT` are unset (paths are parsed, never opened); `host_arena::oversize_and_overfill_are_clean_errors` skips the mlock-failure check when `RLIMIT_MEMLOCK` is high; `breccia-profile` `live_host_probe_reads_real_values` accepts an unlimited memlock limit.
