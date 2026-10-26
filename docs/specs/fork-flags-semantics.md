# Fork-only launch flags — actual semantics

Reference: `/home/korphaus/llama.cpp-qwen4exp-mmap/common/arg.cpp` (commit 1b70350c9). These flags appear in `qwen-dual-serve.sh` and `breccia-flags` parses them, but their **meaning** lives only in the fork's source — this document is what the engine must replicate for behavioural parity.

| Flag | Fork semantics (source) | Effect on the 20 tok/s configuration | Breccia requirement |
|---|---|---|---|
| `-cmoe`, `--cpu-moe` | *"keep all Mixture of Experts (MoE) weights in the CPU"* — appends LLM-FFN-exps→CPU to `tensor_buft_overrides` (`arg.cpp:2763`) | Every `ffn_*_exps` defaults to CPU; the `-ot` entries then pin 13 specific blocks to GPUs (13,577 MiB total). CPU computes the remaining 35 blocks' experts per token. | Default expert placement = host; per-block GPU overrides must beat the default (specific > broad, as `-ot` does here). |
| `--moe-cache-mib N` | *"GPU cache size in MiB for the MoE experts kept in the CPU… split among GPUs like the layers (`--tensor-split`); 0 = disabled"* (`arg.cpp:2781`) | `0` → **no** GPU expert cache; CPU experts are read straight from the mmap/page cache each token. | Expert cache is a *separate* mechanism from placement; size 0 means "no residency budget", not "no experts on GPU". |
| `-lzm`, `--lazy-mode on` | *"read the rows of such tensors from disk on demand instead of keeping them resident (requires mmap); auto = on only >4 GiB"* (`arg.cpp:2713`) | PLE table (28.8 GB `per_layer_token_embd`) is **row-gathered from the GGUF mapping per token** (`CPU_Mapped`, lazy read). | PLE must be an on-demand gather over a mapped file — never fully resident. This is why host RSS stays ~24 GB. |
| `-lm`, `--load-mode none` | *"no special loading mode"* (`arg.cpp:2694`) | No mlock, no forced residency; plain mmap with fadvise hints from the env (`LLAMA_MMAP_FADVISE=random`). | Respect load-mode enum at least for `none`/`mmap`/`mlock`. |
| `-cram`, `--cache-ram N` | Prompt-cache-in-RAM limit in MiB; `0` disables (`arg.cpp:1719`, upstream PR #16391) | Disabled; slot save/restore to `/mnt/agent-storage-a/kv-slots/…` is used instead (context persistence on SSD). | Two distinct persistence mechanisms: RAM prompt cache vs slot files; only the latter is used here. |
| `-ctxcp`, `--ctx-checkpoints N` | Max context checkpoints per slot (`arg.cpp:1701`, PR #15293) | `1` → one full-context checkpoint (~112.6 MiB host at 200k ctx) supporting fast slot/context reuse; more than one blows the 38 G cgroup cap. | Checkpoint count is a memory-budget multiplier — must be explicit in the profile plan. |
| `-kvu`, `--kv-unified` | Single unified KV buffer shared across sequences (`arg.cpp:1727`) | One 200k-token KV buffer (`--kv-unified-per-slot 200000` on top) rather than per-slot slabs. | KV arena layout choice; affects slot-count scaling (M5). |
| `-fit`, `--fit off` | *"whether to adjust unset arguments to fit in device memory"* (`arg.cpp:2881`) | Off: the declared split/overrides are honoured exactly; no auto-shrink that would silently change the measured layout. | Keep `fit off` equivalent: explicit plans, no hidden adjustments in benchmark runs. |
| `--no-host` | *"bypass host buffer allowing extra buffers to be used"* (`arg.cpp:2433`) | Avoids the host-mapped buffer path for tensor staging; enables extra device buffers with the same VRAM. | Buffer-allocation policy flag, not a placement flag. |
| `--no-op-offload` | *"whether to offload host tensor operations to device"* — inverted bool (`arg.cpp:2961`) | Host-side ops (CPU expert math, PLE gather) stay on the host; no automatic device offload. | CPU-side ops must be explicit in the graph, never auto-offloaded. |

## Ordering rule to replicate

`-cmoe` sets a **broad default** (all experts → CPU); `-ot` then adds regex-scoped overrides. In the fork, later/more specific `tensor_buft_overrides` win for the blocks they match — verified by the production memory layout (13 expert blocks resident on the two GPUs, the rest on the host: `docs/phase-0/05-fullmodel-vulkan.md`). Breccia's placement resolver must apply the same precedence: **explicit per-tensor override > role default (`-cmoe`) > profile heuristic**.

## Why this matters for the 85 tok/s goal

The current 20.79 tok/s divides into: ~52 GB of expert weights read from host memory per token (CPU lane) + 13.6 GB of GPU-resident weights + PLE row gathers from a 28.8 GB lazily-mapped table. Any speed-up therefore needs (a) fewer host-resident expert bytes per token (route-tail skipping / better expert packing — M3), (b) faster CPU expert math or GPU residency of hot experts (`--moe-cache-mib` > 0 is the fork's lever we have not measured yet), and (c) PLE gather efficiency. The flag semantics above define what "parity" means before we change any of it.
