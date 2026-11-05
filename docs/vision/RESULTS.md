# Qwen3.8 Flash Next vision encoder: backend measurements (2026-11-05)

Harness: /var/tmp/breccia-coord/vision/vision-encode.cpp (public mtmd API, libmtmd from
llama.cpp-qwen4exp-mmap/build-vulkan 1b70350c9; text model loaded vocab_only, no weights).
mmproj: /srv/strata-data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf (qwen3vl_merger, proj 2560).
Image: wallpaper resized to 448^2 / 1024^2. Warm encode = second of two passes.

| backend | 448^2 (196 tok) | 1024^2 (1024 tok) | peak host RSS | compute buf |
|---|---|---|---|---|
| iGPU Vulkan (RADV Raphael, 2 CU) | 2876 ms | 27491 ms | 0.21 GB | 120 MiB (1024) |
| CPU, 16 threads | 421 ms | 3016 ms | 1.22 GB | 120 MiB |
| RX 6950 XT Vulkan | 83 ms | 445 ms | - | 120 MiB |

Agreement vs CPU (per-token cosine, mean / min; relative L2 error):
- iGPU 448: 0.99970 / 0.98582, 2.6e-2   1024: 0.99973 / 0.98071, 2.3e-2
- RX   448: 0.99976 / 0.98868, 2.4e-2   1024: 0.99978 / 0.99090, 2.2e-2

Token layout: wrapped as 248053 <image> 248054; decoder pos (t,x,y) raster, t=0, nx=ny=32 at 1024^2,
n_pos = max(nx, ny) (14 / 32). Matches the vision-research.md plan.

Verdict: iGPU is 7-9x slower than the CPU -> not used. CPU sidecar = default (no VRAM);
RX ~7x faster than CPU = the target for a native Breccia encoder later.
d1 on the iGPU (:8137) stayed healthy during the iGPU runs.

## Stage 1: Breccia text model on CPU-sidecar image embeddings (2026-11-05)

Path: vision-encode (CPU, mtmd) dumps `.f32` rows -> `breccia --setenv=BRECCIA_IMAGE_EMBD=<prefix>`;
the prompt's single `<|image_pad|>` (248056) expands to nx*ny cells, rows replace the token
embeddings, image cells get mtmd M-RoPE triples (t=p, y=p+i/nx, x=p+i%nx), text resumes at p+max(nx,ny),
PLE uses 248056 at image cells (llama.cpp `ple_image_token_id`), pooled QSA keys take the first member's triple.

Checks:
- Dump validity: CPU llama.cpp fed the `.f32` dump (vision-feed.cpp) = server ref 47/47 tokens.
- Bug found + fixed: `Session::generate()` called `reset()`, which dropped the image and its positions,
  so earlier "vision" runs were text-only over bare image_pad tokens. generate() now keeps an image whose
  cell still holds image_pad in the prompt and re-applies its positions.
- Text regression (2k prompt, 16 gen, RX): old q6 binary vs vision binary: generated + first-token top-20 identical.
- Unit: qsa::tests::mrope_triples_follow_mtmd ok.

| case | greedy prefix vs llama.cpp server | first-token top-20 overlap | Breccia output |
|---|---|---|---|
| 448^2 (196 tok) | 6/47, split at a near-tie ("captures" vs "depicts") | 13/20 | "This image captures a serene, misty mountain landscape at dawn or dusk, featuring snow-capped peaks glowing with soft alpenglow above a dense, dark forest of coniferous trees shrouded in blue-toned..." |
| 1024^2 (1024 tok) | 10/45, split at "mist" vs "painterly" (ref lp -1.06 vs -1.39) | 16/20 | "This image depicts a serene, painterly mountain landscape at twilight, featuring snow-capped peaks glowing with soft pink and orange alpenglow against a cool blue sky, with a dense, dark forest of coniferous trees" |

Ref (server, 448): "This image depicts a serene, misty mountain landscape at dawn or dusk, with snow-capped peaks glowing softly in alpenglow above a dense, dark coniferous forest shrouded in blue haze."
Divergence is likely numeric, unproven (IQ3 GPU kernels + q6_0 KV vs CPU f16 KV), same content and colours. Not yet run: q6 vs f16 KV A/B on image prompts.
Decode 448: 16.0 tok/s, 1024: 25.2 tok/s (single runs, not ABBA; not a speed claim).
