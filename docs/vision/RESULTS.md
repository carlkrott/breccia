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
| 448^2 (196 tok) | 6/47, split at "captures" (p~0.28) vs ref "depicts" (p~0.65) | 13/20 | "This image captures a serene, misty mountain landscape at dawn or dusk, featuring snow-capped peaks glowing with soft alpenglow above a dense, dark forest of coniferous trees shrouded in blue-toned..." |
| 1024^2 (1024 tok) | 10/45, split at a near-tie: "painterly" (p~0.25) vs ref "mist" (p~0.35) | 16/20 | "This image depicts a serene, painterly mountain landscape at twilight, featuring snow-capped peaks glowing with soft pink and orange alpenglow against a cool blue sky, with a dense, dark forest of coniferous trees" |

Ref (server, 448): "This image depicts a serene, misty mountain landscape at dawn or dusk, with snow-capped peaks glowing softly in alpenglow above a dense, dark coniferous forest shrouded in blue haze."
KV gate (same prompts, BRECCIA_KV=f16 vs default q6_0):
| KV | 448 prefix | 1024 prefix | 1024 top-20 |
|---|---|---|---|
| q6_0 | 6/47 | 10/45 | 16/20 |
| f16  | 6/47 | 4/45 ("A serene, painterly landscape...") | 16/20 |
f16 agrees no longer than q6 -> q6 KV costs no vision fidelity here; the gap vs llama.cpp is GPU IQ3 kernels vs CPU
(llama.cpp CPU fed the same dump reproduces the server exactly). Colours checked against img1024.png: pink/salmon
alpenglow on the peaks, blue sky, painterly look -> Breccia's "pink and orange alpenglow ... cool blue sky" is right.
Decode 448: 16.0 tok/s, 1024: 25.2 tok/s (single runs, not ABBA; not a speed claim).

## Stage 2: `breccia serve` takes images (2026-11-05)

`breccia serve --mmproj <mmproj.gguf> [--vision-encoder BIN] [--image-tokens 2048]`. Chat `image_url` parts
(`data:<mime>;base64,...` only, one image per request, 64 MiB body cap) become
`<|vision_start|><|image_pad|>x(nx*ny)<|vision_end|>` in place; the CPU sidecar (`vision-encode`, next to the
binary or on PATH, VE_THREADS default 30) encodes per request with min 1024 / max 2048 tokens; usage counts the
image cells. A request without an image clears any previous one. Without `--mmproj` image parts are a 400.

Token budget on CPU (img2048.png = 2560x1440 wallpaper at 2048^2, 30 threads, warm): 2025 tok 7.9 s, 4096 tok 34 s
(over the 10 s limit -> 2048 max). 16 threads: 2025 tok 9.8 s.

Live (default dual devices, q6 KV, ctx 4096, scripts vis-serve.sh):
| image | grid | encode | request | prompt tok | answer |
|---|---|---|---|---|---|
| 448^2 (lifted to min) | 32x32 | 3.14 s | 36.6 s (first request, warm-up) | 1041 | "...serene, misty mountain landscape at dawn or dusk, with snow-capped peaks glowing softly in alpenglow above a dense, dark coniferous forest under a cool blue sky." |
| 1024^2 | 32x32 | 3.25 s | 12.9 s | 1041 | "...serene, painterly mountain landscape at twilight ... soft pink and orange alpenglow against a cool blue sky..." (= Stage 1 CLI) |
| 2048^2 | 45x45 | 8.72 s | 22.8 s | 2042 | "...serene, stylized mountain landscape at twilight or dawn ... soft pink and orange alpenglow ... dark coniferous forest silhouetted..." |
| text after | - | - | 1.7 s | 14 | normal thinking reply (image cleared) |
