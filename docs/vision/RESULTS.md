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

Non-square check (2560x1440 JPEG wallpaper, same budget): Breccia 60x33 = 1980 cells, encode 8.16 s, request 21.0 s,
"A serene twilight landscape features snow-capped mountains glowing with soft pink alpenglow against a deep blue sky,
rising above a dense, dark coniferous forest." llama.cpp server (--image-min-tokens 1024 --image-max-tokens 2048,
vision/ref-wide.sh): same 1997 prompt tokens, "A serene twilight scene features snow-capped mountains glowing with soft
pink and orange alpenglow against a deep blue sky, framed by a dark, silhouetted forest in the foreground." -> x/y
layout right for nx != ny. Encoder temp dir: $TMPDIR/breccia-vision-<pid>, created 0700 at startup (existing = fatal),
files removed after each encode.

## Stage 3: multi-image requests + on-disk image cache (2026-11-05)

`breccia serve ... [--image-cache DIR] [--image-ttl-min 30]`. Default dir `$XDG_CACHE_HOME|~/.cache/breccia/vision`
(created 0700, must be owned by the server user). Each encoded image is stored as `<key[..2]>/<key>.f32|.pos`, key =
blake3(mmproj path+size+mtime, min/max tokens, image bytes) -> a resent image skips the CPU encoder. A cache hit refreshes
the entry's TTL (sliding 30 min). Entries unused for the TTL, orphan files and stale `.tmp-*` files are pruned at startup and
after each encode; total size capped at 4 GiB (oldest first). Writes go to `.tmp-<pid>-<key>` files, then a rename.

Several `image_url` parts per request: encoded one at a time, in order. Each expands its own `<|image_pad|>` run with
its own M-RoPE span. After each image the prompt length is checked against `--ctx`; going over stops with a 400
naming the image. The pad-run count must match the image count. A request without images drops earlier ones.

Live (ctx 4096, q6 KV, dual default, script vis-multi.sh):
| request | time | prompt tok | result |
|---|---|---|---|
| red circle + blue triangle (first request: includes server warm-up + 2 encodes) | 44.5 s | 2079 | "Red circle / Blue triangle" (2 encodes 3.3 + 3.1 s) |
| same resend | 11.7 s | 2079 | same; both images "from cache in 0.00 s" |
| swapped order | 13.3 s | 2079 | "Blue triangle / Red circle" (order follows the request) |
| 3 x 2048^2 | 9.0 s | - | 400 "image 3 of 3 brings the prompt to 6108 tokens, over the 4096-token context" |
| 2560x1440 wallpaper + red circle | 24.2 s | 3027 | image 1 alpenglow landscape, image 2 "solid red circle ... plain white background" |
| text after | 1.6 s | 14 | normal reply |
What the cache saves is the encode only: ~3.2 s per 1k-token image, ~9 s per 2k one. The resend and swapped requests
(11.7 / 13.3 s, fully cached) are prefill + decode, still paid on every request; most of the 44.5 -> 11.7 s gap is warm-up.
The cache key also covers the encoder binary (path, size, mtime), so a rebuilt or swapped encoder never serves old rows.
Cache dir after: 4 entries (10 MiB per 32x32, ~20 MiB per ~2k-token image), dirs 0700, no `.tmp` left.
Unit tests: server image order + ctx stop (http.rs), cache hit / expiry prune (breccia-cli image_cache_tests).

EmbeddingGemma-2 (MSI :8141) not used: the exact content hash is the right key for reuse; a near-duplicate image
cannot reuse rows (they depend on the exact pixels and grid); the MSI serves the text GGUF without an mmproj.

## Prefix reuse: saved prompt state for image prompts (2026-11-05)

`breccia serve` keeps the prompt state up to the end of the last image (`<|vision_end|>`) in host RAM:
GDN state + conv per block, the used QSA K/V pool prefix (page-major, q6 scales inline), idx tail/pooled, PLE
history, position. Key = blake3(prompt tokens up to the cut + each image's content hash). A hit restores the state
and prefills only the text after the images. Entries expire after the image TTL (`--image-ttl-min`, 30 min sliding,
pruned on every request); RAM cap 1 GiB total, least recently used dropped first. Engine API:
`Session::generate_prefix(prompt, n, debug, Prefix::{None, Save(cells), Resume(&state)})`.

Live (ctx 4096, q6 KV, dual default, script vis-prefix.sh, greedy, order A B A B A so B overwrites A's cells):
| request | time | prompt tok | prefix | output |
|---|---|---|---|---|
| A red+blue (first request: warm-up + 2 encodes) | 45.5 s | 2079 | 2055 cells saved, 135 MiB in 0.32 s | "Red circle / Blue triangle" |
| B wallpaper+red (1 encode 9.0 s) | 27.0 s | 3035 | 3011 cells saved, 145 MiB in 0.35 s | "Image 1: ... mountain ... blue / Image 2: ... circle ... red" |
| A again | 6.7 s | 2079 | restored in 0.03 s | identical to cold |
| B again | 6.7 s | 3035 | restored in 0.03 s | identical to cold |
| A again | 6.7 s | 2079 | restored in 0.03 s | identical to cold |
| text after | 1.6 s | 14 | - | normal reply |
Fully cached resend: 11.7 s (Stage 3, encode cached, prefill recomputed) -> 6.7 s; for B, 27.0 -> 6.7 s.
The remaining 6.7 s is the per-request reset (zeroing the full pools), the image row upload, the ~24-token
tail prefill and decode; not split further.
