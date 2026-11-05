// breccia vision sidecar (stage 0/1): encode one image with llama.cpp's mtmd/clip
// on whatever ggml backend is visible, dump f32 [n_tokens, n_embd] + positions.
// Usage: vision-encode <mmproj> <text-model.gguf> <image> <out-prefix> [max_tokens] [cpu]
// The text model loads vocab_only (metadata + tokenizer, no weights).
// Env: VE_MIN_TOKENS (image_min_tokens), VE_THREADS (default 16), VE_BENCH=1 (encode twice, report warm).
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <mmproj> <model> <image> <out-prefix> [max_tokens] [cpu]\n", argv[0]);
        return 2;
    }
    const int  max_tokens = argc > 5 ? atoi(argv[5]) : 0;
    const bool cpu        = argc > 6 && strcmp(argv[6], "cpu") == 0;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model * model = llama_model_load_from_file(argv[2], mp);
    if (!model) { fprintf(stderr, "vocab load failed\n"); return 1; }

    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu       = !cpu;
    cp.print_timings = true;
    cp.n_threads     = getenv("VE_THREADS") ? atoi(getenv("VE_THREADS")) : 16;
    if (getenv("VE_MIN_TOKENS")) cp.image_min_tokens = atoi(getenv("VE_MIN_TOKENS"));
    cp.warmup        = false;
    if (max_tokens > 0) cp.image_max_tokens = max_tokens;
    auto t0 = std::chrono::steady_clock::now();
    mtmd_context * ctx = mtmd_init_from_file(argv[1], model, cp);
    if (!ctx) { fprintf(stderr, "mmproj load failed\n"); return 1; }
    auto t1 = std::chrono::steady_clock::now();

    mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(ctx, argv[3], false, mtmd_helper_init_opt_default());
    if (!bw.bitmap) { fprintf(stderr, "image load failed\n"); return 1; }

    mtmd_input_text txt = { mtmd_default_marker(), strlen(mtmd_default_marker()), false, true };
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bms[1] = { bw.bitmap };
    if (mtmd_tokenize(ctx, chunks, &txt, bms, 1) != 0) { fprintf(stderr, "tokenize failed\n"); return 1; }

    // shortcut: vocab_only leaves hparams at 0; 2560 = mmproj projection_dim for Qwen3.8 Flash Next, read it from the GGUF if another model shows up
    int n_embd = llama_model_n_embd_inp(model);
    if (n_embd <= 0) n_embd = 2560;
    for (size_t c = 0; c < mtmd_input_chunks_size(chunks); c++) {
        const mtmd_input_chunk * ch = mtmd_input_chunks_get(chunks, c);
        if (mtmd_input_chunk_get_type(ch) != MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            size_t n = 0;
            const llama_token * tk = mtmd_input_chunk_get_tokens_text(ch, &n);
            fprintf(stderr, "text chunk:");
            for (size_t i = 0; i < n; i++) fprintf(stderr, " %d", tk[i]);
            fprintf(stderr, "\n");
            continue;
        }
        const size_t n_tok = mtmd_input_chunk_get_n_tokens(ch);
        // encode twice: the first pass carries pipeline/shader compile cost
        double ms[2] = {0, 0};
        const int passes = getenv("VE_BENCH") ? 2 : 1;
        for (int r = 0; r < passes; r++) {
            auto a = std::chrono::steady_clock::now();
            if (mtmd_encode_chunk(ctx, ch) != 0) { fprintf(stderr, "encode failed\n"); return 1; }
            ms[r] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
        }
        const float * e = mtmd_get_output_embd(ctx);
        const mtmd_image_tokens * it = mtmd_input_chunk_get_tokens_image(ch);

        std::string base(argv[4]);
        FILE * f = fopen((base + ".f32").c_str(), "wb");
        fwrite(e, sizeof(float), n_tok * n_embd, f);
        fclose(f);
        f = fopen((base + ".pos").c_str(), "w");
        for (size_t i = 0; i < n_tok; i++) {
            mtmd_decoder_pos p = mtmd_image_tokens_get_decoder_pos(it, 0, i);
            fprintf(f, "%u %u %u\n", p.t, p.x, p.y);
        }
        fclose(f);

        double sum = 0, sq = 0;
        for (size_t i = 0; i < n_tok * n_embd; i++) { sum += e[i]; sq += (double) e[i] * e[i]; }
        printf("{\"n_tokens\":%zu,\"n_embd\":%d,\"n_pos\":%d,\"load_ms\":%.0f,\"encode_ms_cold\":%.1f,"
               "\"encode_ms\":%.1f,\"mean\":%.6g,\"rms\":%.6g,\"backend\":\"%s\"}\n",
               n_tok, n_embd, (int) mtmd_input_chunk_get_n_pos(ch),
               std::chrono::duration<double, std::milli>(t1 - t0).count(), ms[0], ms[passes - 1],
               sum / (n_tok * n_embd), sqrt(sq / (n_tok * n_embd)), cpu ? "cpu" : "gpu");
    }
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bw.bitmap);
    mtmd_free(ctx);
    llama_model_free(model);
    return 0;
}
