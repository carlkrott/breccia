// breccia vision check: run the full text model (CPU) on an image prompt, the image
// either encoded fresh by mtmd (control) or fed from a vision-encode .f32 dump, so a
// dump can be judged on its own. Greedy decode, first-token top-20 as JSON lines.
// Usage: vision-feed <model> <mmproj> <image> <n_gen> [embd.f32]
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <model> <mmproj> <image> <n_gen> [embd.f32]\n", argv[0]); return 2; }
    const int n_gen = atoi(argv[4]);
    const char * file = argc > 5 ? argv[5] : nullptr;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    static ggml_backend_dev_t no_dev[1] = { nullptr };
    mp.devices = no_dev; // CPU only: no GPU contention with Breccia/d1
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 512; cp.n_ubatch = 512; cp.n_threads = 16; cp.n_threads_batch = 16;
    cp.op_offload = false;
    llama_context * lctx = llama_init_from_model(model, cp);
    if (!lctx) { fprintf(stderr, "ctx failed\n"); return 1; }

    mtmd_context_params mcp = mtmd_context_params_default();
    mcp.use_gpu = false; mcp.n_threads = 16; mcp.warmup = false;
    mtmd_context * mctx = mtmd_init_from_file(argv[2], model, mcp);
    if (!mctx) { fprintf(stderr, "mmproj load failed\n"); return 1; }
    mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(mctx, argv[3], false, mtmd_helper_init_opt_default());
    if (!bw.bitmap) { fprintf(stderr, "image load failed\n"); return 1; }

    std::string p = std::string("<|im_start|>user\n") + mtmd_default_marker() +
                    "Describe this image in one sentence.<|im_end|>\n<|im_start|>assistant\n";
    mtmd_input_text txt = { p.c_str(), p.size(), false, true };
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const mtmd_bitmap * bms[1] = { bw.bitmap };
    if (mtmd_tokenize(mctx, chunks, &txt, bms, 1) != 0) { fprintf(stderr, "tokenize failed\n"); return 1; }

    llama_pos n_past = 0;
    const size_t nc = mtmd_input_chunks_size(chunks);
    for (size_t c = 0; c < nc; c++) {
        const mtmd_input_chunk * ch = mtmd_input_chunks_get(chunks, c);
        const bool last = c + 1 == nc;
        llama_pos np = n_past; // text helper adds to *new_n_past
        int rc;
        if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE && file) {
            const size_t n = mtmd_input_chunk_get_n_tokens(ch);
            const int n_embd = llama_model_n_embd_inp(model);
            std::vector<float> e(n * n_embd);
            FILE * f = fopen(file, "rb");
            if (!f || fread(e.data(), sizeof(float), e.size(), f) != e.size() || fgetc(f) != EOF) {
                fprintf(stderr, "%s is not exactly %zu x %d floats\n", file, n, n_embd); return 1;
            }
            fclose(f);
            fprintf(stderr, "image chunk: %zu rows x %d from %s\n", n, n_embd, file);
            rc = mtmd_helper_decode_image_chunk(mctx, lctx, ch, e.data(), n_past, 0, cp.n_batch, &np, nullptr, nullptr);
        } else {
            if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t n = 0; const llama_token * tk = mtmd_input_chunk_get_tokens_text(ch, &n);
                fprintf(stderr, "text chunk:"); for (size_t i = 0; i < n; i++) fprintf(stderr, " %d", tk[i]); fprintf(stderr, "\n");
            }
            rc = mtmd_helper_eval_chunk_single(mctx, lctx, ch, n_past, 0, cp.n_batch, last, &np);
        }
        if (rc != 0) { fprintf(stderr, "chunk %zu failed: %d\n", c, rc); return 1; }
        n_past = np;
    }
    fprintf(stderr, "n_past after prompt: %d\n", n_past);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int nv = llama_vocab_n_tokens(vocab);
    llama_batch b = llama_batch_init(1, 0, 1);
    printf("{\"gen\":[");
    for (int g = 0; g < n_gen; g++) {
        const float * lg = llama_get_logits_ith(lctx, -1);
        std::vector<int> ix(nv);
        for (int i = 0; i < nv; i++) ix[i] = i;
        std::partial_sort(ix.begin(), ix.begin() + 20, ix.end(), [&](int a, int z) { return lg[a] > lg[z]; });
        if (g == 0) {
            fprintf(stderr, "top:");
            for (int i = 0; i < 20; i++) fprintf(stderr, " [%d,%.4f]", ix[i], lg[ix[i]]);
            fprintf(stderr, "\n");
        }
        const llama_token t = ix[0];
        printf("%s%d", g ? "," : "", t);
        if (llama_vocab_is_eog(vocab, t)) break;
        b.n_tokens = 1; b.token[0] = t; b.pos[0] = n_past++; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
        if (llama_decode(lctx, b) != 0) { fprintf(stderr, "decode failed\n"); return 1; }
    }
    printf("]}\n");
    llama_batch_free(b);
    mtmd_input_chunks_free(chunks);
    mtmd_bitmap_free(bw.bitmap);
    mtmd_free(mctx);
    llama_free(lctx);
    llama_model_free(model);
    return 0;
}
