/* Reference decode driver for tools/refcheck.py.
 *
 * This is deliberately *not* a re-implementation: it includes the reference
 * clone's own `ggml-quants.h` and calls the upstream dequantize_row_* functions
 * over random block bytes, so `refcheck.py` diffs this crate's port against
 * llama.cpp (MIT, commit 6050b46) rather than against a second port.
 *
 *   refcheck_drv <ggml-type-id> <block-bytes> <nblocks>  < blocks  > f32 values
 *
 * Build (done by refcheck.py):
 *   gcc -O2 -ffp-contract=off -I<clone>/ggml/src -I<clone>/ggml/include \
 *       tools/refcheck_drv.c <clone>/ggml/src/ggml-quants.c -o refcheck_drv -lm
 */
#include "ggml-quants.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* stubs for the quantizer helpers this driver never calls */
void ggml_abort(const char *file, int line, const char *fmt, ...);
void ggml_abort(const char *file, int line, const char *fmt, ...) {
    (void)file; (void)line; (void)fmt; abort();
}
size_t ggml_row_size(enum ggml_type type, int64_t ne);
size_t ggml_row_size(enum ggml_type type, int64_t ne) { (void)type; (void)ne; abort(); }
size_t ggml_type_size(enum ggml_type type);
size_t ggml_type_size(enum ggml_type type) { (void)type; abort(); }
const char * ggml_type_name(enum ggml_type type);
const char * ggml_type_name(enum ggml_type type) { (void)type; abort(); }

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: refcheck_drv <type> <block-bytes> <nblocks>\n"); return 2; }
    int ty = atoi(argv[1]);
    size_t bb = (size_t)atol(argv[2]);
    long n = atol(argv[3]);
    if (bb == 0 || n <= 0) { fprintf(stderr, "empty input\n"); return 2; }

    unsigned char *in = malloc(bb * (size_t)n);
    if (!in || fread(in, bb, (size_t)n, stdin) != (size_t)n) { fprintf(stderr, "short read\n"); return 3; }

    size_t ne = 0;
    switch (ty) {
        case 12: case 13: case 14: case 16: case 17: case 18: case 21: case 22: case 23: ne = 256; break;
        case 20: ne = 32; break;
        case 42: ne = 64; break;
        default: fprintf(stderr, "unsupported type %d\n", ty); return 4;
    }

    float *out = malloc(sizeof(float) * ne * (size_t)n);
    switch (ty) {
        case 16: dequantize_row_iq2_xxs((const block_iq2_xxs *)in, out, (int64_t)ne * n); break;
        case 17: dequantize_row_iq2_xs ((const block_iq2_xs  *)in, out, (int64_t)ne * n); break;
        case 18: dequantize_row_iq3_xxs((const block_iq3_xxs *)in, out, (int64_t)ne * n); break;
        case 21: dequantize_row_iq3_s ((const block_iq3_s  *)in, out, (int64_t)ne * n); break;
        case 20: dequantize_row_iq4_nl((const block_iq4_nl *)in, out, (int64_t)ne * n); break;
        case 13: dequantize_row_q5_K  ((const block_q5_K   *)in, out, (int64_t)ne * n); break;
        case 42: dequantize_row_q2_0  ((const block_q2_0   *)in, out, (int64_t)ne * n); break;
        case 22: dequantize_row_iq2_s ((const block_iq2_s  *)in, out, (int64_t)ne * n); break;
        case 23: dequantize_row_iq4_xs((const block_iq4_xs *)in, out, (int64_t)ne * n); break;
        case 12: dequantize_row_q4_K  ((const block_q4_K   *)in, out, (int64_t)ne * n); break;
        case 14: dequantize_row_q6_K  ((const block_q6_K   *)in, out, (int64_t)ne * n); break;
    }
    fwrite(out, sizeof(float), ne * (size_t)n, stdout);
    return 0;
}
