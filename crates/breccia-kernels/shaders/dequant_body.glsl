// Standalone block dequantiser: one invocation per block, QK floats written.
//
// Same decode units as the fused GEMV kernel, so a passing dequant conformance
// test and a passing GEMV conformance test cover the same code path.
//
// `O` holds rows * (bpr * QK) floats when cols % QK != 0 would overrun; the
// host therefore allocates ceil-aligned row storage and reads back only
// `cols` values per row (documented in `dequant_to_f32`).
//
// Adapted from ggml-vulkan (MIT) vulkan-shaders/dequant_funcs.glsl +
// dequant_iq3_s.comp, commit 6050b46.

#ifndef BRECCIA_DEQUANT_BODY_GLSL
#define BRECCIA_DEQUANT_BODY_GLSL

layout(local_size_x = 64) in;

#define UNITS (QK / 8u)
// See gemv_body.glsl: grid.x is capped at the guaranteed 65535 workgroups.
#define WG_X_MAX 65535u

void main() {
    uint wg = gl_WorkGroupID.y * WG_X_MAX + gl_WorkGroupID.x;
    uint blk = wg * 64u + gl_LocalInvocationID.x;
    uint total = pc.rows * pc.bpr;
    if (blk >= total) {
        return;
    }
    uint row = blk / pc.bpr;
    uint b = blk - row * pc.bpr;
    uint off = blk * BLOCK_WORDS;
    uint e0 = b * QK;
    uint out_base = row * pc.bpr * QK + e0;

    float v[8];
    for (uint u = 0u; u < UNITS; ++u) {
        DEQUANT_UNIT(off, u, v);
        for (uint j = 0u; j < 8u; ++j) {
            uint e = e0 + u * 8u + j;
            if (e < pc.cols) {
                O[out_base + u * 8u + j] = v[j];
            }
        }
    }
}

#endif // BRECCIA_DEQUANT_BODY_GLSL
