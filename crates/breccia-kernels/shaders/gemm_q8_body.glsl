// Multi-user GEMM on **packed int8 dot products** (§11j): `gemm_body.glsl`'s
// geometry (one 32-lane group per row, 8 rows per WG, users in registers, users
// past USERS_MAX in z) with the activation read as q8_1 (`ops_q8_quantize`: int8 +
// one f16 scale per 32 elements) instead of fp32.
//
// Why: at m = 8 the fp32 body is bound by re-reading X (2 x vec4 per lane per
// block per user, ~6.5 MB per user for a 640 x 2560 expert), not by the weights
// or the dispatch. q8 X is 4x smaller and `dotPacked4x8EXT` does 4 MACs per slot.
//
// The including shader defines QK, BLOCK_WORDS, includes blocks.glsl, and defines
// `DEQUANT_UNIT_I8(off, u, c0, c1, d)`: the unit's 8 weights as two packed int8x4
// words (signs applied) and the one float scale they share. Requires
// `cols % QK == 0` (the host checks): no tail block.

#ifndef BRECCIA_GEMM_Q8_BODY_GLSL
#define BRECCIA_GEMM_Q8_BODY_GLSL

layout(local_size_x = 256) in;

layout(std430, binding = 4) readonly buffer Q8Vec { uint QW[]; };      // 4 x int8 per word
layout(std430, binding = 5) readonly buffer DQVec { uint16_t DQ[]; };  // f16 per 32 elements

#define pc_users pc.splits
#define ROW_BASE_WORDS pc.base
#define LANES 32u
#define ROWS_PER_WG 8u
#define WG_X_MAX 65535u
#define UNITS (QK / 8u)
#define STEP (UNITS >= LANES ? 1u : (LANES / UNITS))
#define USERS_MAX 8u

void main() {
    uint lane = gl_LocalInvocationID.x % LANES;
    uint wg = gl_WorkGroupID.y * WG_X_MAX + gl_WorkGroupID.x;
    uint row = wg * ROWS_PER_WG + gl_LocalInvocationID.x / LANES;
    if (row >= pc.rows) {
        return;
    }
    uint upz = pc.upz == 0u ? USERS_MAX : min(pc.upz, USERS_MAX);
    uint u0 = gl_WorkGroupID.z * upz;
    uint m = min(max(pc_users, 1u) - u0, upz);

    const uint u = UNITS >= LANES ? lane : (lane % UNITS);
    const uint b_off = UNITS >= LANES ? 0u : (lane / UNITS);
    const uint bpr = pc.bpr;
    const uint row_base = ROW_BASE_WORDS + row * bpr * BLOCK_WORDS;
    const uint xwords = pc.cols / 4u;  // q8 words per user
    const uint xgroups = pc.cols / 32u; // f16 scales per user

    float acc0[USERS_MAX];
    float acc1[USERS_MAX];
    for (uint j = 0u; j < m; ++j) {
        acc0[j] = 0.0;
        acc1[j] = 0.0;
    }

    for (uint b0 = 0u; b0 < bpr; b0 += STEP) {
        uint b = b0 + b_off;
        if (b >= bpr) {
            continue;
        }
        uint e = b * QK + u * 8u; // first element; e % 8 == 0, so one 32-group
        int c0;
        int c1;
        float d;
        DEQUANT_UNIT_I8(row_base + b * BLOCK_WORDS, u, c0, c1, d);
        for (uint j = 0u; j < m; ++j) {
            uint ux = u0 + j;
            uint xw = ux * xwords + (e >> 2u);
            int s8 = dotPacked4x8EXT(c0, int(QW[xw])) + dotPacked4x8EXT(c1, int(QW[xw + 1u]));
            float dq = unpackHalf2x16(uint(DQ[ux * xgroups + (e >> 5u)])).x;
            float s = dq * d * float(s8);
            if ((b0 & 1u) == 0u) {
                acc0[j] += s;
            } else {
                acc1[j] += s;
            }
        }
    }

    for (uint j = 0u; j < m; ++j) {
        float facc = acc0[j] + acc1[j];
        [[unroll]] for (uint off = 16u; off > 0u; off >>= 1u) {
            facc += subgroupShuffleXor(facc, off);
        }
        if (lane == 0u) {
            O[(u0 + j) * pc.rows + row] = facc;
        }
    }
}

#endif // BRECCIA_GEMM_Q8_BODY_GLSL
