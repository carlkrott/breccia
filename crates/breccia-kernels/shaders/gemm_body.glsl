// Multi-user fused dequant + matrix-vector, for one expert and its `m` users.
//
// This is the piece every current kernel lacks: `gemv_body.glsl` decodes a weight
// block once per *row* and dots it against one activation vector, so a chunk of `m`
// tokens that share an expert pays the decode `m` times and reads the weights `m`
// times. Here the block is decoded **once into registers** and dotted with all `m`
// users before moving on, so the decode and the DRAM traffic are amortised over the
// users (`docs/phase-4/01-forward-pass.md` §8v).
//
// Layout (the `down` role: `rows` = n_embd, `cols` = n_ff):
//   W  one expert's quantised rows, `rows * bpr * BLOCK_WORDS` u16 words
//   X  `m` activation rows, `m * cols` f32 (row-major)
//   O  `m * rows` f32, user-major: `O[u * rows + row]`
//
// Geometry matches `gemv_body.glsl` exactly: one 32-lane group per output row,
// ROWS_PER_WG rows per workgroup, lane `l` owns the 8 elements `[l*8, l*8+8)` of
// each block, two alternating fp32 accumulators per user, and the same subgroup
// shuffle reduction. With `m == 1` that makes the arithmetic **identical** to the
// single-user kernel — same decodes, same accumulation order, same reduction — so
// the m = 1 case is bit-comparable with the existing GEMV (asserted in
// `tests/gemm_many.rs`).

#ifndef BRECCIA_GEMM_BODY_GLSL
#define BRECCIA_GEMM_BODY_GLSL

layout(local_size_x = 256) in;

// `W` (u16 words, binding 0), `X` (vec4, binding 1), `O` (binding 2), the quant
// tables (binding 3) and the push constants `pc{rows, cols, bpr, splits}` all come
// from `blocks.glsl`; the **user count rides in `pc.splits`** because a GEMM
// dispatch never splits K.
#define pc_users pc.splits
// `pc.base` (the 5th push-constant word, declared in blocks.glsl) is the arena
// offset of this expert's blob in 16-bit words: the resident arena holds all 512
// experts, so a GEMM addresses one of them exactly as the `MoeMap`/`GemvBatch` path
// does. The single-expert callers push 0.
#define ROW_BASE_WORDS pc.base

#define LANES 32u
#define ROWS_PER_WG 8u
#define WG_X_MAX 65535u
#define UNITS (QK / 8u)
#define STEP (UNITS >= LANES ? 1u : (LANES / UNITS))
/// Users one invocation can hold in registers (two accumulators each).
#define USERS_MAX 8u

void main() {
    uint lane = gl_LocalInvocationID.x % LANES;
    uint wg = gl_WorkGroupID.y * WG_X_MAX + gl_WorkGroupID.x;
    uint row = wg * ROWS_PER_WG + gl_LocalInvocationID.x / LANES;
    if (row >= pc.rows) {
        return;
    }
    // Users past USERS_MAX ride the z dimension: workgroup z owns users
    // [z*USERS_MAX, z*USERS_MAX + m). One-group dispatches (z = 0) are unchanged.
    uint u0 = gl_WorkGroupID.z * USERS_MAX;
    uint m = min(max(pc_users, 1u) - u0, USERS_MAX);

    const uint u = UNITS >= LANES ? lane : (lane % UNITS);
    const uint b_off = UNITS >= LANES ? 0u : (lane / UNITS);
    const uint bpr = pc.bpr;
    const uint full = pc.cols / QK;
    const uint row_base = ROW_BASE_WORDS + row * bpr * BLOCK_WORDS;

    float acc0[USERS_MAX];
    float acc1[USERS_MAX];
    for (uint j = 0u; j < m; ++j) {
        acc0[j] = 0.0;
        acc1[j] = 0.0;
    }
    float v[8];

    for (uint b0 = 0u; b0 < bpr; b0 += STEP) {
        uint b = b0 + b_off;
        if (b >= bpr) {
            continue;
        }
        uint e = b * QK + u * 8u;
        // One decode for every user of this block.
        DEQUANT_UNIT(row_base + b * BLOCK_WORDS, u, v);
        vec4 v0 = vec4(v[0], v[1], v[2], v[3]);
        vec4 v1 = vec4(v[4], v[5], v[6], v[7]);
        for (uint j = 0u; j < m; ++j) {
            uint xb = ((u0 + j) * pc.cols + e) >> 2u;
            float s;
            if (b < full) {
                s = dot(v0, X[xb]) + dot(v1, X[xb + 1u]);
            } else {
                s = 0.0;
                for (uint t = 0u; t < 8u; ++t) {
                    uint idx = e + t;
                    if (idx < pc.cols) {
                        // element `idx & 3` of the vec4 word; X is the same
                        // zero-padded buffer the single-user kernels read.
                        s += v[t] * X[((u0 + j) * pc.cols + idx) >> 2u][idx & 3u];
                    }
                }
            }
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

#endif // BRECCIA_GEMM_BODY_GLSL
