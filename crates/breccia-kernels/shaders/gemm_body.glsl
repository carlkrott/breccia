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

#ifdef BRECCIA_GEMM_RPL
#include "gemm_rpl.glsl"
#else

// BRECCIA_GEMM_PAIR (§11l item 4): `W` holds `pc.rows` gate rows then `pc.rows` up
// rows of one type (`w_concat`); each row decodes both, shares the X loads, keeps
// both accumulator pairs in the single-kernel order and writes `silu(g) * u` (the
// `ops_swiglu` expression), so one dispatch replaces gate + up + swiglu.
#ifdef BRECCIA_GEMM_PAIR
#include "ops_math.glsl"
#endif

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
    uint upz = pc.upz == 0u ? USERS_MAX : min(pc.upz, USERS_MAX);
    uint u0 = gl_WorkGroupID.z * upz;
    uint m = min(max(pc_users, 1u) - u0, upz);

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
#ifdef BRECCIA_GEMM_PAIR
    const uint up_base = ROW_BASE_WORDS + (pc.rows + row) * bpr * BLOCK_WORDS;
    float accu0[USERS_MAX];
    float accu1[USERS_MAX];
    for (uint j = 0u; j < m; ++j) {
        accu0[j] = 0.0;
        accu1[j] = 0.0;
    }
    float w[8];
#endif

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
#ifdef BRECCIA_GEMM_PAIR
        DEQUANT_UNIT(up_base + b * BLOCK_WORDS, u, w);
        vec4 w0 = vec4(w[0], w[1], w[2], w[3]);
        vec4 w1 = vec4(w[4], w[5], w[6], w[7]);
#endif
        for (uint j = 0u; j < m; ++j) {
            uint xb = ((u0 + j) * pc.cols + e) >> 2u;
            float s;
#ifdef BRECCIA_GEMM_PAIR
            float su;
#endif
            if (b < full) {
                vec4 xa = X[xb];
                vec4 xc = X[xb + 1u];
                s = dot(v0, xa) + dot(v1, xc);
#ifdef BRECCIA_GEMM_PAIR
                su = dot(w0, xa) + dot(w1, xc);
#endif
            } else {
                s = 0.0;
#ifdef BRECCIA_GEMM_PAIR
                su = 0.0;
#endif
                for (uint t = 0u; t < 8u; ++t) {
                    uint idx = e + t;
                    if (idx < pc.cols) {
                        // element `idx & 3` of the vec4 word; X is the same
                        // zero-padded buffer the single-user kernels read.
                        float x = X[((u0 + j) * pc.cols + idx) >> 2u][idx & 3u];
                        s += v[t] * x;
#ifdef BRECCIA_GEMM_PAIR
                        su += w[t] * x;
#endif
                    }
                }
            }
            if ((b0 & 1u) == 0u) {
                acc0[j] += s;
#ifdef BRECCIA_GEMM_PAIR
                accu0[j] += su;
#endif
            } else {
                acc1[j] += s;
#ifdef BRECCIA_GEMM_PAIR
                accu1[j] += su;
#endif
            }
        }
    }

    for (uint j = 0u; j < m; ++j) {
        float facc = acc0[j] + acc1[j];
        [[unroll]] for (uint off = 16u; off > 0u; off >>= 1u) {
            facc += subgroupShuffleXor(facc, off);
        }
#ifdef BRECCIA_GEMM_PAIR
        float fu = accu0[j] + accu1[j];
        [[unroll]] for (uint off = 16u; off > 0u; off >>= 1u) {
            fu += subgroupShuffleXor(fu, off);
        }
        if (lane == 0u) {
            O[(u0 + j) * pc.rows + row] = breccia_silu(facc) * fu;
        }
#else
        if (lane == 0u) {
            O[(u0 + j) * pc.rows + row] = facc;
        }
#endif
    }
}

#endif // BRECCIA_GEMM_RPL
#endif // BRECCIA_GEMM_BODY_GLSL
