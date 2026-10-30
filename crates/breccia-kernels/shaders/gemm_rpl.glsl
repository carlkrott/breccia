// Register-blocked build of `gemm_body.glsl` (§12 item 1, `-DBRECCIA_GEMM_RPL=n`).
//
// The single-row kernel issues one X load per FMA, so it runs at the cache's
// load rate (~1 TFLOP/s on the MI50, ~8 % of fp32 peak), not at DRAM or ALU
// speed. Here each 32-lane group owns `RPL` consecutive rows: it decodes the
// block of every row, then loads each user's X words **once** and dots them
// against all `RPL` rows, cutting the X traffic `RPL`-fold.
//
// Per row everything else is the single-row kernel's: the lane -> element map,
// the block order, the two alternating accumulators and the shuffle reduction,
// so every output is bit-identical to `gemm_body.glsl` (asserted in
// `tests/gemm_many.rs`). Host contract: `pc.rows % RPL == 0` (`GemmMany`
// only picks this pipeline then).

layout(local_size_x = 256) in;

#define pc_users pc.splits
#define ROW_BASE_WORDS pc.base

#define LANES 32u
#define ROWS_PER_WG 8u
#define WG_X_MAX 65535u
#define UNITS (QK / 8u)
#define STEP (UNITS >= LANES ? 1u : (LANES / UNITS))
#define USERS_MAX 8u
#define RPL BRECCIA_GEMM_RPL

void main() {
    uint lane = gl_LocalInvocationID.x % LANES;
    uint wg = gl_WorkGroupID.y * WG_X_MAX + gl_WorkGroupID.x;
    uint row0 = (wg * ROWS_PER_WG + gl_LocalInvocationID.x / LANES) * RPL;
    if (row0 >= pc.rows) {
        return;
    }
    uint upz = pc.upz == 0u ? USERS_MAX : min(pc.upz, USERS_MAX);
    uint u0 = gl_WorkGroupID.z * upz;
    uint m = min(max(pc_users, 1u) - u0, upz);

    const uint u = UNITS >= LANES ? lane : (lane % UNITS);
    const uint b_off = UNITS >= LANES ? 0u : (lane / UNITS);
    const uint bpr = pc.bpr;
    const uint full = pc.cols / QK;
    const uint row_words = bpr * BLOCK_WORDS;
    const uint base0 = ROW_BASE_WORDS + row0 * row_words;

    float acc0[RPL][USERS_MAX];
    float acc1[RPL][USERS_MAX];
    [[unroll]] for (uint r = 0u; r < RPL; ++r) {
        for (uint j = 0u; j < m; ++j) {
            acc0[r][j] = 0.0;
            acc1[r][j] = 0.0;
        }
    }
    float v[8];
    float vs[RPL][8];
    vec4 va[RPL];
    vec4 vb[RPL];

    for (uint b0 = 0u; b0 < bpr; b0 += STEP) {
        uint b = b0 + b_off;
        if (b >= bpr) {
            continue;
        }
        uint e = b * QK + u * 8u;
        [[unroll]] for (uint r = 0u; r < RPL; ++r) {
            DEQUANT_UNIT(base0 + r * row_words + b * BLOCK_WORDS, u, v);
            va[r] = vec4(v[0], v[1], v[2], v[3]);
            vb[r] = vec4(v[4], v[5], v[6], v[7]);
            vs[r] = v;
        }
        for (uint j = 0u; j < m; ++j) {
            uint xb = ((u0 + j) * pc.cols + e) >> 2u;
            float s[RPL];
            if (b < full) {
                vec4 xa = X[xb];
                vec4 xc = X[xb + 1u];
                [[unroll]] for (uint r = 0u; r < RPL; ++r) {
                    s[r] = dot(va[r], xa) + dot(vb[r], xc);
                }
            } else {
                [[unroll]] for (uint r = 0u; r < RPL; ++r) {
                    s[r] = 0.0;
                }
                for (uint t = 0u; t < 8u; ++t) {
                    uint idx = e + t;
                    if (idx < pc.cols) {
                        float x = X[((u0 + j) * pc.cols + idx) >> 2u][idx & 3u];
                        [[unroll]] for (uint r = 0u; r < RPL; ++r) {
                            s[r] += vs[r][t] * x;
                        }
                    }
                }
            }
            if ((b0 & 1u) == 0u) {
                [[unroll]] for (uint r = 0u; r < RPL; ++r) {
                    acc0[r][j] += s[r];
                }
            } else {
                [[unroll]] for (uint r = 0u; r < RPL; ++r) {
                    acc1[r][j] += s[r];
                }
            }
        }
    }

    [[unroll]] for (uint r = 0u; r < RPL; ++r) {
        for (uint j = 0u; j < m; ++j) {
            float facc = acc0[r][j] + acc1[r][j];
            [[unroll]] for (uint off = 16u; off > 0u; off >>= 1u) {
                facc += subgroupShuffleXor(facc, off);
            }
            if (lane == 0u) {
                O[(u0 + j) * pc.rows + row0 + r] = facc;
            }
        }
    }
}
