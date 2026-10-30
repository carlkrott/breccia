// Tiled build of `gemm_body.glsl` (`-DBRECCIA_GEMM_TILE=1 -DTILE_RM=n -DTILE_RN=n`).
//
// The GEMV-shaped kernels re-decode every weight block once per 8 users and re-read
// X once per row group. Here a 256-thread workgroup owns a TM x TN output tile
// (TM = 16*RM rows, TN = 16*RN users) and walks K in steps of BK = 32 elements:
//   1. the TM x BK weight slice is decoded ONCE into LDS (fp32, [row][k], one
//      DEQUANT_UNIT per thread per 8 elements), the TN x BK X slice is copied to LDS;
//   2. every thread accumulates an RM x RN register tile from LDS (vec4 along k:
//      RM + RN LDS reads feed 4*RM*RN FMAs).
// So the decode cost is O(rows * cols) per TN users and the inner loop is LDS-fed FMAs.
// Same bindings / push constants / output layout `O[u * rows + row]` as the GEMV-shaped
// kernels (users in `pc.splits`, `pc.upz` ignored). NOT bit-identical (summation order
// differs): `tests/gemm_many.rs::tiled_gemm_matches_plain` bounds the error.
// Host contract: dispatch (ceil(rows / TM), ceil(users / TN), 1); any rows / users / cols.

layout(local_size_x = 256) in;

#define pc_users pc.splits
#define ROW_BASE_WORDS pc.base

#define RM TILE_RM
#define RN TILE_RN
#define TM (RM * 16u)
#define TN (RN * 16u)
#define BK 32u
#define UNITS (QK / 8u)
// vec4 per LDS row: BK/4 = 8 plus one pad so 8 consecutive rows hit 8 distinct bank quads.
#define KS 9u

shared vec4 Ws[TM * KS];
shared vec4 Xs[TN * KS];

void main() {
    const uint tid = gl_LocalInvocationID.x;
    const uint tx = tid & 15u;
    const uint ty = tid >> 4u;
    const uint row0 = gl_WorkGroupID.x * TM;
    const uint user0 = gl_WorkGroupID.y * TN;
    const uint users = max(pc_users, 1u);
    const uint cols = pc.cols;
    const uint bpr = pc.bpr;
    const uint row_words = bpr * BLOCK_WORDS;
    const uint nunits = bpr * UNITS;
    const uint steps = (bpr * QK + BK - 1u) / BK;
    const bool x4 = (cols & 3u) == 0u;

    float acc[RM][RN];
    [[unroll]] for (uint i = 0u; i < RM; ++i) {
        [[unroll]] for (uint j = 0u; j < RN; ++j) {
            acc[i][j] = 0.0;
        }
    }

    for (uint s = 0u; s < steps; ++s) {
        const uint kb = s * BK;
        // Decode the weight slice: thread -> (row r, unit j of 4), consecutive threads
        // consecutive rows so the LDS vec4 stores are conflict-free.
        for (uint t = tid; t < TM * 4u; t += 256u) {
            uint r = t % TM;
            uint j = t / TM;
            uint g = kb / 8u + j;
            vec4 a = vec4(0.0);
            vec4 b = vec4(0.0);
            if (g < nunits) {
                uint row = min(row0 + r, pc.rows - 1u);
                float v[8];
                DEQUANT_UNIT(ROW_BASE_WORDS + row * row_words + (g / UNITS) * BLOCK_WORDS, g % UNITS, v);
                a = vec4(v[0], v[1], v[2], v[3]);
                b = vec4(v[4], v[5], v[6], v[7]);
            }
            Ws[r * KS + 2u * j] = a;
            Ws[r * KS + 2u * j + 1u] = b;
        }
        // X slice: 8 vec4 (128 B contiguous) per user; k >= cols and users >= m read as 0.
        for (uint t = tid; t < TN * 8u; t += 256u) {
            uint ul = t >> 3u;
            uint k4 = t & 7u;
            uint gu = user0 + ul;
            uint k = kb + k4 * 4u;
            vec4 x = vec4(0.0);
            if (gu < users) {
                if (x4) {
                    if (k < cols) {
                        x = X[(gu * cols + k) >> 2u];
                    }
                } else {
                    for (uint c = 0u; c < 4u; ++c) {
                        if (k + c < cols) {
                            uint idx = gu * cols + k + c;
                            x[c] = X[idx >> 2u][idx & 3u];
                        }
                    }
                }
            }
            Xs[ul * KS + k4] = x;
        }
        barrier();
        [[unroll]] for (uint k4 = 0u; k4 < 8u; ++k4) {
            vec4 w[RM];
            vec4 x[RN];
            [[unroll]] for (uint i = 0u; i < RM; ++i) {
                w[i] = Ws[(tx + 16u * i) * KS + k4];
            }
            [[unroll]] for (uint j = 0u; j < RN; ++j) {
                x[j] = Xs[(ty + 16u * j) * KS + k4];
            }
            [[unroll]] for (uint i = 0u; i < RM; ++i) {
                [[unroll]] for (uint j = 0u; j < RN; ++j) {
                    acc[i][j] = fma(w[i].x, x[j].x, acc[i][j]);
                    acc[i][j] = fma(w[i].y, x[j].y, acc[i][j]);
                    acc[i][j] = fma(w[i].z, x[j].z, acc[i][j]);
                    acc[i][j] = fma(w[i].w, x[j].w, acc[i][j]);
                }
            }
        }
        barrier();
    }

    [[unroll]] for (uint j = 0u; j < RN; ++j) {
        uint u = user0 + ty + 16u * j;
        if (u < users) {
            [[unroll]] for (uint i = 0u; i < RM; ++i) {
                uint row = row0 + tx + 16u * i;
                if (row < pc.rows) {
                    O[u * pc.rows + row] = acc[i][j];
                }
            }
        }
    }
}
