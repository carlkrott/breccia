// Quantised row gather + dequantise.
//
// The embedding paths of the model are row lookups straight out of a packed
// table, and both tables are too large to copy:
//
//   * `token_embd.weight`        [2560, 248320]  IQ3_S   260 MiB  (always resident)
//   * `per_layer_token_embd`     [160, 320001536] IQ4_NL  26.8 GiB (on demand)
//
// (`docs/specs/qwen4exp-architecture.md` §6.5, §9.1; `docs/phase-2/02-format-loader.md`
// §2 — both are `&[u8]` slices of the read-only `breccia_format` mapping.)
//
// One **subgroup per output row**, so a batch of `rows` is one dispatch and the
// weight traffic is exactly `rows * row_bytes`. Row selection happens through
// `IX[]`, an index list of `rows` u32 table-row numbers:
//
//   * the **staging** path passes the rows it wants gathered (the host copies
//     only those rows out of the mmap, `rows * row_bytes` bytes) and `IX` is the
//     identity — that is how the 26.8 GiB PLE table is read;
//   * the **resident** path binds the whole table once and passes the token ids
//     as `IX` — that is how the 260 MiB `token_embd` is read after it is cached.
//
// Both are one code path, and neither copies the table.
//
// LANE MAPPING is the decode kernels' 8-element unit contract, unchanged: a
// 32-lane subgroup walks one row, lane `l` owns unit `l` of the block it is on,
// i.e. 8 consecutive elements `[8l, 8l+8)`. For a 256-element super-block that
// is one unit per lane and a 512 B contiguous read per subgroup; for the narrow
// blocks (IQ4_NL 32/18, Q2_0 64/18, the virtual 8-element F16/BF16/F32) a
// subgroup covers `32/UNITS` blocks per iteration, which keeps the same
// coalescing. The first version of this kernel gave one *thread* a whole block
// and measured 25 GB/s at 256 rows (each thread striding through its own 110 B
// block); the subgroup layout is what `docs/phase-3/01` measured at 271-480 GB/s
// for the same decode units, and `docs/phase-3/04` records the before/after.
//
// Output is `rows * bpr * QK` floats (each output row padded to a whole block,
// exactly like `gemv::dequant_to_f32`); `padded_cols() == cols` for both tables
// above (2560 = 10 IQ3_S blocks, 160 = 5 IQ4_NL blocks).
//
// The decode itself is `blocks.glsl`'s `DEQUANT_UNIT`, i.e. the same unit the
// fused GEMV and the standalone dequantiser use, so a passing gather test covers
// the same integer/scale arithmetic those do.
//
// The declared `X` (binding 1) and `T` (binding 3) buffers come from
// `blocks.glsl`'s shared header: `X` is unused here (bind a 16-byte dummy, as
// `dequant_to_f32` already does) and `T` is the quant table buffer, which is
// required by the IQ decode units.

#ifndef BRECCIA_OPS_GATHER_GLSL
#define BRECCIA_OPS_GATHER_GLSL

layout(local_size_x = 256) in;

/// Table row number for each output row.
layout(std430, binding = 4) readonly buffer RowIdx { uint IX[]; };

#define LANES 32u
#define UNITS (QK / 8u)
/// Blocks one subgroup covers per iteration (>= 1; 1 for a 256-element block).
#define BPI (UNITS >= LANES ? 1u : (LANES / UNITS))
/// Rows (one subgroup each) per 256-thread workgroup.
#define ROWS_PER_WG (256u / LANES)
#define WG_X_MAX 65535u

void main() {
    uint lane = gl_LocalInvocationID.x % LANES;
    uint sub = gl_LocalInvocationID.x / LANES;
    uint wg = gl_WorkGroupID.y * WG_X_MAX + gl_WorkGroupID.x;
    uint row = wg * ROWS_PER_WG + sub;
    if (row >= pc.rows) {
        return;
    }
    uint src = IX[row];
    uint u = UNITS >= LANES ? lane : (lane % UNITS);
    uint b_off = UNITS >= LANES ? 0u : (lane / UNITS);
    uint out_row = row * pc.bpr * QK;

    float v[8];
    for (uint b0 = 0u; b0 < pc.bpr; b0 += BPI) {
        uint b = b0 + b_off;
        if (b >= pc.bpr) {
            break;
        }
        DEQUANT_UNIT((src * pc.bpr + b) * BLOCK_WORDS, u, v);
        uint e0 = b * QK + u * 8u;
        uint o = out_row + e0;
        if (e0 + 8u <= pc.cols) {
            for (uint j = 0u; j < 8u; ++j) {
                O[o + j] = v[j];
            }
        } else {
            // Trailing partial block (a padding column, not a whole block).
            for (uint j = 0u; j < 8u; ++j) {
                if (e0 + j < pc.cols) {
                    O[o + j] = v[j];
                }
            }
        }
    }
}

#endif // BRECCIA_OPS_GATHER_GLSL
