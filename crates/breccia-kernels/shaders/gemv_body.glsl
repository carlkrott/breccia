// Fused dequant + mat-vector multiply for batch-1 (decode) inference.
//
// One 32-lane group per output row, ROWS_PER_WG rows per workgroup, and an
// optional split of each row's blocks across `splits` workgroups (split-K) so
// small matrices still fill the GPU: a 640-row expert matrix is only 80
// workgroups, i.e. 1.3 per CU on the RX 6950 XT, and measures ~100 GB/s;
// splitting each row 3-4 ways lifts the same shape to ~300 GB/s
// (docs/phase-3/01-dequant-gemv.md §4).
//
// Lane `l` owns unit `l` of the QK-element blocks it walks, i.e. the 8
// consecutive elements [l*8, l*8+8) — so a 256-element super-block is covered
// by exactly one warp iteration and every weight byte of a row is read exactly
// once. Accumulation is fp32 (two alternating accumulators to keep two FMA
// chains in flight) and the reduction across the 32-lane group uses shuffles,
// which is portable across Wave32 (RX 6950 XT, gfx1030) and Wave64 (Radeon
// VII/MI50, gfx906): an xor-shuffle with offsets <= 16 never crosses an aligned
// 32-lane boundary.
//
// Output: with splits == 1 the row result goes straight to `O[row]`. With
// splits > 1 each split writes `O[sid * rows + row]` and `gemv_reduce.comp`
// sums the partials (coalesced), which is cheaper than atomics (this device has
// no float storage-buffer atomics enabled).
//
// Adapted from ggml-vulkan (MIT) vulkan-shaders/mul_mat_vec_iq3_s.comp (the
// "one warp covers a super-block, per-lane 8-element unit" shape), commit
// 6050b46. Differences: raw u16-word block views instead of typed structs,
// fp32-only accumulation, an in-loop tail mask for cols % QK != 0, and the
// split-K path.

#ifndef BRECCIA_GEMV_BODY_GLSL
#define BRECCIA_GEMV_BODY_GLSL

layout(local_size_x = 256) in;

#define LANES 32u
#define ROWS_PER_WG 8u
// maxComputeWorkGroupCount[0] is only guaranteed to be 65535, and a 248320-row
// matrix already needs 31040 workgroups (1.3 M rows in the whole-expert-tensor
// benchmark needs 163840). Wrap the row groups over grid.y so the dispatch stays
// inside the guaranteed limit on any Vulkan device.
#define WG_X_MAX 65535u
#define UNITS (QK / 8u)
#define STEP (UNITS >= LANES ? 1u : (LANES / UNITS))

// --- batched multi-expert dispatch (MoE decode shape) -----------------------
// With BRECCIA_EXPERT_MAP the kernel reads a per-row indirection table instead
// of assuming row r starts at r*bpr blocks: N experts' blobs are concatenated
// in one device buffer (the M3 slot-ring layout) and each row names its blob
// base, so one dispatch covers the layer's whole routed set (10 x 640 rows in
// this model) instead of 10 latency-bound 640-row dispatches
// (docs/phase-3/02-expert-batching-quants.md).
//
//   M[0 .. rows)       row weight base, in 16-bit words into W
//   M[rows .. 2*rows)  row activation base, in vec4 units into X
//
// The activation base is what makes per-expert x-vectors work: the down
// projection consumes a different SwiGLU intermediate per expert, while
// gate/up consume one shared vector (X_BASE = 0 for every row).
#ifdef BRECCIA_EXPERT_MAP
layout(std430, binding = 4) readonly buffer RowMap { uint M[]; };
#define ROW_BASE(r) (M[r])
#define X_BASE(r) (M[pc.rows + (r)])
#else
#define ROW_BASE(r) ((r) * pc.bpr * BLOCK_WORDS)
#define X_BASE(r) 0u
#endif

void main() {
    uint lane = gl_LocalInvocationID.x % LANES;
    uint wg = gl_WorkGroupID.y * WG_X_MAX + gl_WorkGroupID.x;
    uint row = wg * ROWS_PER_WG + gl_LocalInvocationID.x / LANES;
    if (row >= pc.rows) {
        return;
    }

    const uint u = UNITS >= LANES ? lane : (lane % UNITS);
    const uint b_off = UNITS >= LANES ? 0u : (lane / UNITS);
    const uint bpr = pc.bpr;
    const uint full = pc.cols / QK; // complete blocks
    const uint splits = max(pc.splits, 1u);
    const uint sid = splits > 1u ? gl_WorkGroupID.z : 0u;
    // Contiguous block chunk per split: each warp walks a contiguous range, so
    // the DRAM stream stays sequential.
    const uint chunk = (bpr + splits - 1u) / splits;
    const uint b_start = min(chunk * sid, bpr);
    const uint b_end = min(b_start + chunk, bpr);

    const uint row_base = ROW_BASE(row);
    const uint xb = X_BASE(row); // vec4 units

    float acc0 = 0.0;
    float acc1 = 0.0;
    float v[8];

    for (uint b0 = b_start; b0 < b_end; b0 += STEP) {
        uint b = b0 + b_off;
        if (b < b_start || b >= b_end) {
            continue; // only reachable when UNITS < LANES (partial step)
        }
        uint e = b * QK + u * 8u;
        DEQUANT_UNIT(row_base + b * BLOCK_WORDS, u, v);
        float s;
        if (b < full) {
            vec4 xa = X[xb + (e >> 2u)];
            vec4 xc = X[xb + (e >> 2u) + 1u];
            s = dot(vec4(v[0], v[1], v[2], v[3]), xa) +
                dot(vec4(v[4], v[5], v[6], v[7]), xc);
        } else {
            // Trailing partial block: mask per element. `X` is padded by the
            // host to a multiple of 8 elements with zeros, so no reads escape.
            s = 0.0;
            for (uint j = 0u; j < 8u; ++j) {
                uint idx = e + j;
                if (idx < pc.cols) {
                    s += v[j] * X[xb + (idx >> 2u)][idx & 3u];
                }
            }
        }
        if ((b0 & 1u) == 0u) {
            acc0 += s;
        } else {
            acc1 += s;
        }
    }

    float acc = acc0 + acc1;
    [[unroll]] for (uint off = 16u; off > 0u; off >>= 1u) {
        acc += subgroupShuffleXor(acc, off);
    }
    if (lane == 0u) {
        O[sid * pc.rows + row] = acc;
    }
}

#endif // BRECCIA_GEMV_BODY_GLSL
