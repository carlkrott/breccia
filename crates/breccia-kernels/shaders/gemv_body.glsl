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
// fp32 accumulation by default (`BRECCIA_F16_ACCUM` builds the opt-in fp16
// variant), an in-loop tail mask for cols % QK != 0, and the split-K path.

#ifndef BRECCIA_GEMV_BODY_GLSL
#define BRECCIA_GEMV_BODY_GLSL

#ifdef BRECCIA_GEMV_EPILOGUE_SILU_DIV
#include "ops_math.glsl"
#endif

#ifdef BRECCIA_HC_GATED_EPILOGUE
// Fused HC `up` + gated mean + inject partials (§8az). The GEMV half is the
// ordinary one-token dot product; only the *row mapping* and the *epilogue*
// differ, which is why this is a build variant of the same shader rather than a
// new kernel (`AGENTS.md` § "Adding a kernel", lowest-risk class: same bindings,
// same resources, an added entry point's worth of code).
#include "ops_math.glsl"
#ifndef BRECCIA_OWN_BINDINGS
layout(std430, binding = 4) readonly buffer XnVec { float XN[]; };      // [hc*n_embd]
layout(std430, binding = 5) readonly buffer InjWVec { uint16_t WINJ[]; }; // [hc, hc*n_embd] BF16
layout(std430, binding = 6) writeonly buffer PartVec { float P[]; };   // [hc, chunks]
layout(std430, binding = 7) writeonly buffer MixVec { float MIX[]; };  // [n_embd]
#endif
/// BF16 at element index `i`. `WINJ` is a **16-bit** array (one element per
/// index), unlike `ops_hc_gated_mean.comp`'s `uint[]` word view — reading it as a
/// word and shifting by `i & 1` would read every odd element as zero.
float hc_gated_bf16(uint i) {
    return uintBitsToFloat(uint(WINJ[i]) << 16);
}
#endif

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

#ifdef BRECCIA_HC_GATED_EPILOGUE
// One slot per row of the workgroup: slot `s` holds `gate[c][d]` for
// (c = s / per, d = per*wg + s % per).
shared float s_hcgate[ROWS_PER_WG];

/// The gated mean, the `gate` store and the `inject` partials, all from the values
/// the workgroup just computed.
///
/// * `mixed[d] = (1/hc) * sum_c xn[c][d] * sigmoid(gate[c][d])` — **the same
///   four-term order and the same divide** as `ops_hc_gated_mean.comp`, so this
///   value is bit-identical to the separate dispatch.
/// * `O` still receives `gate` in its row-major layout, so the workspace keeps its
///   invariant (nothing on the hot path reads it after the fold).
/// * `inject` partials: the chunk is this workgroup's `d` range (2 elements at
///   hc = 4), so the number of partials rises from `n_embd/256` to `n_embd/2` and
///   the summation *grouping* changes — the only non-bit-identical part of this
///   fusion (measured and bounded in `docs/phase-4/01-forward-pass.md` §8az).
void hc_gated_epilogue(uint wg, uint slot) {
    const uint per = ROWS_PER_WG / ((pc.hc == 0u) ? 1u : pc.hc);
    const uint lid = gl_LocalInvocationID.x;
    barrier();
    // Keep `gate` in the workspace (row-major [hc*n_embd]).
    if (lid < ROWS_PER_WG) {
        uint c = lid / per;
        uint j = lid - c * per;
        if (c < pc.hc) {
            O[(c * pc.n_embd) + (per * wg + j)] = s_hcgate[lid];
        }
    }
    // The gated mean, one thread per `d` of this workgroup.
    if (lid < per) {
        uint d = per * wg + lid;
        float acc = 0.0;
        for (uint c = 0u; c < pc.hc; ++c) {
            acc += XN[c * pc.n_embd + d] * breccia_sigmoid(s_hcgate[c * per + lid]);
        }
        MIX[d] = acc / float(pc.hc);
    }
    // inject[c_out][chunk] = sum_{c_src, d in this workgroup} xn[c_src][d] * w[c_out][c_src][d]
    // One thread per output stream, walking the workgroup's `per` values of `d`
    // (the whole chunk, so the partial is complete before it is stored — two
    // threads writing the same slot would each contribute half the sum).
    const uint d_begin = per * wg;
    const uint d_end = d_begin + per;
    if (lid < pc.hc) {
        float s = 0.0;
        for (uint d = d_begin; d < d_end; ++d) {
            for (uint cs = 0u; cs < pc.hc; ++cs) {
                uint at = lid * (pc.hc * pc.n_embd) + cs * pc.n_embd + d;
                s += XN[cs * pc.n_embd + d] * hc_gated_bf16(at);
            }
        }
        P[lid * pc.chunks + wg] = s;
    }
}
#endif

void main() {
#ifdef BRECCIA_WORD_PACK
    // Fill the LDS codebook once per workgroup (see `blocks.glsl`). Ahead of every
    // early return, so the barrier is uniform across the workgroup.
    for (uint i = gl_LocalInvocationID.x; i < 16u; i += 32u) {
        s_iq4nl[i] = float(int(T[TBL_IQ4NL_VALUES + i]));
    }
    barrier();
#endif
    uint lane = gl_LocalInvocationID.x % LANES;
    uint wg = gl_WorkGroupID.y * WG_X_MAX + gl_WorkGroupID.x;
    uint slot = gl_LocalInvocationID.x / LANES;
#ifdef BRECCIA_HC_GATED_EPILOGUE
    // **Column-blocked row mapping.** The gated mean needs all `hc` streams at one
    // `d`; the row-major mapping puts them 2560 rows apart, one per workgroup.
    // Here workgroup `wg` owns `ROWS_PER_WG / hc` consecutive `d` and **all** `hc`
    // streams, so slot `s` is (stream `s / per`, `d = per*wg + s % per`). Same
    // workgroup count, same warps per row, same K-walk — only which row a warp
    // owns changes, which is why the main loop below is untouched.
    const uint per = ROWS_PER_WG / ((pc.hc == 0u) ? 1u : pc.hc);
    uint c_s = slot / per;
    uint j_s = slot - c_s * per;
    if (c_s >= pc.hc) {
        return;
    }
    uint d_s = per * wg + j_s;
    uint row = c_s * pc.n_embd + d_s;
#else
    uint row = wg * ROWS_PER_WG + slot;
    if (row >= pc.rows) {
        return;
    }
#endif

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
    // A tail-skipped expert row (`moe_map.comp` marker): keep the control flow
    // uniform (same reduction, same store) but walk no blocks, so it writes 0.
#ifdef BRECCIA_EXPERT_MAP
    const uint b_stop = row_base == 0xFFFFFFFFu ? b_start : b_end;
#else
    const uint b_stop = b_end;
#endif

#ifdef BRECCIA_F16_ACCUM
    // fp16 accumulation (opt-in, `Accum::F16`; docs/phase-3/03-gaps-closed.md).
    // Two accumulators packed as one f16vec2 so the pair is a single packed
    // f16 add (V_PK_ADD_F16 on gfx9/gfx10) instead of two scalar adds. Each
    // block's 8-element partial product is still formed in f32; the running sum
    // over the row is the part that rounds to fp16.
    f16vec2 acc = f16vec2(float16_t(0.0), float16_t(0.0));
#else
    float acc0 = 0.0;
    float acc1 = 0.0;
#endif
    float v[8];

    for (uint b0 = b_start; b0 < b_stop; b0 += STEP) {
        uint b = b0 + b_off;
        if (b < b_start || b >= b_stop) {
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
#ifdef BRECCIA_F16_ACCUM
        acc += ((b0 & 1u) == 0u) ? f16vec2(float16_t(s), float16_t(0.0))
                                 : f16vec2(float16_t(0.0), float16_t(s));
#else
        if ((b0 & 1u) == 0u) {
            acc0 += s;
        } else {
            acc1 += s;
        }
#endif
    }

#ifdef BRECCIA_F16_ACCUM
    // Convert to f32 before the cross-lane reduce: the fp16 rounding this path
    // measures is the per-lane running sum, and keeping the 5 shuffle-adds in
    // f32 avoids needing 16-bit subgroup shuffles (not universally supported).
    f16vec2 hsum = acc;
    float facc = float(hsum.x + hsum.y);
#else
    float facc = acc0 + acc1;
#endif
    [[unroll]] for (uint off = 16u; off > 0u; off >>= 1u) {
        facc += subgroupShuffleXor(facc, off);
    }
    if (lane == 0u) {
#ifdef BRECCIA_HC_GATED_EPILOGUE
        s_hcgate[slot] = facc;
#else
#ifdef BRECCIA_GEMV_EPILOGUE_SILU_DIV
        // The bottleneck activation as the GEMV's epilogue: `silu(acc / s)` on
        // exactly the value a separate `ops_act.comp` dispatch would read back
        // (`s` = the stream count), after the full accumulation and the
        // cross-lane reduce. Bit-identical by construction — same arithmetic,
        // same order, one fewer dispatch per call.
        facc = breccia_silu(facc / pc.ep);
#endif
        O[sid * pc.rows + row] = facc;
#endif
    }
#ifdef BRECCIA_HC_GATED_EPILOGUE
    hc_gated_epilogue(wg, slot);
#endif
}

#endif // BRECCIA_GEMV_BODY_GLSL
