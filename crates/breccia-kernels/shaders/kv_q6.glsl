// q6_0 paged K/V cache (`BRECCIA_KV_Q6`, the default layout since 2026-11-05).
//
// ```text
// row   = (page * n_kv_head + kv_head) * page_cells + cell_in_page
// a row = head_dim/64 superblocks of 13 u32 words (256 -> 52 words = 208 B):
//   [0]            packHalf2x16(d0, d1): the fp16 scales of its two 32-element blocks
//   [1+6s .. 4+6s] block s: 32 low nibbles, 8 per word, element i at bits 4*(i&7)
//   [5+6s, 6+6s]   block s: 32 high 2-bit pairs, 16 per word, element i at 2*(i&15)
// value = (q - 32) * d,   q = nibble | high << 4  in [0, 63]
// ```
//
// The quantiser is ggml's `q6_0`-style rule the fake-quant gate measured
// (`gg_roundtrip(6)` in `qsa_kv_append.comp`): d = signed max / -32,
// q = min(63, floor(x / d + 32.5)), d stored as fp16. One superblock per writer
// invocation, so no two invocations share a word.

#define KV6_SB_WORDS 13u

uint kv6_row_words(uint head_dim) {
    return (head_dim >> 6) * KV6_SB_WORDS;
}

// `float NAME(uint row, uint e)`: element `e` of cache row `row` in buffer `BUF`.
// A macro because GLSL functions cannot take a buffer; `pc.head_dim` must exist.
#define KV6_DEFINE_LOAD(NAME, BUF)                                                    \
    float NAME(uint row, uint e) {                                                    \
        uint at = row * kv6_row_words(pc.head_dim) + (e >> 6) * KV6_SB_WORDS;         \
        uint s = (e >> 5) & 1u;                                                       \
        uint i = e & 31u;                                                             \
        float d = unpackHalf2x16(BUF[at])[s];                                         \
        uint w = at + 1u + s * 6u;                                                    \
        uint q = bitfieldExtract(BUF[w + (i >> 3)], int(4u * (i & 7u)), 4) |          \
                 (bitfieldExtract(BUF[w + 4u + (i >> 4)], int(2u * (i & 15u)), 2) << 4); \
        return (float(q) - 32.0) * d;                                                 \
    }
