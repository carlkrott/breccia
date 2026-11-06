//! QSA attention + lightning indexer — the full-attention sublayer of the 12
//! layers at `i % 4 == 3`.
//!
//! ```text
//!  q_full = attn_q @ x        [12288] = 24 heads x (q 256 | gate 256)
//!  K = rmsnorm(attn_k @ x, attn_k_norm) -> rope    [2 x 256]
//!  V = attn_v @ x                                  [2 x 256]   (never rotated)
//!  Q = rmsnorm(q half, attn_q_norm) -> rope        [24 x 256]
//!  indexer: q_idx = rope(rmsnorm(index_q @ x))     [4 x 128]
//!           raw_k = index_k @ x                   [128]        (no norm, no rope)
//!           pooled[b] = rope(rmsnorm(mean(raw_k[4b..4b+3])))    (one per 4-cell block)
//!  sel = top_min(n_blocks, 512) blocks by score + the incomplete tail block
//!  A = softmax(q . k / sqrt(256) over the selected cells, causal)
//!  A = A * sigmoid(gate)  (per head, BEFORE attn_output)
//!  out = attn_output @ A                           [6144 -> 2560]
//! ```
//!
//! (`docs/specs/qwen4exp-architecture.md` §4.3, §5.3, §10; `qwen4exp.cpp:738-1013`.)
//!
//! # Two layouts this module defines (the M5 contract)
//!
//! ## Attention KV — paged f16, 16 cells per page
//!
//! ```text
//! k_pool, v_pool :: u32 words, two f16 per word
//!   element index = ((page * n_kv_head + kv_head) * PAGE_CELLS + cell_in_page) * head_dim + d
//! page_table :: u32 per logical page (identity today; nothing assumes it)
//! ```
//!
//! `PAGE_CELLS = 16` makes one page of one layer exactly 16 KiB of K + 16 KiB of
//! V = the 2048 B/cell of `docs/phase-3/05` §3 (2 heads x 256 x 2 types x 2 B),
//! and 16 cells is the smallest granule worth streaming on a 4 MB L2. It is
//! deliberately **not** the indexer's 4-cell block: blocks are a *scoring*
//! granule, pages a *storage* one, and keeping them independent is what lets M5
//! change the page size (or add quantised bodies) without touching the indexer.
//!
//! ## Indexer key cache — pooled only
//!
//! `tail[3][128]` + `pooled[n_blocks + 1][128]` f32 (≈ 128 B/cell amortised,
//! strata's shape, `qsa.hpp:197-237`) instead of llama.cpp's `2*128` f32 per cell
//! (1024 B/cell = 3.0 GiB at full context). The raw tail is only 3 rows because
//! the cell that completes a block arrives in registers on the completion step.
//!
//! One deliberate simplification: strata keeps a **spare** pooled row for the
//! incomplete block so it can be scored. Here the incomplete block's cells are
//! always appended to the selection list instead, which is the same guarantee
//! (the current token can always attend to its own block) without a row whose
//! position moves every time a block completes.
//!
//! # Precision
//!
//! Weights are whatever the caller's GEMV decodes; activations are f32; K/V in
//! the cache round-trip through IEEE binary16 (spec §12.1, open question Q8).
//! Attention is therefore ~1e-3 relative against an f64 reference on the K/V path
//! — the conformance bar in `docs/phase-3/06` reflects that and the f16 store is
//! the only intentional precision loss in the sublayer.

pub mod bench;
pub mod cpu;

use ash::vk;
use anyhow::{anyhow, Result};
use breccia_vk::ctx::{Buf, Gpu};

use crate::gemv::alloc;
use crate::ops::{grid_for, Pc, Pipe, RmsNorm, Rope, RopeTable, Softmax};
use crate::proj::Proj;

/// `attention.head_count` (query heads).
pub const MODEL_N_HEAD: usize = 24;
/// `attention.head_count_kv`.
pub const MODEL_N_KV_HEAD: usize = 2;
/// `attention.key_length` / `value_length`.
pub const MODEL_HEAD_DIM: usize = 256;
/// `rope.dimension_count`.
pub const MODEL_N_ROT: usize = 64;
/// `attention.indexer.head_count` — query heads of the indexer.
pub const IDX_N_HEAD: usize = 4;
/// `attention.indexer.key_length`.
pub const IDX_DIM: usize = 128;
/// `attention.compress_ratios` at a QSA layer: one pooled key per 4 cells.
pub const IDX_BLOCK: usize = 4;
/// `attention.indexer.top_k` — cells of whole-block budget.
pub const IDX_TOP_K: usize = 2048;
/// Whole blocks the budget buys (`2048 / 4`).
pub const IDX_TOPK_BLOCKS: usize = IDX_TOP_K / IDX_BLOCK;

/// The selection budget in blocks, at runtime: `BRECCIA_IDX_TOP_K` takes a **cell**
/// count (default [`IDX_TOP_K`] = 2048, i.e. 512 blocks of 4 cells) and is rounded
/// down to whole [`IDX_BLOCK`] blocks, minimum one block. Smaller is faster and
/// approximate: it is the width of both attention passes and the number of KV cells
/// read per token, so it is the main long-context quality/speed dial. Read once per
/// process (the workspace size, the push constants and the engine's exactness cap
/// all have to agree).
pub fn idx_topk_blocks() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        let cells = std::env::var("BRECCIA_IDX_TOP_K")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(IDX_TOP_K);
        (cells / IDX_BLOCK).max(1)
    })
}

/// The cell count the selection budget implies ([`idx_topk_blocks`] x [`IDX_BLOCK`]).
/// Above this many visible cells the indexer's top-k runs; below it every block is
/// selected and the layer is dense causal attention.
pub fn dense_cell_limit() -> usize {
    idx_topk_blocks() * IDX_BLOCK
}
/// The spec's selection width, `min(n_kv, idx_top_k + idx_block - 1)`.
pub const SELECT_WIDTH: usize = IDX_TOP_K + IDX_BLOCK - 1;
/// Cells per KV page (this module's choice, spec §7 / `docs/phase-3/05` §7).
pub const PAGE_CELLS: usize = 16;

/// Rows of the QSA score matrix held at once. The prefill attention loop walks a
/// chunk in tiles of this many tokens, so `attn` is `24 x ATTN_TILE x slots`
/// instead of `24 x T x slots` — 24 KiB/token instead of 192 at an 8k context.
/// Attention is row-independent, so the tiling is exact.
pub const ATTN_TILE: usize = 1024;
/// `attn_q` output width: 24 heads x (256 q + 256 gate).
pub const QGATE_DIM: usize = MODEL_N_HEAD * 2 * MODEL_HEAD_DIM;
/// `attn_output` input width: 24 x 256.
pub const ATTN_OUT_DIM: usize = MODEL_N_HEAD * MODEL_HEAD_DIM;

const _: () = assert!(SELECT_WIDTH == 2051);
const _: () = assert!(QGATE_DIM == 12288);
const _: () = assert!(ATTN_OUT_DIM == 6144);
// `qsa_attn_apply_gqa.comp`: one invocation per `d`, GROUP = 12 heads per kv head.
const _: () = assert!(MODEL_HEAD_DIM == 256 && MODEL_N_HEAD == 12 * MODEL_N_KV_HEAD);

/// Calls of at least this many tokens use the kv-head-group apply. Below it the
/// per-head kernel stays: the group kernel runs `2 * T` workgroups, which a
/// decode step cannot spread over the device.
const GQA_MIN_TOKENS: usize = 16;

/// `BRECCIA_QSA_GQA=0` keeps the per-head apply at every `T` (A/B and revert).
fn gqa_apply(tokens: usize) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    tokens >= GQA_MIN_TOKENS
        && *ON.get_or_init(|| std::env::var("BRECCIA_QSA_GQA").map_or(true, |v| v != "0"))
}

/// Same gate for the scores (`qsa_attn_scores_gqa.comp`); `BRECCIA_QSA_GQA_SCORES=0`
/// keeps the per-head scores alone.
fn gqa_scores(tokens: usize) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    gqa_apply(tokens)
        && *ON.get_or_init(|| std::env::var("BRECCIA_QSA_GQA_SCORES").map_or(true, |v| v != "0"))
}

/// Prefill attention as one fused pass (`qsa_attn_flash_gqa.comp`) on the GQA
/// sizes instead of the scores/softmax/apply triple. On for chunks of at least
/// [`FLASH_MIN_TOKENS`]; `BRECCIA_QSA_FLASH=0` off, `=1` any size.
fn flash_attn(tokens: usize) -> bool {
    static MODE: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    let mode = *MODE.get_or_init(|| std::env::var("BRECCIA_QSA_FLASH").ok().map(|v| v != "0"));
    gqa_scores(tokens) && mode.unwrap_or(tokens >= FLASH_MIN_TOKENS)
}

/// RX 6950 XT ABBA, 8k prompt, prefill-only, flash vs tiled: chunk 8192 507 vs 447 tok/s
/// (+13 %), 4096 523 vs 556 (-6 %), 2048 509 vs 531 (-4 %); 1k prompt chunk 1024 -9 %.
/// One workgroup per token x kv head walks every slot serially; the tiled path is wider.
/// Tiled at chunk 4096 beats flash at 8192, so flash only helps an explicit 8k+ chunk.
const FLASH_MIN_TOKENS: usize = 8192;

/// Ids (4 slots each) per split-apply workgroup: ctx 1k -> 17 chunks, 4k -> 33.
/// `CHUNK_BLOCKS` in `qsa_attn_apply_split.comp` must match.
const SPLIT_CHUNK_BLOCKS: usize = 16;

/// Calls below `GQA_MIN_TOKENS` rows (decode, speculative verify) apply with the
/// split-K pair (`qsa_attn_apply_split.comp` + `_reduce`) instead of the per-head
/// kernel's `T * 24` serial workgroups. `BRECCIA_QSA_APPLY_SPLIT=0` opts out.
fn split_apply(tokens: usize) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    tokens < GQA_MIN_TOKENS
        && *ON.get_or_init(|| std::env::var("BRECCIA_QSA_APPLY_SPLIT").map_or(true, |v| v != "0"))
}

/// Which top-k selection kernel to run (see [`Qsa::topk_with`]).
///
/// `Rank` counts each block's rank in one parallel pass — O(n_blocks^2) compares
/// per row, which wins at 1k-4k (0.76 ms vs the serial kernel's 6.30 ms at 4k) but
/// costs 268 M compares per row at a 64k cache. `Serial` is k block-wide argmax
/// passes — O(k * n_blocks) with a sequential dependency of `k` (up to 512) passes.
/// `Radix` (`qsa_topk_radix.comp`, upstream strata's radix select) takes the same
/// set in one O(n) pass per bit-byte: 1024-row chunk at 128k cells, RX 5.8 vs serial
/// 34.1 us/row/layer, MI50 5.7 vs 91.7; 1 row at 32k RX 66 vs rank 303 us
/// (`qsa_idx_bench`, 2026-11-05). `Auto` = `Radix`; `BRECCIA_QSA_TOPK=serial|rank`
/// forces an old kernel (`rank` also sizes the candidate buffer it needs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TopkPath {
    Auto,
    Serial,
    Rank,
    /// `qsa_topk_radix.comp`: radix select, same set, ids ascending, scores kept.
    Radix,
}

impl TopkPath {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(TopkPath::Auto),
            "serial" => Some(TopkPath::Serial),
            "rank" => Some(TopkPath::Rank),
            "radix" => Some(TopkPath::Radix),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            TopkPath::Auto => "auto",
            TopkPath::Serial => "serial",
            TopkPath::Rank => "rank",
            TopkPath::Radix => "radix",
        }
    }
}

fn cand_topk() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("BRECCIA_QSA_TOPK_CAND").map_or(true, |v| v != "0"))
}

/// The top-k path [`Qsa::topk`] runs: `BRECCIA_QSA_TOPK`, else radix.
fn topk_path() -> TopkPath {
    match std::env::var("BRECCIA_QSA_TOPK")
        .ok()
        .as_deref()
        .and_then(TopkPath::parse)
    {
        Some(p) if p != TopkPath::Auto => p,
        _ => TopkPath::Radix,
    }
}

/// Indexer scores with `qsa_idx_score_tiled.comp` (upstream strata's FP32 tiled
/// GEMM): 1024-row chunk at 128k cells RX 2.8 vs 56.2 us/row/layer, MI50 5.6 vs
/// 179; max rel diff 6e-7. `BRECCIA_QSA_SCORE=naive` keeps the per-(token, block)
/// kernel.
fn tiled_score() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("BRECCIA_QSA_SCORE").map_or(true, |v| v != "naive"))
}

/// The candidate buffer (`tokens x n_blocks` f32, 128 MiB per 1024-row chunk at
/// 128k cells) is only for the forced rank path.
fn rank_forced() -> bool {
    std::env::var("BRECCIA_QSA_TOPK").is_ok_and(|v| v == "rank")
}

/// Calls of `1..GQA_MIN_TOKENS` rows (decode, speculative verify) score with
/// `qsa_attn_scores_small.comp`: a thread per slot, K read once per kv head. Decode at
/// ctx 1k: 70 -> 52-60 ms/token, same 32 tokens as the per-head kernel (not bit-equal).
/// `BRECCIA_QSA_SMALL_MIN=2` keeps decode on the per-head kernel, `=99` turns it off.
fn small_scores(tokens: usize) -> bool {
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    tokens < GQA_MIN_TOKENS
        && tokens
            >= *MIN.get_or_init(|| {
                std::env::var("BRECCIA_QSA_SMALL_MIN")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1)
            })
}

/// Bytes of one 16-cell page of one layer, K and V together (32 KiB).
pub fn page_bytes() -> u64 {
    (2 * MODEL_N_KV_HEAD * PAGE_CELLS * MODEL_HEAD_DIM * 2) as u64
}

/// How the attention **V** cache is stored (`docs/phase-4/01-forward-pass.md` §8bf/§8bg).
///
/// `F16` is the default and the reference; `Q8_0` is the lossy 0.5625x layout,
/// selected by `BRECCIA_KV_Q8V=1` ([`VLayout::from_env`]). K is f16 in both — this
/// round is V only (§8bf: V alone reaches the 4-agent milestone, K buys headroom).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VLayout {
    /// Paged f16 (`2 B` per element, the `docs/phase-3/05` §3 baseline).
    #[default]
    F16,
    /// q8_0: 32 x i8 + one f32 scale per 32-element block (`36 B` per 32 elements).
    Q8_0,
    /// q6_0 **K and V** (`shaders/kv_q6.glsl`): 6-bit codes + one fp16 scale per 32
    /// elements, 208 B per 256-element row (2.46x vs f16). The serving default since
    /// 2026-11-05 (32k tail gate KL 1.03e-2 / top-1 .984, same class as q8_0).
    Q6_0,
}

/// `BRECCIA_KV_OCTAVE_FQ` (quality gate, f16 cache only): round-trip K/V through the
/// vLLM Octave format on append. Encoding: bits 0-3 K (1 block k3 = 4-bit RoPE block,
/// 3-bit NoPE blocks; 2 compact 3-bit), bits 4-7 V bits (3/4). Unknown -> panic.
pub fn octave_fq_mode() -> u32 {
    static M: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *M.get_or_init(|| match std::env::var("BRECCIA_KV_OCTAVE_FQ").as_deref() {
        Err(_) | Ok("") | Ok("off") => 0,
        Ok("k3v4") => 1 | 4 << 4,
        Ok("k3v3") => 1 | 3 << 4,
        Ok("compact") => 2 | 3 << 4,
        Ok("k3") => 1,
        Ok("v4") => 4 << 4,
        Ok("v3") => 3 << 4,
        // llama.cpp `-ctk X -ctv X` reference points (bit 8 = ggml blocks of 32).
        // Octave method extended to 5/6 bits (K uniform per 64-dim block = kmode bits).
        Ok("k5v5") => 5 | 5 << 4,
        Ok("k6v6") => 6 | 6 << 4,
        Ok("q6_0") => 256 | 6 | 6 << 4,
        Ok("q8_0") => 256 | 8 | 8 << 4,
        Ok("q5_0") => 256 | 5 | 5 << 4,
        Ok("q4_0") => 256 | 4 | 4 << 4,
        Ok(o) => panic!("BRECCIA_KV_OCTAVE_FQ={o}: want k3v4|k3v3|compact|k3|v4|v3|k5v5|k6v6|q8_0|q6_0|q5_0|q4_0|off"),
    })
}

/// Process-wide V layout: the launch flag's value, else the env default. One
/// static (not one per function) so `set_current` and `current` agree.
static V_LAYOUT: std::sync::OnceLock<VLayout> = std::sync::OnceLock::new();

impl VLayout {
    /// The launch flag's choice, when set ([`VLayout::set_current`]); otherwise the
    /// env default. The flag exists so a serving config can name the layout instead
    /// of depending on the environment the process happens to inherit.
    pub fn current() -> Self {
        *V_LAYOUT.get_or_init(Self::from_env)
    }

    /// Set the layout for this process (first call wins; a later call is a no-op).
    /// Must run before the first `QsaState` is built — the layout sizes the cache.
    pub fn set_current(v: VLayout) {
        let _ = V_LAYOUT.set(v);
    }

    /// `BRECCIA_KV=f16|q8v|q6` names the layout; else `BRECCIA_KV_Q8V=1` selects
    /// [`VLayout::Q8_0`]; else the Octave fake-quant gate (which rewrites an f16
    /// cache) keeps f16; else the default [`VLayout::Q6_0`]. f16 is the exact opt-in.
    pub fn from_env() -> Self {
        if let Ok(v) = std::env::var("BRECCIA_KV") {
            return Self::parse(&v).unwrap_or_else(|| panic!("BRECCIA_KV={v}: want f16|q8v|q6"));
        }
        match std::env::var("BRECCIA_KV_Q8V").ok().as_deref() {
            Some("1") | Some("true") | Some("on") | Some("yes") => VLayout::Q8_0,
            _ if octave_fq_mode() != 0 => VLayout::F16,
            _ => VLayout::Q6_0,
        }
    }

    /// The `--kv` / `BRECCIA_KV` spelling of a layout.
    pub fn parse(v: &str) -> Option<Self> {
        match v {
            "f16" => Some(VLayout::F16),
            "q8v" | "q8_0" => Some(VLayout::Q8_0),
            "q6" | "q6_0" => Some(VLayout::Q6_0),
            _ => None,
        }
    }

    /// The flag's name in a report line.
    pub fn name(self) -> &'static str {
        match self {
            VLayout::F16 => "f16",
            VLayout::Q8_0 => "q8_0",
            VLayout::Q6_0 => "q6_0",
        }
    }
}

/// Per-layer, per-sequence QSA state: the paged K/V cache, its page table and
/// the indexer's pooled-key cache.
pub struct QsaState {
    pub k_pool: Buf,
    pub v_pool: Buf,
    pub page_table: Buf,
    /// `(idx_block-1, idx_dim)` f32 raw keys of the block being filled.
    pub idx_tail: Buf,
    /// `(max_blocks, idx_dim)` f32 pooled keys.
    pub idx_pooled: Buf,
    /// IMRoPE `(t, h, w)` triple per cell, identity `(i, i, i)` for text, so no
    /// call ever re-uploads positions; [`QsaState::set_positions`] rewrites a
    /// range for an image (and the text after it), [`QsaState::reset`] restores it.
    pub positions: Buf,
    /// Cells the cache can hold (a multiple of `PAGE_CELLS`).
    pub capacity_cells: usize,
    /// Position table length (the rope table's `max_pos`).
    pub max_pos: usize,
    /// The V layout `v_pool` is sized for and the attention kernels read.
    pub v_layout: VLayout,
    /// Mapped `page_table` when `k_pool`/`v_pool` alias a session's shared
    /// [`KvPool`] (the session writes this sequence's pages as cells are used);
    /// 0 = this state owns its pools and the table is identity.
    pub pt_map: usize,
}

/// **Every** byte of QSA state, from a function, because three silent wrong-value
/// bugs this week came from a size written as a literal
/// (`docs/phase-4/01-forward-pass.md` §8bc/§8bf).
pub mod sizes {
    use super::{IDX_BLOCK, IDX_DIM, MODEL_HEAD_DIM, MODEL_N_KV_HEAD, PAGE_CELLS};

    /// Bytes of one f16 K/V pool for `pages` pages (`2 f16 per u32 word`).
    pub fn f16_pool_bytes(pages: usize) -> u64 {
        (pages * MODEL_N_KV_HEAD * PAGE_CELLS * MODEL_HEAD_DIM * 2) as u64
    }

    /// Elements per q8_0 block and bytes per block (`Q8_0`: 32 x i8 + one f32 scale).
    pub const Q8_BLOCK_ELEMS: usize = 32;
    /// `32 x i8 + f32 scale`.
    pub const Q8_BLOCK_BYTES: usize = Q8_BLOCK_ELEMS + 4;

    /// q8_0 blocks per KV row (`head_dim / 32` = 8, exact by construction).
    pub fn q8_blocks_per_row() -> usize {
        assert_eq!(
            MODEL_HEAD_DIM % Q8_BLOCK_ELEMS,
            0,
            "the q8_0 V layout requires head_dim to be a multiple of the block size"
        );
        MODEL_HEAD_DIM / Q8_BLOCK_ELEMS
    }

    /// Bytes of the **q8_0 V pool**: every row is `q8_blocks_per_row()` blocks of
    /// `32 x i8 + f32 scale`, versus `2 B` per element for f16.
    pub fn q8v_pool_bytes(pages: usize) -> u64 {
        (pages * MODEL_N_KV_HEAD * PAGE_CELLS * q8_blocks_per_row() * Q8_BLOCK_BYTES) as u64
    }

    /// The ratio a q8_0 **V** pool holds against f16 — the number the capacity
    /// argument in §8bf turns on.
    pub fn q8v_vs_f16() -> f64 {
        q8v_pool_bytes(1) as f64 / f16_pool_bytes(1) as f64
    }

    /// **Words** of a q6_0 K or V row: `head_dim / 64` superblocks of 13 words
    /// (`shaders/kv_q6.glsl`) = 52 words = 208 B.
    pub fn q6_row_words() -> usize {
        assert_eq!(MODEL_HEAD_DIM % 64, 0, "q6_0 rows are whole 64-element superblocks");
        MODEL_HEAD_DIM / 64 * 13
    }

    /// Bytes of one q6_0 K or V pool.
    pub fn q6_pool_bytes(pages: usize) -> u64 {
        (pool_rows(pages) * q6_row_words() * 4) as u64
    }

    /// Bytes of the indexer's pooled keys, from the block count they are indexed by.
    pub fn pooled_idx_bytes(blocks: usize) -> u64 {
        (blocks * IDX_DIM * 4) as u64
    }

    /// **Words** of the f16 pool one cell row occupies (`head_dim / 2`).
    pub fn f16_row_words() -> usize {
        assert_eq!(
            MODEL_HEAD_DIM % 2,
            0,
            "an f16 pool stores pairs of elements"
        );
        MODEL_HEAD_DIM / 2
    }

    /// **Words** of the q8_0 V pool one cell row occupies: `head_dim / 32` blocks of
    /// 9 words (8 packed i8 + one f32 scale) = 72 words = 288 B.
    pub fn q8v_row_words() -> usize {
        q8_blocks_per_row() * (Q8_BLOCK_BYTES / 4)
    }

    /// Rows a pool of `pages` pages holds: `pages x n_kv_head x page_cells`. Both
    /// pool layouts index exactly this many rows (`*_pool_bytes` must agree with it).
    pub fn pool_rows(pages: usize) -> usize {
        pages * MODEL_N_KV_HEAD * PAGE_CELLS
    }

    /// Bytes of the indexer raw tail: `(idx_block - 1)` rows, because the completing
    /// cell arrives in registers (`qsa_idx_tail.comp`).
    pub fn idx_tail_bytes() -> u64 {
        ((IDX_BLOCK - 1) * IDX_DIM * 4) as u64
    }

    /// Blocks the pooled-key cache needs for `cells` cells.
    pub fn pooled_blocks(cells: usize) -> usize {
        cells / IDX_BLOCK + 1
    }
}

/// The six buffers [`QsaState::with_layout`] allocates, in its slot order.
fn buffer_sizes(capacity_cells: usize, max_pos: usize, v_layout: VLayout) -> [(u64, &'static str); 6] {
    let pages = capacity_cells.div_ceil(PAGE_CELLS);
    let (k_bytes, v_bytes) = match v_layout {
        VLayout::F16 => (sizes::f16_pool_bytes(pages), sizes::f16_pool_bytes(pages)),
        VLayout::Q8_0 => (sizes::f16_pool_bytes(pages), sizes::q8v_pool_bytes(pages)),
        VLayout::Q6_0 => (sizes::q6_pool_bytes(pages), sizes::q6_pool_bytes(pages)),
    };
    [
        (k_bytes, "kv k_pool"),
        (v_bytes, "kv v_pool"),
        (pages as u64 * 4, "kv page_table"),
        (sizes::idx_tail_bytes(), "indexer tail"),
        (sizes::pooled_idx_bytes(sizes::pooled_blocks(pages * PAGE_CELLS)), "indexer pooled"),
        (max_pos as u64 * 12, "position table"),
    ]
}

/// Bytes one layer's [`QsaState`] allocates (for VRAM budgeting before any session exists).
pub fn state_bytes(capacity_cells: usize, max_pos: usize, v_layout: VLayout, kv_host: bool) -> u64 {
    // With host K/V ([`kv_host_on`]) the K/V pools (the first two) live in host memory.
    let skip = if kv_host { 2 } else { 0 };
    buffer_sizes(capacity_cells, max_pos, v_layout).iter().skip(skip).map(|b| b.0).sum()
}

/// `BRECCIA_KV_POOL_CELLS`: K/V cells per QSA layer shared by every sequence
/// of a session. Pages are handed out as cells are written and returned on
/// reset, so one long and several short sequences fit the same budget.
pub fn kv_pool_cells() -> Option<usize> {
    std::env::var("BRECCIA_KV_POOL_CELLS")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .filter(|&n| n > 0)
}

/// Bytes of one layer's shared K/V pool of `cells` cells.
pub fn kv_pool_bytes(cells: usize, v_layout: VLayout) -> u64 {
    let s = buffer_sizes(cells, 8, v_layout);
    s[0].0 + s[1].0
}

/// One QSA layer's K/V page pool, shared by a session's sequences (each maps
/// its logical pages to pool pages through its own page table).
pub struct KvPool {
    pub k: Buf,
    pub v: Buf,
    pub pages: usize,
}

impl KvPool {
    pub fn new(gpu: &Gpu, cells: usize) -> Result<Self> {
        let pages = cells.div_ceil(PAGE_CELLS);
        let s = buffer_sizes(pages * PAGE_CELLS, 8, VLayout::current());
        let host = kv_host_on(&gpu.pci_address);
        let mk = |n: u64| if host { alloc_host(gpu, n) } else { alloc(gpu, n) };
        let k = mk(s[0].0).map_err(|e| anyhow!("allocating the shared KV pool (k): {e}"))?;
        let v = match mk(s[1].0) {
            Ok(v) => v,
            Err(e) => {
                gpu.destroy_buf(&k);
                return Err(anyhow!("allocating the shared KV pool (v): {e}"));
            }
        };
        // Zeroed once: a page reused from another sequence holds finite stale
        // values (masked by position like the zeroes of a fresh pool), never NaN.
        let poison = std::env::var_os("BRECCIA_FAULT_POISON_QSA").is_some();
        for dst in [&k, &v] {
            let bytes = if poison {
                crate::ops::u32_to_bytes(&vec![f32::NAN.to_bits(); (dst.size / 4) as usize])
            } else {
                vec![0u8; dst.size as usize]
            };
            if let Err(e) = crate::gemv::upload_into(gpu, dst, &bytes) {
                gpu.destroy_buf(&k);
                gpu.destroy_buf(&v);
                return Err(anyhow!("initialising the shared KV pool: {e}"));
            }
        }
        Ok(KvPool { k, v, pages })
    }

    pub fn bytes(&self) -> u64 {
        self.k.size + self.v.size
    }

    pub fn release(&self, gpu: &Gpu) {
        gpu.destroy_buf(&self.k);
        gpu.destroy_buf(&self.v);
    }
}

/// `BRECCIA_KV_HOST`: allocate the K/V pools in host memory (GTT) that the GPU
/// reads over PCIe. Decode reads only the selected top-k cells per layer, so the
/// pools need not take VRAM; the indexer, page table and positions stay on the card.
/// `1` = every card; otherwise a comma list of PCI address suffixes
/// (`03:00.0`) = only those cards (e.g. the RX, keeping the MI50's KV in VRAM).
pub fn kv_host_on(pci: &str) -> bool {
    static V: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    kv_host_matches(V.get_or_init(|| std::env::var("BRECCIA_KV_HOST").unwrap_or_default()), pci)
}

fn kv_host_matches(spec: &str, pci: &str) -> bool {
    spec == "1"
        || spec.split(',').map(str::trim).any(|e| !e.is_empty() && e != "0" && pci.ends_with(e))
}

fn alloc_host(gpu: &Gpu, size: u64) -> Result<Buf> {
    let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    gpu.buf(
        size.max(4) + crate::gemv::TAIL_PAD,
        vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::TRANSFER_SRC,
        host,
        host,
    )
    .map_err(|e| anyhow!("{e}"))
}

/// `(i, i, i)` for `i < max_pos`: the text position table.
fn identity_triples(max_pos: usize) -> Vec<u32> {
    (0..max_pos as u32).flat_map(|i| [i, i, i]).collect()
}

/// IMRoPE `(t, h, w)` per cell with images at `(cell0, nx, ny)` (mtmd
/// `MTMD_POS_TYPE_MROPE`): image token `i` at base `p` is `(p, p + i/nx, p + i%nx)`,
/// the image advances the position by `max(nx, ny)` (`mtmd_input_chunk_get_n_pos`),
/// so every later cell sits `nx*ny - max(nx, ny)` behind its index.
pub fn mrope_triples(max_pos: usize, spans: &[(usize, usize, usize)]) -> Result<Vec<u32>> {
    let mut out = Vec::with_capacity(3 * max_pos);
    let mut lag = 0usize; // cell index - position
    let mut c = 0usize;
    for &(c0, nx, ny) in spans {
        let n = nx * ny;
        if c0 < c || nx == 0 || ny == 0 || c0 + n > max_pos {
            return Err(anyhow!(
                "image span (cell {c0}, {nx}x{ny}) overlaps an earlier one or leaves the {max_pos}-cell table"
            ));
        }
        out.extend((c..c0).flat_map(|i| {
            let p = (i - lag) as u32;
            [p, p, p]
        }));
        let p = c0 - lag;
        out.extend((0..n).flat_map(|i| [p as u32, (p + i / nx) as u32, (p + i % nx) as u32]));
        lag += n - nx.max(ny);
        c = c0 + n;
    }
    out.extend((c..max_pos).flat_map(|i| {
        let p = (i - lag) as u32;
        [p, p, p]
    }));
    Ok(out)
}

impl QsaState {
    /// Allocate for `capacity_cells` cells (rounded up to whole pages) and a rope
    /// table of `max_pos` positions, with the V layout from `BRECCIA_KV_Q8V`.
    pub fn new(gpu: &Gpu, capacity_cells: usize, max_pos: usize) -> Result<Self> {
        Self::with_layout(gpu, capacity_cells, max_pos, VLayout::current())
    }

    /// [`QsaState::new`] with the V layout given explicitly (the conformance test
    /// needs both layouts in one process, which an env var cannot express).
    pub fn with_layout(
        gpu: &Gpu,
        capacity_cells: usize,
        max_pos: usize,
        v_layout: VLayout,
    ) -> Result<Self> {
        if capacity_cells == 0 || max_pos == 0 {
            return Err(anyhow!(
                "qsa state needs capacity_cells > 0 and max_pos > 0"
            ));
        }
        let pages = capacity_cells.div_ceil(PAGE_CELLS);
        // The V pool's byte count is the one number the capacity argument turns on,
        // so it is chosen from the layout and then **checked against the row count
        // the kernels index** (cells x n_kv_head rows) instead of being trusted.
        let v_bytes = match v_layout {
            VLayout::F16 => sizes::f16_pool_bytes(pages),
            VLayout::Q8_0 => sizes::q8v_pool_bytes(pages),
            VLayout::Q6_0 => sizes::q6_pool_bytes(pages),
        };
        // K is f16 except in the q6_0 layout, which quantises both.
        let k_bytes = match v_layout {
            VLayout::Q6_0 => sizes::q6_pool_bytes(pages),
            _ => sizes::f16_pool_bytes(pages),
        };
        assert_eq!(
            sizes::f16_pool_bytes(pages) as usize,
            sizes::pool_rows(pages) * sizes::f16_row_words() * 4,
            "the f16 pool must hold cells x n_kv_head rows of f16_row_words() words"
        );
        assert_eq!(
            sizes::q8v_pool_bytes(pages) as usize,
            sizes::pool_rows(pages) * sizes::q8v_row_words() * 4,
            "the q8_0 V pool must hold cells x n_kv_head rows of q8_blocks_per_row() blocks"
        );
        assert!(
            v_bytes == sizes::f16_pool_bytes(pages)
                || v_bytes == sizes::q8v_pool_bytes(pages)
                || v_bytes == sizes::q6_pool_bytes(pages),
            "the V pool is one of the two layouts the kernels index, not a literal"
        );

        let sizes = buffer_sizes(capacity_cells, max_pos, v_layout);
        assert_eq!((sizes[0].0, sizes[1].0), (k_bytes, v_bytes));
        assert_eq!(
            sizes.len(),
            6,
            "QsaState has six buffers; the positional slot list must match it"
        );
        let mut bufs: Vec<Buf> = Vec::with_capacity(sizes.len());
        for (i, (bytes, what)) in sizes.iter().enumerate() {
            let a = if i < 2 && kv_host_on(&gpu.pci_address) { alloc_host(gpu, *bytes) } else { alloc(gpu, *bytes) };
            match a {
                Ok(b) => bufs.push(b),
                Err(e) => {
                    for b in &bufs {
                        gpu.destroy_buf(b);
                    }
                    return Err(anyhow!("allocating the qsa state ({what}): {e}"));
                }
            }
        }
        let mut it = bufs.into_iter();
        let mut take = || it.next().expect("six qsa state buffers were allocated");
        let k_pool = take();
        let v_pool = take();
        let page_table = take();
        let idx_tail = take();
        let idx_pooled = take();
        let positions = take();
        let all = [
            &k_pool,
            &v_pool,
            &page_table,
            &idx_tail,
            &idx_pooled,
            &positions,
        ];

        // Page table: identity. Position table: 0, 1, 2, ... Both are written
        // once here, so no call ever re-uploads them (a captured graph cannot
        // freeze a token's position, `qsa.hpp:236-241`).
        let table: Vec<u32> = (0..pages as u32).collect();
        let pos = identity_triples(max_pos);
        let prepared: [(&str, &Buf, Vec<u8>); 2] = [
            (
                "KV page table",
                &page_table,
                crate::ops::u32_to_bytes(&table),
            ),
            ("position table", &positions, crate::ops::u32_to_bytes(&pos)),
        ];
        for (what, dst, bytes) in prepared {
            if let Err(e) = crate::gemv::upload_into(gpu, dst, &bytes) {
                for b in all {
                    gpu.destroy_buf(b);
                }
                return Err(anyhow!("uploading the {what}: {e}"));
            }
        }
        for (what, dst) in [("k_pool", &k_pool), ("v_pool", &v_pool)] {
            // From the buffer, not from the f16 pool's size: with `BRECCIA_KV_Q8V`
            // the V pool is 0.5625x the K pool and a shared length would overrun.
            let zeros = vec![0u8; dst.size as usize];
            if let Err(e) = crate::gemv::upload_into(gpu, dst, &zeros) {
                for b in all {
                    gpu.destroy_buf(b);
                }
                return Err(anyhow!("zeroing {what}: {e}"));
            }
        }
        // `BRECCIA_FAULT_POISON_QSA=1`: fill every mutable buffer with NaN after
        // construction. A fresh process's K/V pools start zeroed but its indexer tail
        // and pooled keys start as whatever the allocator left; poisoning all four
        // proves that *nothing* reads a cell past the current position (a leak shows
        // up as NaN logits), which is the argument for why `QsaState::reset` closes a
        // gap on paper without changing a token (docs/phase-4/01-forward-pass.md §8az).
        if std::env::var_os("BRECCIA_FAULT_POISON_QSA").is_some() {
            let nan = f32::NAN.to_bits();
            let words = |b: &Buf| (b.size / 4) as usize;
            for dst in [&k_pool, &v_pool, &idx_tail, &idx_pooled] {
                let poison: Vec<u32> = vec![nan; words(dst)];
                crate::gemv::upload_into(gpu, dst, &crate::ops::u32_to_bytes(&poison))
                    .map_err(|e| anyhow!("poisoning the QSA state: {e}"))?;
            }
        }
        Ok(QsaState {
            k_pool,
            v_pool,
            page_table,
            idx_tail,
            idx_pooled,
            positions,
            capacity_cells: pages * PAGE_CELLS,
            max_pos,
            v_layout,
            pt_map: 0,
        })
    }

    /// A state whose K/V cells live in `pool` (shared with other sequences):
    /// own page table (mapped, all entries 0 until [`Self::map_pages`]),
    /// indexer tail, pooled keys and positions.
    pub fn with_pool(gpu: &Gpu, capacity_cells: usize, max_pos: usize, pool: &KvPool) -> Result<Self> {
        if capacity_cells == 0 || max_pos == 0 {
            return Err(anyhow!("qsa state needs capacity_cells > 0 and max_pos > 0"));
        }
        let pages = capacity_cells.div_ceil(PAGE_CELLS);
        let sizes = buffer_sizes(capacity_cells, max_pos, VLayout::current());
        let flags = vk::MemoryPropertyFlags::HOST_COHERENT;
        let (page_table, map) = gpu
            .buf_mapped(
                sizes[2].0 + crate::gemv::TAIL_PAD,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                flags,
                flags | vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .map_err(|e| anyhow!("allocating the KV page table: {e}"))?;
        // SAFETY: `map` covers the whole table, which no command reads yet.
        unsafe { std::ptr::write_bytes(map, 0, sizes[2].0 as usize) };
        let mut bufs = vec![page_table];
        for (bytes, what) in &sizes[3..] {
            match alloc(gpu, *bytes) {
                Ok(b) => bufs.push(b),
                Err(e) => {
                    for b in &bufs {
                        gpu.destroy_buf(b);
                    }
                    return Err(anyhow!("allocating the qsa state ({what}): {e}"));
                }
            }
        }
        let alias = |b: &Buf| Buf { buf: b.buf, mem: b.mem, size: b.size, id: b.id };
        let mut it = bufs.into_iter();
        let st = QsaState {
            k_pool: alias(&pool.k),
            v_pool: alias(&pool.v),
            page_table: it.next().unwrap(),
            idx_tail: it.next().unwrap(),
            idx_pooled: it.next().unwrap(),
            positions: it.next().unwrap(),
            capacity_cells: pages * PAGE_CELLS,
            max_pos,
            v_layout: VLayout::current(),
            pt_map: map as usize,
        };
        let pos = identity_triples(max_pos);
        if let Err(e) = crate::gemv::upload_into(gpu, &st.positions, &crate::ops::u32_to_bytes(&pos)) {
            st.destroy(gpu);
            return Err(anyhow!("uploading the position table: {e}"));
        }
        if std::env::var_os("BRECCIA_FAULT_POISON_QSA").is_some() {
            for dst in [&st.idx_tail, &st.idx_pooled] {
                let poison = vec![f32::NAN.to_bits(); (dst.size / 4) as usize];
                if let Err(e) = crate::gemv::upload_into(gpu, dst, &crate::ops::u32_to_bytes(&poison)) {
                    st.destroy(gpu);
                    return Err(anyhow!("poisoning the QSA state: {e}"));
                }
            }
        }
        Ok(st)
    }

    /// Point every logical page back at pool page 0, the sink no sequence owns
    /// (call with nothing in flight: the write is not queue-ordered).
    pub fn clear_pages(&self) {
        if self.pt_map != 0 {
            let n = self.capacity_cells.div_ceil(PAGE_CELLS) * 4;
            // SAFETY: the mapping covers the whole table.
            unsafe { std::ptr::write_bytes(self.pt_map as *mut u8, 0, n) };
        }
    }

    /// Point logical pages `first..first + phys.len()` at pool pages `phys`
    /// (shared-pool states only). Host-coherent: visible to the next submit.
    pub fn map_pages(&self, first: usize, phys: &[u32]) {
        assert!(self.pt_map != 0, "map_pages on a state that owns its pools");
        assert!((first + phys.len()) * PAGE_CELLS <= self.capacity_cells);
        // SAFETY: in bounds (asserted); in-flight commands read only pages
        // already mapped, never these.
        unsafe {
            std::ptr::copy_nonoverlapping(
                phys.as_ptr(),
                (self.pt_map as *mut u32).add(first),
                phys.len(),
            )
        };
    }

    /// Return every **mutable** QSA buffer to its construction-time contents.
    ///
    /// A session is reused across requests (`Session::reset`), so this is what makes
    /// generation N+1 see exactly what a fresh process would: the paged K/V pools,
    /// the raw indexer tail (the last, incomplete block's keys) and the pooled
    /// indexer keys are all per-sequence state. `page_table` and `positions` are
    /// written once at construction and never mutated by a kernel, so re-uploading
    /// them is a belt-and-braces assertion of that invariant rather than a fix.
    ///
    /// Until this existed, `idx_tail` and `idx_pooled` held *uninitialised VRAM* in a
    /// fresh process and the previous request's keys in a reused one; both are masked
    /// by the current position today (`docs/phase-4/01-forward-pass.md` §8az carries
    /// the proof that the outputs were already identical), so this closes the gap on
    /// paper without changing a single token.
    pub fn reset(&self, gpu: &Gpu) -> Result<()> {
        let own = [("KV k_pool", &self.k_pool), ("KV v_pool", &self.v_pool)];
        let shared = self.pt_map != 0;
        for (what, dst) in own.into_iter().filter(|_| !shared).chain([
            ("indexer tail", &self.idx_tail),
            ("indexer pooled", &self.idx_pooled),
        ]) {
            let zeros = vec![0u8; dst.size as usize];
            crate::gemv::upload_into(gpu, dst, &zeros)
                .map_err(|e| anyhow!("zeroing the {what} of the QSA state: {e}"))?;
        }
        // A shared-pool table is the session's to rewrite as pages are handed out.
        if !shared {
            let pages = self.capacity_cells.div_ceil(PAGE_CELLS);
            let table: Vec<u32> = (0..pages as u32).collect();
            crate::gemv::upload_into(gpu, &self.page_table, &crate::ops::u32_to_bytes(&table))
                .map_err(|e| anyhow!("restoring the KV page table: {e}"))?;
        }
        let pos = identity_triples(self.max_pos);
        crate::gemv::upload_into(gpu, &self.positions, &crate::ops::u32_to_bytes(&pos))
            .map_err(|e| anyhow!("restoring the position table: {e}"))?;
        Ok(())
    }

    /// Rewrite the position table for a prompt holding images, one
    /// `(cell0, nx, ny)` per image (sorted by cell); `&[]` restores identity.
    /// The whole table is re-uploaded (`max_pos * 12` bytes, once per request).
    pub fn set_image_positions(&self, gpu: &Gpu, spans: &[(usize, usize, usize)]) -> Result<()> {
        let pos = mrope_triples(self.max_pos, spans)?;
        crate::gemv::upload_into(gpu, &self.positions, &crate::ops::u32_to_bytes(&pos))
            .map_err(|e| anyhow!("writing the position table: {e}"))
    }

    /// Bytes this state owns (a shared pool is counted once, by its session).
    pub fn bytes(&self) -> u64 {
        let pools = if self.pt_map != 0 { 0 } else { self.k_pool.size + self.v_pool.size };
        pools + [
            &self.page_table,
            &self.idx_tail,
            &self.idx_pooled,
            &self.positions,
        ]
        .iter()
        .map(|b| b.size)
        .sum::<u64>()
    }

    /// Pooled-key rows the state can hold.
    pub fn max_blocks(&self) -> usize {
        self.capacity_cells / IDX_BLOCK + 1
    }

    pub fn release(self, gpu: &Gpu) {
        self.destroy(gpu);
    }

    /// Free what this state owns (not a shared pool's K/V).
    pub fn destroy(&self, gpu: &Gpu) {
        if self.pt_map == 0 {
            gpu.destroy_buf(&self.k_pool);
            gpu.destroy_buf(&self.v_pool);
        }
        for b in [&self.page_table, &self.idx_tail, &self.idx_pooled, &self.positions] {
            gpu.destroy_buf(b);
        }
    }
}

/// Everything one QSA call needs to keep, sized from `(tokens, n_blocks, slots)`.
pub struct QsaWorkspace {
    pub tokens: usize,
    /// `[T, 12288]` the `attn_q` projection (q and gate interleaved per head).
    pub q_full: Buf,
    /// `[T, 24, 256]` normalised + rotated queries.
    pub q: Buf,
    /// `[T, 2, 256]` K (normed + rotated in place).
    pub k_cur: Buf,
    /// `[T, 2, 256]` V (raw).
    pub v_cur: Buf,
    /// `[T, 4, 128]` indexer queries (normed + rotated in place).
    pub idx_q: Buf,
    /// `[T, 128]` raw indexer keys.
    pub idx_k: Buf,
    /// `[T, n_blocks]` indexer block scores (destroyed by the top-k).
    pub idx_scores: Buf,
    /// `[T, max_ids]` selected block ids, descending score then the tail block.
    pub ids: Buf,
    /// `[T]` ids per token.
    pub counts: Buf,
    /// `[T]` proven lower bound on the `topk_blocks`-th score
    /// (`qsa_topk_thr.comp`): blocks below it skip the rank select.
    pub thr: Buf,
    /// `[T, n_blocks]` candidate ids: every block scoring at least `thr`, in block
    /// order (`qsa_topk_pick.comp`). The rank select scores only these.
    pub cand: Buf,
    /// `[T]` how many candidates each token has.
    pub ncand: Buf,
    /// `[24*T, slots]` attention scores, softmaxed in place.
    pub attn: Buf,
    /// Rows of [`QsaWorkspace::attn`] (the attention tile; `tokens` when the call is
    /// smaller than [`ATTN_TILE`] or not on the GQA path).
    pub attn_rows: usize,
    /// `[T, 24, 256]` gated attention output (the `attn_output` GEMV's input).
    pub attn_out: Buf,
    /// `[24*T, chunks, 256]` split-apply partial sums (one word unless
    /// [`split_apply`] holds for this `T`).
    pub attn_part: Buf,
    pub n_blocks: usize,
    pub slots: usize,
    pub max_ids: usize,
    /// Selection budget in blocks for this workspace ([`idx_topk_blocks`]); it sizes
    /// `max_ids` and goes into every top-k push constant.
    pub topk_blocks: usize,
}

impl QsaWorkspace {
    pub fn bytes(&self) -> u64 {
        [
            &self.q_full,
            &self.q,
            &self.k_cur,
            &self.v_cur,
            &self.idx_q,
            &self.idx_k,
            &self.idx_scores,
            &self.ids,
            &self.counts,
            &self.thr,
            &self.cand,
            &self.ncand,
            &self.attn,
            &self.attn_out,
            &self.attn_part,
        ]
        .iter()
        .map(|b| b.size)
        .sum()
    }
}

/// The QSA-only weight tensors (§6.3): the norms. Projections are the caller's
/// GEMVs.
pub struct QsaWeights<'a> {
    /// `attn_q_norm.weight` f32 `[256]` (per query head).
    pub q_norm: &'a Buf,
    /// `attn_k_norm.weight` f32 `[256]` (per kv head).
    pub k_norm: &'a Buf,
    /// `indexer.q_norm.weight` f32 `[128]`.
    pub idx_q_norm: &'a Buf,
    /// `indexer.k_norm.weight` f32 `[128]`.
    pub idx_k_norm: &'a Buf,
}

/// The six projections of one QSA layer.
pub struct QsaProjections<'a, 'g, 'w> {
    pub q: Proj<'a, 'g>,
    pub w_q: &'w Buf,
    pub k: Proj<'a, 'g>,
    pub w_k: &'w Buf,
    pub v: Proj<'a, 'g>,
    pub w_v: &'w Buf,
    pub idx_q: Proj<'a, 'g>,
    pub w_idx_q: &'w Buf,
    pub idx_k: Proj<'a, 'g>,
    pub w_idx_k: &'w Buf,
    pub out: Proj<'a, 'g>,
    pub w_out: &'w Buf,
}

/// The QSA pipelines, compiled once per device.
pub struct Qsa<'g> {
    q_norm: Pipe<'g>,
    kv_append: Pipe<'g>,
    kv_append_q6: Pipe<'g>,
    /// `BRECCIA_KV_Q8V`: quantise V on write (`qsa_kv_append.comp` q8_0 build).
    kv_append_q8: Pipe<'g>,
    /// `BRECCIA_KV_OCTAVE_FQ`: Octave round-trip before the f16 store (quality gate).
    kv_append_oct: Pipe<'g>,
    idx_tail: Pipe<'g>,
    idx_pool: Pipe<'g>,
    idx_score: Pipe<'g>,
    idx_score_tiled: Pipe<'g>,
    topk_radix: Pipe<'g>,
    topk: Pipe<'g>,
    topk_rank: Pipe<'g>,
    topk_thr: Pipe<'g>,
    topk_pick: Pipe<'g>,
    topk_cand: Pipe<'g>,
    attn_scores: Pipe<'g>,
    attn_scores_q6: Pipe<'g>,
    attn_scores_gqa: Pipe<'g>,
    attn_scores_gqa_q6: Pipe<'g>,
    attn_scores_small: Pipe<'g>,
    attn_scores_small_q6: Pipe<'g>,
    attn_apply: Pipe<'g>,
    attn_apply_q6: Pipe<'g>,
    /// `BRECCIA_KV_Q8V`: read V as q8_0, dequantise in registers in the dot.
    attn_apply_q8: Pipe<'g>,
    /// Prefill apply: one workgroup per (token, kv head), V loaded once for the
    /// 12 heads that share it (`qsa_attn_apply_gqa.comp`, plan §12 item 3).
    attn_apply_gqa: Pipe<'g>,
    attn_apply_gqa_q6: Pipe<'g>,
    attn_apply_gqa_q8: Pipe<'g>,
    /// Prefill scores + online softmax + apply in one pass, no score matrix
    /// (`qsa_attn_flash_gqa.comp`).
    attn_flash: Pipe<'g>,
    attn_flash_q6: Pipe<'g>,
    attn_flash_q8: Pipe<'g>,
    /// Decode/verify apply, split over slot chunks (`qsa_attn_apply_split.comp`)
    /// and summed by `qsa_attn_apply_reduce.comp`.
    attn_apply_split: Pipe<'g>,
    attn_apply_split_q6: Pipe<'g>,
    attn_apply_split_q8: Pipe<'g>,
    attn_apply_reduce: Pipe<'g>,
    softmax: Softmax<'g>,
    rope_hd: Rope<'g>,
    rope_idx: Rope<'g>,
    rms_k: RmsNorm<'g>,
    rms_idx: RmsNorm<'g>,
    table: RopeTable,
    table_buf: Buf,
}

impl<'g> Qsa<'g> {
    /// Compile the pipelines for a window of `max_pos` positions.
    pub fn new(gpu: &'g Gpu, max_pos: usize) -> Result<Self> {
        // Pipes first: on any failure they free themselves through `Drop`, and
        // the rope table (the only other device object here) is created last.
        let q_norm = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_q_norm.spv")),
            3,
            "qsa q norm",
        )?;
        let kv_append = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_kv_append.spv")),
            5,
            "qsa kv append",
        )?;
        let kv_append_q6 = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_kv_append_q6.spv")),
            5,
            "qsa kv append (q6_0 K+V)",
        )?;
        let kv_append_q8 = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_kv_append_q8v.spv")),
            5,
            "qsa kv append (q8_0 V)",
        )?;
        let kv_append_oct = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_kv_append_octfq.spv")),
            5,
            "qsa kv append (octave fake-quant)",
        )?;
        let idx_tail = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_idx_tail.spv")),
            2,
            "qsa indexer tail",
        )?;
        let idx_pool = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_idx_pool.spv")),
            6,
            "qsa indexer pool",
        )?;
        let idx_score = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_idx_score.spv")),
            3,
            "qsa indexer score",
        )?;
        let topk = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_topk.spv")),
            3,
            "qsa indexer top-k",
        )?;
        let topk_thr = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_topk_thr.spv")),
            2,
            "qsa indexer top-k threshold",
        )?;
        let topk_pick = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_topk_pick.spv")),
            4,
            "qsa indexer top-k pick",
        )?;
        let topk_cand = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_topk_cand.spv")),
            5,
            "qsa indexer top-k candidates",
        )?;
        let idx_score_tiled = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_idx_score_tiled.spv")),
            3,
            "qsa indexer score (tiled)",
        )?;
        let topk_radix = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_topk_radix.spv")),
            3,
            "qsa indexer top-k (radix)",
        )?;
        let topk_rank = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_topk_rank.spv")),
            4,
            "qsa indexer top-k (rank)",
        )?;
        let attn_scores = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_attn_scores.spv")),
            6,
            "qsa attention scores",
        )?;
        let attn_scores_q6 = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_attn_scores_q6.spv")),
            6,
            "qsa attention scores (q6_0 K+V)",
        )?;
        let attn_scores_gqa = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_attn_scores_gqa.spv")),
            6,
            "qsa attention scores (kv-head groups)",
        )?;
        let attn_scores_gqa_q6 = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_attn_scores_gqa_q6.spv")),
            6,
            "qsa attention scores (kv-head groups) (q6_0 K+V)",
        )?;
        let attn_scores_small = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_attn_scores_small.spv")),
            6,
            "qsa attention scores (small T)",
        )?;
        let attn_scores_small_q6 = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_attn_scores_small_q6.spv")),
            6,
            "qsa attention scores (small T) (q6_0 K+V)",
        )?;
        let attn_apply = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_attn_apply.spv")),
            7,
            "qsa attention apply",
        )?;
        let attn_apply_q6 = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_attn_apply_q6.spv")),
            7,
            "qsa attention apply (q6_0 K+V)",
        )?;
        let attn_apply_q8 = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_attn_apply_q8v.spv")),
            7,
            "qsa attention apply (q8_0 V)",
        )?;
        let attn_apply_gqa = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_attn_apply_gqa.spv")),
            7,
            "qsa attention apply (kv-head groups)",
        )?;
        let attn_apply_gqa_q6 = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_attn_apply_gqa_q6.spv")),
            7,
            "qsa attention apply (kv-head groups) (q6_0 K+V)",
        )?;
        let attn_apply_gqa_q8 = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_attn_apply_gqa_q8v.spv")),
            7,
            "qsa attention apply (kv-head groups, q8_0 V)",
        )?;
        let attn_flash = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_attn_flash_gqa.spv")),
            8,
            "qsa flash attention (kv-head groups)",
        )?;
        let attn_flash_q6 = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_attn_flash_gqa_q6.spv")),
            8,
            "qsa flash attention (kv-head groups) (q6_0 K+V)",
        )?;
        let attn_flash_q8 = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_attn_flash_gqa_q8v.spv")),
            8,
            "qsa flash attention (kv-head groups, q8_0 V)",
        )?;
        let attn_apply_split = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_attn_apply_split.spv")),
            7,
            "qsa attention apply (split)",
        )?;
        let attn_apply_split_q6 = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_attn_apply_split_q6.spv")),
            7,
            "qsa attention apply (split) (q6_0 K+V)",
        )?;
        let attn_apply_split_q8 = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_attn_apply_split_q8v.spv")),
            7,
            "qsa attention apply (split, q8_0 V)",
        )?;
        let attn_apply_reduce = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_attn_apply_reduce.spv")),
            3,
            "qsa attention apply (split reduce)",
        )?;
        let softmax = Softmax::new(gpu)?;
        let rope_hd = Rope::new(gpu, MODEL_HEAD_DIM, MODEL_N_ROT, max_pos)?.triple_positions();
        let rope_idx = Rope::new(gpu, IDX_DIM, MODEL_N_ROT, max_pos)?.triple_positions();
        let rms_k = RmsNorm::new(gpu, MODEL_HEAD_DIM, 1)?;
        let rms_idx = RmsNorm::new(gpu, IDX_DIM, 1)?;
        let table = RopeTable::model(max_pos)?;
        let table_buf = table.upload(gpu)?;
        Ok(Qsa {
            q_norm,
            kv_append,
            kv_append_q6,
            kv_append_q8,
            kv_append_oct,
            idx_tail,
            idx_pool,
            idx_score,
            idx_score_tiled,
            topk_radix,
            topk,
            topk_rank,
            topk_thr,
            topk_pick,
            topk_cand,
            attn_scores,
            attn_scores_q6,
            attn_scores_gqa,
            attn_scores_gqa_q6,
            attn_scores_small,
            attn_scores_small_q6,
            attn_apply,
            attn_apply_q6,
            attn_apply_q8,
            attn_apply_gqa,
            attn_apply_gqa_q6,
            attn_apply_gqa_q8,
            attn_flash,
            attn_flash_q6,
            attn_flash_q8,
            attn_apply_split,
            attn_apply_split_q6,
            attn_apply_split_q8,
            attn_apply_reduce,
            softmax,
            rope_hd,
            rope_idx,
            rms_k,
            rms_idx,
            table,
            table_buf,
        })
    }

    pub fn table(&self) -> &RopeTable {
        &self.table
    }

    /// Allocate the per-call scratch for `tokens` tokens and a cache length of
    /// `n_kv_after` cells (the length **after** this call).
    ///
    /// The cache length, not the block count, is the argument on purpose: the
    /// number of *complete* blocks is `floor(n_kv/4)` and the tail block is
    /// selected separately, so a caller passing `ceil(n_kv/4)` would score a block
    /// that does not exist yet. Everything derived here — `n_blocks`, `max_ids`
    /// (`min(n_blocks, 512) + 1`), and the score-matrix width `4 * max_ids` — comes
    /// from that one number, which keeps a short context cheap: 64 tokens allocate
    /// a 17-block, 72-slot matrix rather than the 2052-slot worst case.
    pub fn workspace(&self, gpu: &Gpu, tokens: usize, n_kv_after: usize) -> Result<QsaWorkspace> {
        if tokens == 0 {
            return Err(anyhow!("qsa workspace needs tokens >= 1"));
        }
        let n_blocks = n_kv_after / IDX_BLOCK;
        let topk_blocks = idx_topk_blocks();
        let max_ids = n_blocks.min(topk_blocks) + 1;
        let slots = 4 * max_ids;
        // The prefill attention is tiled over rows; a call that does not use both
        // GQA kernels keeps a full-size score matrix.
        let attn_rows = if gqa_scores(tokens) && gqa_apply(tokens) {
            tokens.min(ATTN_TILE)
        } else {
            tokens
        };
        let sizes = [
            tokens * QGATE_DIM,
            tokens * ATTN_OUT_DIM,
            tokens * MODEL_N_KV_HEAD * MODEL_HEAD_DIM,
            tokens * MODEL_N_KV_HEAD * MODEL_HEAD_DIM,
            tokens * IDX_N_HEAD * IDX_DIM,
            tokens * IDX_DIM,
            tokens * n_blocks.max(1),
            tokens * max_ids,
            tokens,
            tokens,
            if rank_forced() { tokens * n_blocks.max(1) } else { 1 },
            tokens,
            MODEL_N_HEAD * attn_rows * slots,
            tokens * ATTN_OUT_DIM,
            // A prefill chunk workspace is reused by speculative verify at T < 16 rows.
            if split_apply(1) {
                tokens.min(GQA_MIN_TOKENS - 1) * ATTN_OUT_DIM * max_ids.div_ceil(SPLIT_CHUNK_BLOCKS)
            } else {
                1
            },
        ];
        assert_eq!(
            sizes.len(),
            15,
            "QsaWorkspace has fifteen fields; the positional slot list must match it"
        );
        let mut bufs: Vec<Buf> = Vec::with_capacity(sizes.len());
        for (i, n) in sizes.iter().enumerate() {
            match alloc(gpu, (*n * 4) as u64) {
                Ok(b) => bufs.push(b),
                Err(e) => {
                    for b in &bufs {
                        gpu.destroy_buf(b);
                    }
                    return Err(anyhow!("allocating qsa workspace slot {i}: {e}"));
                }
            }
        }
        let mut it = bufs.into_iter();
        let mut take = || it.next().expect("fifteen qsa buffers were allocated");
        Ok(QsaWorkspace {
            tokens,
            q_full: take(),
            q: take(),
            k_cur: take(),
            v_cur: take(),
            idx_q: take(),
            idx_k: take(),
            idx_scores: take(),
            ids: take(),
            counts: take(),
            thr: take(),
            cand: take(),
            ncand: take(),
            attn: take(),
            attn_rows,
            attn_out: take(),
            attn_part: take(),
            n_blocks,
            slots,
            max_ids,
            topk_blocks,
        })
    }

    pub fn release(&self, gpu: &Gpu, ws: QsaWorkspace) {
        for b in [
            &ws.q_full,
            &ws.q,
            &ws.k_cur,
            &ws.v_cur,
            &ws.idx_q,
            &ws.idx_k,
            &ws.idx_scores,
            &ws.ids,
            &ws.counts,
            &ws.thr,
            &ws.cand,
            &ws.ncand,
            &ws.attn,
            &ws.attn_out,
            &ws.attn_part,
        ] {
            gpu.destroy_buf(b);
        }
    }

    pub fn table_buf(&self) -> &Buf {
        &self.table_buf
    }

    // -- individual stages ------------------------------------------------

    /// `attn_q` output -> contiguous queries, per-head RMSNorm applied.
    pub fn q_norm(&self, ws: &QsaWorkspace, gamma: &Buf) -> Result<()> {
        let pc = Pc::new()
            .u((ws.tokens * MODEL_N_HEAD) as u32)
            .u(MODEL_HEAD_DIM as u32)
            .u(MODEL_HEAD_DIM as u32)
            .f(crate::ops::MODEL_RMS_EPS)
            .u(MODEL_N_HEAD as u32);
        let (gx, gy) = grid_for(ws.tokens * MODEL_N_HEAD, 1);
        self.q_norm
            .run(&[&ws.q_full, gamma, &ws.q], &pc, (gx, gy, 1))
    }

    /// Per-head RMSNorm of K in place (`rows = n_kv_head * T`).
    pub fn k_norm(&self, ws: &QsaWorkspace, gamma: &Buf) -> Result<()> {
        self.rms_k.run(
            &ws.k_cur,
            Some(gamma),
            &ws.k_cur,
            ws.tokens * MODEL_N_KV_HEAD,
        )
    }

    /// Indexer query RMSNorm in place (`rows = idx_n_head * T`).
    pub fn idx_q_norm(&self, ws: &QsaWorkspace, gamma: &Buf) -> Result<()> {
        self.rms_idx
            .run(&ws.idx_q, Some(gamma), &ws.idx_q, ws.tokens * IDX_N_HEAD)
    }

    /// IMRoPE of the queries (rows = 24*T, `pos_div = 24`).
    pub fn rope_q(&self, ws: &QsaWorkspace, state: &QsaState, pos0: usize) -> Result<()> {
        self.rope_hd.run_at(
            &ws.q,
            &self.table_buf,
            &state.positions,
            &ws.q,
            ws.tokens * MODEL_N_HEAD,
            pos0 as i32,
            MODEL_N_HEAD,
        )
    }

    /// IMRoPE of K (rows = 2*T, `pos_div = 2`).
    pub fn rope_k(&self, ws: &QsaWorkspace, state: &QsaState, pos0: usize) -> Result<()> {
        self.rope_hd.run_at(
            &ws.k_cur,
            &self.table_buf,
            &state.positions,
            &ws.k_cur,
            ws.tokens * MODEL_N_KV_HEAD,
            pos0 as i32,
            MODEL_N_KV_HEAD,
        )
    }

    /// IMRoPE of the indexer queries (rows = 4*T, `pos_div = 4`).
    pub fn rope_idx_q(&self, ws: &QsaWorkspace, state: &QsaState, pos0: usize) -> Result<()> {
        self.rope_idx.run_at(
            &ws.idx_q,
            &self.table_buf,
            &state.positions,
            &ws.idx_q,
            ws.tokens * IDX_N_HEAD,
            pos0 as i32,
            IDX_N_HEAD,
        )
    }

    /// Append this call's K/V to the paged cache (V quantised to q8_0 when the
    /// state's layout says so).
    pub fn kv_append(&self, state: &QsaState, ws: &QsaWorkspace, pos0: usize) -> Result<()> {
        self.kv_append_octave(state, ws, pos0, octave_fq_mode())
    }

    /// [`Self::kv_append`] with an explicit Octave fake-quant mode
    /// ([`octave_fq_mode`] encoding; 0 = plain).
    pub fn kv_append_octave(
        &self,
        state: &QsaState,
        ws: &QsaWorkspace,
        pos0: usize,
        oct: u32,
    ) -> Result<()> {
        if pos0 + ws.tokens > state.capacity_cells {
            return Err(anyhow!(
                "kv_append would write cells {pos0}..{} but the cache holds {} \
                 (a write past the page table reads an unmapped page index and can \
                 scribble over live cells)",
                pos0 + ws.tokens,
                state.capacity_cells
            ));
        }
        let pc = Pc::new()
            .u(ws.tokens as u32)
            .i(pos0 as i32)
            .u(MODEL_N_KV_HEAD as u32)
            .u(MODEL_HEAD_DIM as u32)
            .u(PAGE_CELLS as u32)
            .u(oct);
        let (pipe, gx, gy) = self.kv_append_target(state, ws.tokens, oct);
        pipe.run(
            &[
                &state.k_pool,
                &state.v_pool,
                &state.page_table,
                &ws.k_cur,
                &ws.v_cur,
            ],
            &pc,
            (gx, gy, 1),
        )
    }

    /// The append pipeline **and** work shape for this state's V layout. The two
    /// differ because q8_0's scale is a max over the block, so the q8_0 build runs
    /// one invocation per (token, kv_head, 32-element block) and the f16 build one
    /// per (token, kv_head, d-pair). Same bindings, same push constants.
    fn kv_append_target(&self, state: &QsaState, tokens: usize, oct: u32) -> (&Pipe<'g>, u32, u32) {
        assert!(
            oct == 0 || state.v_layout == VLayout::F16,
            "BRECCIA_KV_OCTAVE_FQ rewrites an f16 cache: run it with BRECCIA_KV=f16"
        );
        let (pipe, units, wg) = match state.v_layout {
            VLayout::F16 if oct != 0 => (&self.kv_append_oct, tokens * MODEL_N_KV_HEAD, 128),
            VLayout::F16 => (
                &self.kv_append,
                tokens * MODEL_N_KV_HEAD * sizes::f16_row_words(),
                128,
            ),
            VLayout::Q8_0 => (
                &self.kv_append_q8,
                tokens * MODEL_N_KV_HEAD * sizes::q8_blocks_per_row(),
                32,
            ),
            VLayout::Q6_0 => (
                &self.kv_append_q6,
                tokens * MODEL_N_KV_HEAD * (MODEL_HEAD_DIM / 64),
                128,
            ),
        };
        let (gx, gy) = grid_for(units, wg);
        (pipe, gx, gy)
    }

    /// The apply pipeline for this state's V layout (f16 load, or q8_0 dequantised
    /// in registers); the work shape is the same `(token, head, d)` in both.
    fn attn_apply_target(&self, state: &QsaState) -> &Pipe<'g> {
        match state.v_layout {
            VLayout::F16 => &self.attn_apply,
            VLayout::Q8_0 => &self.attn_apply_q8,
            VLayout::Q6_0 => &self.attn_apply_q6,
        }
    }

    /// The three K-reading score pipelines (per-head, small, gqa) for this state's
    /// K layout: f16 unless the layout is q6_0.
    fn scores_targets(&self, state: &QsaState) -> (&Pipe<'g>, &Pipe<'g>, &Pipe<'g>) {
        match state.v_layout {
            VLayout::Q6_0 => (&self.attn_scores_q6, &self.attn_scores_small_q6, &self.attn_scores_gqa_q6),
            _ => (&self.attn_scores, &self.attn_scores_small, &self.attn_scores_gqa),
        }
    }

    /// Pool the indexer keys of every block completed by this call.
    pub fn idx_pool(
        &self,
        state: &QsaState,
        ws: &QsaWorkspace,
        gamma: &Buf,
        pos0: usize,
    ) -> Result<()> {
        let (b0, n_new) = blocks_completed(pos0, ws.tokens);
        if n_new == 0 {
            return Ok(());
        }
        if b0 + n_new > state.max_blocks() {
            return Err(anyhow!(
                "idx_pool would write pooled rows {b0}..{} but the state holds {}",
                b0 + n_new,
                state.max_blocks()
            ));
        }
        let pc = Pc::new()
            .u(n_new as u32)
            .u(b0 as u32)
            .i(pos0 as i32)
            .u(ws.tokens as u32)
            .u(IDX_BLOCK as u32)
            .u(IDX_DIM as u32)
            .u(MODEL_N_ROT as u32)
            .f(crate::ops::MODEL_RMS_EPS);
        let (gx, gy) = grid_for(n_new, 1);
        self.idx_pool.run(
            &[
                &ws.idx_k,
                &state.idx_tail,
                gamma,
                &self.table_buf,
                &state.idx_pooled,
                &state.positions,
            ],
            &pc,
            (gx, gy, 1),
        )
    }

    /// Store this call's raw indexer keys into the rolling 3-slot tail.
    pub fn idx_tail(&self, state: &QsaState, ws: &QsaWorkspace, pos0: usize) -> Result<()> {
        self.idx_tail_raw(state, &ws.idx_k, ws.tokens, pos0)
    }

    /// [`Self::idx_tail`] for `tokens` raw keys held in `idx_k` (`[tokens, 128]`):
    /// a speculative rollback re-stores a saved prefix of a chunk's keys.
    pub fn idx_tail_raw(
        &self,
        state: &QsaState,
        idx_k: &Buf,
        tokens: usize,
        pos0: usize,
    ) -> Result<()> {
        let pc = Pc::new()
            .u(tokens as u32)
            .i(pos0 as i32)
            .u(IDX_DIM as u32)
            .u(IDX_BLOCK as u32);
        let (gx, gy) = grid_for(tokens * IDX_DIM, 128);
        self.idx_tail
            .run(&[idx_k, &state.idx_tail], &pc, (gx, gy, 1))
    }

    /// Score every completed block against every token's indexer query.
    pub fn idx_score(&self, state: &QsaState, ws: &QsaWorkspace, pos0: usize) -> Result<()> {
        // T=1 (decode): the 32-query tile wastes 31 rows; naive measured 3-15% faster.
        self.idx_score_with(state, ws, pos0, tiled_score() && ws.tokens > 1)
    }

    /// [`Self::idx_score`] on a named kernel (tests and A/B): `tiled` =
    /// `qsa_idx_score_tiled.comp`, else the per-(token, block) kernel.
    pub fn idx_score_with(
        &self,
        state: &QsaState,
        ws: &QsaWorkspace,
        pos0: usize,
        tiled: bool,
    ) -> Result<()> {
        if ws.n_blocks > state.max_blocks() {
            return Err(anyhow!(
                "idx_score covers {} blocks but the state holds {} pooled rows",
                ws.n_blocks,
                state.max_blocks()
            ));
        }
        let pc = Pc::new()
            .u(ws.tokens as u32)
            .u(ws.n_blocks as u32)
            .u(IDX_N_HEAD as u32)
            .u(IDX_DIM as u32)
            .i(pos0 as i32)
            .u(IDX_BLOCK as u32)
            .f(1.0 / (IDX_DIM as f32).sqrt());
        if tiled {
            let grid = (ws.n_blocks.div_ceil(64) as u32, ws.tokens.div_ceil(32) as u32, 1);
            return self.idx_score_tiled.run(
                &[&ws.idx_q, &state.idx_pooled, &ws.idx_scores],
                &pc,
                grid,
            );
        }
        let (gx, gy) = grid_for(ws.tokens * ws.n_blocks, 1);
        self.idx_score.run(
            &[&ws.idx_q, &state.idx_pooled, &ws.idx_scores],
            &pc,
            (gx, gy, 1),
        )
    }

    /// Top-k block selection, plus each query's own block if the selection missed
    /// it (see `qsa_topk.comp` for why it is per token and never duplicated).
    ///
    /// `pos0` is the sequence position of the first token of the call.
    /// Which selection kernel [`Qsa::topk`] uses. `Auto` is the measured choice;
    /// the others exist for A/B and for contexts where the auto rule is wrong.
    pub fn topk(&self, ws: &QsaWorkspace, pos0: usize) -> Result<()> {
        self.topk_with(ws, pos0, topk_path())
    }

    /// Compute the rank select's score threshold (`qsa_topk_thr.comp`).
    fn topk_threshold(&self, ws: &QsaWorkspace) -> Result<()> {
        let groups = ws.topk_blocks.min(ws.n_blocks).max(1);
        let pc = Pc::new()
            .u(ws.tokens as u32)
            .u(ws.n_blocks as u32)
            .u(groups as u32)
            .u(0);
        let (gx, gy) = grid_for(ws.tokens, 1);
        self.topk_thr
            .run_then_barrier(&[&ws.idx_scores, &ws.thr], &pc, (gx, gy, 1))
    }

    /// Threshold + candidate compaction (`qsa_topk_pick.comp`).
    fn topk_pick(&self, ws: &QsaWorkspace) -> Result<()> {
        let groups = ws.topk_blocks.min(ws.n_blocks).max(1);
        let pc = Pc::new()
            .u(ws.tokens as u32)
            .u(ws.n_blocks as u32)
            .u(groups as u32)
            .u(ws.n_blocks as u32);
        let (gx, gy) = grid_for(ws.tokens, 1);
        // One dispatch writes thr/cand/ncand and the next reads them: unbatched
        // runs need the barrier in the command buffer (a batch's tracker inserts it).
        self.topk_pick.run_then_barrier(
            &[&ws.idx_scores, &ws.cand, &ws.ncand, &ws.thr],
            &pc,
            (gx, gy, 1),
        )
    }

    /// Rank the candidates (`qsa_topk_cand.comp`).
    fn topk_cand(&self, ws: &QsaWorkspace) -> Result<()> {
        let pc = Pc::new()
            .u(ws.tokens as u32)
            .u(ws.n_blocks as u32)
            .u(ws.topk_blocks as u32)
            .u(ws.max_ids as u32)
            .u(ws.n_blocks as u32);
        let chunks = ws.n_blocks.div_ceil(256).max(16) as u32;
        self.topk_cand.run(
            &[&ws.idx_scores, &ws.cand, &ws.ncand, &ws.thr, &ws.ids],
            &pc,
            (chunks, ws.tokens as u32, 1),
        )
    }

    /// [`Qsa::topk`] on a named path (tests and A/B).
    pub fn topk_with(&self, ws: &QsaWorkspace, pos0: usize, path: TopkPath) -> Result<()> {
        let pc = Pc::new()
            .u(ws.tokens as u32)
            .u(ws.n_blocks as u32)
            .u(ws.topk_blocks as u32)
            .u(ws.max_ids as u32)
            .i(pos0 as i32)
            .u(0); // dense off: a word Pc does not push is undefined in the shader
        if path == TopkPath::Radix {
            let (gx, gy) = grid_for(ws.tokens, 1);
            return self
                .topk_radix
                .run(&[&ws.idx_scores, &ws.ids, &ws.counts], &pc, (gx, gy, 1));
        }
        if path == TopkPath::Rank {
            // The candidate buffer exists only when rank is forced by env.
            if cand_topk() && ws.cand.size >= (ws.tokens * ws.n_blocks * 4) as u64 {
                // Threshold + compact, then rank the candidates only. Exact: every
                // candidate outscores every non-candidate, so rank-among-candidates
                // is rank-among-all.
                self.topk_pick(ws)?;
                self.topk_cand(ws)?;
            } else {
                // A/B: the whole-row rank with the threshold prune.
                self.topk_threshold(ws)?;
                let bufs = [&ws.idx_scores, &ws.ids, &ws.counts, &ws.thr];
                let (gx, gy) = grid_for(ws.tokens * ws.n_blocks.div_ceil(256), 1);
                self.topk_rank.run(&bufs, &pc, (gx, gy, 1))?;
            }
            let bufs = [&ws.idx_scores, &ws.ids, &ws.counts, &ws.thr];
            let pc1 = Pc::new()
                .u(ws.tokens as u32)
                .u(ws.n_blocks as u32)
                .u(ws.topk_blocks as u32)
                .u(ws.max_ids as u32)
                .i(pos0 as i32)
                .u(1);
            let (gx, gy) = grid_for(ws.tokens, 1);
            return self.topk_rank.run(&bufs, &pc1, (gx, gy, 1));
        }
        let (gx, gy) = grid_for(ws.tokens, 1);
        self.topk
            .run(&[&ws.idx_scores, &ws.ids, &ws.counts], &pc, (gx, gy, 1))
    }

    /// [`Self::topk`] when every visible block is selected anyway (`n_blocks <=
    /// IDX_TOPK_BLOCKS`): no scores read, ids in ascending block order.
    pub fn topk_dense(&self, ws: &QsaWorkspace, pos0: usize) -> Result<()> {
        let pc = Pc::new()
            .u(ws.tokens as u32)
            .u(ws.n_blocks as u32)
            .u(ws.topk_blocks as u32)
            .u(ws.max_ids as u32)
            .i(pos0 as i32)
            .u(1);
        let (gx, gy) = grid_for(ws.tokens, 1);
        self.topk
            .run(&[&ws.idx_scores, &ws.ids, &ws.counts], &pc, (gx, gy, 1))
    }

    /// Scores over the selected cells, then softmax in place.
    /// Whole-call attention (one tile). Callers with a chunk larger than
    /// [`ATTN_TILE`] must use [`Qsa::forward`], which alternates attend and apply per
    /// tile; this is for decode/small batches and the stage bench.
    pub fn attend(&self, state: &QsaState, ws: &QsaWorkspace, pos0: usize) -> Result<()> {
        debug_assert!(
            ws.attn_rows == ws.tokens,
            "attend() on a tiled workspace would leave only the last tile's scores"
        );
        self.attend_tile(state, ws, pos0, 0, ws.tokens)
    }

    fn attend_tile(
        &self,
        state: &QsaState,
        ws: &QsaWorkspace,
        pos0: usize,
        row0: usize,
        tile: usize,
    ) -> Result<()> {
        let pc = Pc::new()
            .u(ws.tokens as u32)
            .u(MODEL_N_HEAD as u32)
            .u(MODEL_HEAD_DIM as u32)
            .u(ws.slots as u32)
            .u(ws.max_ids as u32)
            .i(pos0 as i32)
            .u(PAGE_CELLS as u32)
            .u(MODEL_N_KV_HEAD as u32);
        if gqa_scores(ws.tokens) {
            // One tile: `forward` runs attend+apply per tile, because the score
            // matrix is reused (applying after all tiles would read the last one).
            {
                let pct = Pc::new()
                    .u(tile as u32)
                    .u(MODEL_N_HEAD as u32)
                    .u(MODEL_HEAD_DIM as u32)
                    .u(ws.slots as u32)
                    .u(ws.max_ids as u32)
                    .i(pos0 as i32)
                    .u(PAGE_CELLS as u32)
                    .u(MODEL_N_KV_HEAD as u32)
                    .u(row0 as u32);
                let (gx, gy) = grid_for(tile * MODEL_N_KV_HEAD, 1);
                self.scores_targets(state).2.run(
                    &[
                        &ws.q,
                        &state.k_pool,
                        &state.page_table,
                        &ws.ids,
                        &ws.counts,
                        &ws.attn,
                    ],
                    &pct,
                    (gx, gy, 1),
                )?;
                self.softmax
                    .run(&ws.attn, &ws.attn, MODEL_N_HEAD * tile, ws.slots)?;
            }
            return Ok(());
        }
        let (pipe, units) = if small_scores(ws.tokens) {
            (
                self.scores_targets(state).1,
                ws.tokens * MODEL_N_KV_HEAD * ws.slots.div_ceil(256),
            )
        } else {
            (self.scores_targets(state).0, ws.tokens * MODEL_N_HEAD)
        };
        let (gx, gy) = grid_for(units, 1);
        pipe.run(
            &[
                &ws.q,
                &state.k_pool,
                &state.page_table,
                &ws.ids,
                &ws.counts,
                &ws.attn,
            ],
            &pc,
            (gx, gy, 1),
        )?;
        self.softmax
            .run(&ws.attn, &ws.attn, MODEL_N_HEAD * ws.tokens, ws.slots)
    }

    /// Fused prefill attention over all `ws.tokens` rows: scores, online softmax and
    /// gated apply straight into `attn_out`, without the `attn` score matrix.
    pub fn flash(&self, state: &QsaState, ws: &QsaWorkspace, pos0: usize) -> Result<()> {
        let pc = Pc::new()
            .u(ws.tokens as u32)
            .u(MODEL_N_HEAD as u32)
            .u(MODEL_HEAD_DIM as u32)
            .u(ws.slots as u32)
            .u(ws.max_ids as u32)
            .i(pos0 as i32)
            .u(PAGE_CELLS as u32)
            .u(MODEL_N_KV_HEAD as u32)
            .u(0);
        let pipe = match state.v_layout {
            VLayout::F16 => &self.attn_flash,
            VLayout::Q8_0 => &self.attn_flash_q8,
            VLayout::Q6_0 => &self.attn_flash_q6,
        };
        let (gx, gy) = grid_for(ws.tokens * MODEL_N_KV_HEAD, 1);
        pipe.run(
            &[
                &ws.q,
                &state.k_pool,
                &state.v_pool,
                &state.page_table,
                &ws.ids,
                &ws.counts,
                &ws.q_full,
                &ws.attn_out,
            ],
            &pc,
            (gx, gy, 1),
        )
    }

    /// Probability-weighted V, gated by `sigmoid(gate)` from the `attn_q` half.
    /// See [`Qsa::attend`]: one tile, for callers whose chunk fits one.
    pub fn apply(&self, state: &QsaState, ws: &QsaWorkspace, pos0: usize) -> Result<()> {
        debug_assert!(
            ws.attn_rows == ws.tokens,
            "apply() on a tiled workspace would read only the last tile's scores"
        );
        self.apply_tile(state, ws, pos0, 0, ws.tokens)
    }

    fn apply_tile(
        &self,
        state: &QsaState,
        ws: &QsaWorkspace,
        pos0: usize,
        row0: usize,
        tile: usize,
    ) -> Result<()> {
        let pc = Pc::new()
            .u(ws.tokens as u32)
            .u(MODEL_N_HEAD as u32)
            .u(MODEL_HEAD_DIM as u32)
            .u(ws.slots as u32)
            .u(ws.max_ids as u32)
            .i(pos0 as i32)
            .u(PAGE_CELLS as u32)
            .u(MODEL_N_KV_HEAD as u32);
        if split_apply(ws.tokens) {
            let chunks = ws.max_ids.div_ceil(SPLIT_CHUNK_BLOCKS);
            let pipe = match state.v_layout {
                VLayout::F16 => &self.attn_apply_split,
                VLayout::Q8_0 => &self.attn_apply_split_q8,
                VLayout::Q6_0 => &self.attn_apply_split_q6,
            };
            let (gx, gy) = grid_for(ws.tokens * MODEL_N_HEAD * chunks, 1);
            pipe.run(
                &[
                    &ws.attn,
                    &state.v_pool,
                    &state.page_table,
                    &ws.ids,
                    &ws.counts,
                    &ws.q_full,
                    &ws.attn_part,
                ],
                &pc,
                (gx, gy, 1),
            )?;
            let rpc = Pc::new()
                .u(ws.tokens as u32)
                .u(MODEL_N_HEAD as u32)
                .u(MODEL_HEAD_DIM as u32)
                .u(chunks as u32);
            let (gx, gy) = grid_for(ws.tokens * ATTN_OUT_DIM, 256);
            return self.attn_apply_reduce.run(
                &[&ws.attn_part, &ws.q_full, &ws.attn_out],
                &rpc,
                (gx, gy, 1),
            );
        }
        if gqa_apply(ws.tokens) {
            {
                let pct = Pc::new()
                    .u(tile as u32)
                    .u(MODEL_N_HEAD as u32)
                    .u(MODEL_HEAD_DIM as u32)
                    .u(ws.slots as u32)
                    .u(ws.max_ids as u32)
                    .i(pos0 as i32)
                    .u(PAGE_CELLS as u32)
                    .u(MODEL_N_KV_HEAD as u32)
                    .u(row0 as u32);
                let pipe = match state.v_layout {
                    VLayout::F16 => &self.attn_apply_gqa,
                    VLayout::Q8_0 => &self.attn_apply_gqa_q8,
                    VLayout::Q6_0 => &self.attn_apply_gqa_q6,
                };
                let (gx, gy) = grid_for(tile * MODEL_N_KV_HEAD, 1);
                pipe.run(
                    &[
                        &ws.attn,
                        &state.v_pool,
                        &state.page_table,
                        &ws.ids,
                        &ws.counts,
                        &ws.q_full,
                        &ws.attn_out,
                    ],
                    &pct,
                    (gx, gy, 1),
                )?;
            }
            return Ok(());
        }
        let gqa = gqa_apply(ws.tokens);
        let (gx, gy) = if gqa {
            grid_for(ws.tokens * MODEL_N_KV_HEAD, 1)
        } else {
            grid_for(ws.tokens * ATTN_OUT_DIM, 256)
        };
        let pipe = match (gqa, state.v_layout) {
            (true, VLayout::F16) => &self.attn_apply_gqa,
            (true, VLayout::Q8_0) => &self.attn_apply_gqa_q8,
            (true, VLayout::Q6_0) => &self.attn_apply_gqa_q6,
            (false, _) => self.attn_apply_target(state),
        };
        pipe.run(
            &[
                &ws.attn,
                &state.v_pool,
                &state.page_table,
                &ws.ids,
                &ws.counts,
                &ws.q_full,
                &ws.attn_out,
            ],
            &pc,
            (gx, gy, 1),
        )
    }

    /// One whole QSA sublayer: six GEMVs and everything above, in §4.3 order.
    ///
    /// `x` is `[T, 2560]` (the HC_READ `mixed`), `out` receives `[T, 2560]`
    /// (`attn_output @ A`). `pos0` is the sequence position of the first token.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        state: &QsaState,
        ws: &QsaWorkspace,
        w: &QsaWeights<'_>,
        proj: &QsaProjections<'_, '_, '_>,
        x: &Buf,
        out: &Buf,
        pos0: usize,
    ) -> Result<()> {
        let t = ws.tokens;
        let gpu = self.q_norm.gpu();
        // GPU-time probe stage boundaries (`BRECCIA_GPU_TIME`, no-op when off).
        crate::batch_cmd::probe::enter(gpu, crate::batch_cmd::probe::QKV);
        proj.q.run(proj.w_q, x, &ws.q_full, t)?;
        proj.k.run(proj.w_k, x, &ws.k_cur, t)?;
        proj.v.run(proj.w_v, x, &ws.v_cur, t)?;
        proj.idx_q.run(proj.w_idx_q, x, &ws.idx_q, t)?;
        proj.idx_k.run(proj.w_idx_k, x, &ws.idx_k, t)?;

        crate::batch_cmd::probe::enter(gpu, crate::batch_cmd::probe::RMSNORM);
        self.q_norm(ws, w.q_norm)?;
        self.k_norm(ws, w.k_norm)?;
        self.idx_q_norm(ws, w.idx_q_norm)?;
        crate::batch_cmd::probe::enter(gpu, crate::batch_cmd::probe::ROPE);
        self.rope_q(ws, state, pos0)?;
        self.rope_k(ws, state, pos0)?;
        self.rope_idx_q(ws, state, pos0)?;

        crate::batch_cmd::probe::enter(gpu, crate::batch_cmd::probe::KV_APPEND);
        self.kv_append(state, ws, pos0)?;
        self.idx_pool(state, ws, w.idx_k_norm, pos0)?;
        crate::batch_cmd::probe::enter(gpu, crate::batch_cmd::probe::QSA_AUX);
        self.idx_tail(state, ws, pos0)?;
        if ws.n_blocks <= ws.topk_blocks && idx_skip() {
            self.topk_dense(ws, pos0)?;
        } else {
            // A/B knob (default on): `BRECCIA_IDX_SCORE=0` skips the scoring pass so
            // its share of `qsa-aux` can be measured directly.
            if std::env::var("BRECCIA_IDX_SCORE").map_or(true, |v| v != "0") {
                self.idx_score(state, ws, pos0)?;
            }
            self.topk(ws, pos0)?;
        }
        crate::batch_cmd::probe::enter(gpu, crate::batch_cmd::probe::ATTN);
        // Attention is row-independent, so the pair is tiled together: `attn` is one
        // tile deep, and each tile's scores are consumed before the next overwrites
        // them. A call that is not GQA-sized has `attn_rows == tokens` (one tile).
        let mut row0 = 0usize;
        if flash_attn(t) {
            self.flash(state, ws, pos0)?;
            row0 = t;
        }
        while row0 < t {
            let tile = (t - row0).min(ws.attn_rows).max(1);
            self.attend_tile(state, ws, pos0, row0, tile)?;
            self.apply_tile(state, ws, pos0, row0, tile)?;
            row0 += tile;
        }
        crate::batch_cmd::probe::enter(gpu, crate::batch_cmd::probe::ATTN_OUT);
        proj.out.run(proj.w_out, &ws.attn_out, out, t)?;
        Ok(())
    }
}

impl Drop for Qsa<'_> {
    fn drop(&mut self) {
        // `table_buf` was handed to this struct; the pipelines' own `Drop` frees
        // their kernels.
        let gpu = self.q_norm.gpu();
        gpu.destroy_buf(&self.table_buf);
    }
}

/// Indexer skip at <= 2k context (§11l item 2); `BRECCIA_IDX_SKIP=0` opts out.
fn idx_skip() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("BRECCIA_IDX_SKIP").map_or(true, |v| v != "0"))
}

/// `(first block completed by this call, how many complete)`.
///
/// Block `b` is completed by cell `4b + 3` (`qsa.hpp:197-237`), so the blocks this
/// call completes are exactly those whose trigger cell lies in
/// `[pos0, pos0 + tokens)`:
///
/// ```text
/// b0     = max(0, pos0 / idx_block)        // ceil((pos0 - 3) / 4)
/// b_last = (last - 3) / idx_block           // last = pos0 + tokens - 1
/// ```
pub fn blocks_completed(pos0: usize, tokens: usize) -> (usize, usize) {
    if tokens == 0 {
        return (0, 0);
    }
    let last = pos0 + tokens - 1;
    if last < IDX_BLOCK - 1 {
        return (0, 0);
    }
    let b0 = pos0 / IDX_BLOCK;
    let b_last = (last - (IDX_BLOCK - 1)) / IDX_BLOCK;
    if b_last < b0 {
        return (b0, 0);
    }
    (b0, b_last - b0 + 1)
}

// ---------------------------------------------------------------------------
// Benchmark support (`breccia bench ops --op qsa`)
// ---------------------------------------------------------------------------

/// Names of [`Qsa::stage_timings`]'s stages, in dispatch order.
pub const STAGES: [&str; 19] = [
    "q-gemv",
    "k-gemv",
    "v-gemv",
    "idx-q-gemv",
    "idx-k-gemv",
    "q-norm",
    "k-norm",
    "idx-q-norm",
    "rope-q",
    "rope-k",
    "rope-idx-q",
    "kv-append",
    "idx-pool",
    "idx-tail",
    "idx-score",
    "topk",
    "attn-scores+softmax",
    "attn-apply+gate",
    "out-gemv",
];

impl Qsa<'_> {
    /// Time every stage of one QSA sublayer on its own, in dispatch order.
    ///
    /// Same protocol as [`Gdn::stage_timings`](crate::gdn::Gdn::stage_timings):
    /// each stage is its own back-to-back chain with a full barrier between
    /// dispatches, and the layer total is the sum. `attend` includes the softmax
    /// dispatch and `apply` the output gate, because those are the two composite
    /// stages the graph will issue as one unit.
    ///
    /// `seed_selection` replaces the top-k's output with a full `max_ids` selection
    /// after the top-k dispatch. It exists for the benchmark's long-context decode
    /// cell: there the cache is zero-filled, every indexer score ties, `counts`
    /// comes out 0 and the two attention stages time an empty selection (launch
    /// overhead). The bench passes `true` for that cell only, for both V layouts, so
    /// the f16/q8_0 ratio is comparable. `false` is the previous behaviour, bit for
    /// bit.
    #[allow(clippy::too_many_arguments)]
    pub fn stage_timings(
        &self,
        state: &QsaState,
        ws: &QsaWorkspace,
        w: &QsaWeights<'_>,
        proj: &QsaProjections<'_, '_, '_>,
        x: &Buf,
        out: &Buf,
        pos0: usize,
        target_s: f64,
        seed_selection: bool,
    ) -> Result<[crate::GemvTiming; 19]> {
        let t = ws.tokens;
        let q = proj.q.run_timed(proj.w_q, x, &ws.q_full, t, target_s)?;
        let k = proj.k.run_timed(proj.w_k, x, &ws.k_cur, t, target_s)?;
        let v = proj.v.run_timed(proj.w_v, x, &ws.v_cur, t, target_s)?;
        let iq = proj
            .idx_q
            .run_timed(proj.w_idx_q, x, &ws.idx_q, t, target_s)?;
        let ik = proj
            .idx_k
            .run_timed(proj.w_idx_k, x, &ws.idx_k, t, target_s)?;

        let pc_q = Pc::new()
            .u((t * MODEL_N_HEAD) as u32)
            .u(MODEL_HEAD_DIM as u32)
            .u(MODEL_HEAD_DIM as u32)
            .f(crate::ops::MODEL_RMS_EPS)
            .u(MODEL_N_HEAD as u32);
        let (qgx, qgy) = grid_for(t * MODEL_N_HEAD, 1);
        let qn = self.q_norm.run_timed(
            &[&ws.q_full, w.q_norm, &ws.q],
            &pc_q,
            (qgx, qgy, 1),
            target_s,
        )?;
        let kn = self.rms_k.run_timed(
            &ws.k_cur,
            Some(w.k_norm),
            &ws.k_cur,
            t * MODEL_N_KV_HEAD,
            target_s,
        )?;
        let iqn = self.rms_idx.run_timed(
            &ws.idx_q,
            Some(w.idx_q_norm),
            &ws.idx_q,
            t * IDX_N_HEAD,
            target_s,
        )?;

        let table = &self.table_buf;
        let rq = self.rope_hd.run_at_timed(
            &ws.q,
            table,
            &state.positions,
            &ws.q,
            t * MODEL_N_HEAD,
            pos0 as i32,
            MODEL_N_HEAD,
            target_s,
        )?;
        let rk = self.rope_hd.run_at_timed(
            &ws.k_cur,
            table,
            &state.positions,
            &ws.k_cur,
            t * MODEL_N_KV_HEAD,
            pos0 as i32,
            MODEL_N_KV_HEAD,
            target_s,
        )?;
        let ri = self.rope_idx.run_at_timed(
            &ws.idx_q,
            table,
            &state.positions,
            &ws.idx_q,
            t * IDX_N_HEAD,
            pos0 as i32,
            IDX_N_HEAD,
            target_s,
        )?;

        let pc_kv = Pc::new()
            .u(t as u32)
            .i(pos0 as i32)
            .u(MODEL_N_KV_HEAD as u32)
            .u(MODEL_HEAD_DIM as u32)
            .u(PAGE_CELLS as u32)
            .u(octave_fq_mode());
        let (kv_pipe, kx, ky) = self.kv_append_target(state, t, octave_fq_mode());
        let kva = kv_pipe.run_timed(
            &[
                &state.k_pool,
                &state.v_pool,
                &state.page_table,
                &ws.k_cur,
                &ws.v_cur,
            ],
            &pc_kv,
            (kx, ky, 1),
            target_s,
        )?;

        let (b0, n_new) = blocks_completed(pos0, t);
        let pool = if n_new == 0 {
            crate::GemvTiming::default()
        } else {
            let pc_pool = Pc::new()
                .u(n_new as u32)
                .u(b0 as u32)
                .i(pos0 as i32)
                .u(t as u32)
                .u(IDX_BLOCK as u32)
                .u(IDX_DIM as u32)
                .u(MODEL_N_ROT as u32)
                .f(crate::ops::MODEL_RMS_EPS);
            let (pgx, pgy) = grid_for(n_new, 1);
            self.idx_pool.run_timed(
                &[
                    &ws.idx_k,
                    &state.idx_tail,
                    w.idx_k_norm,
                    table,
                    &state.idx_pooled,
                    &state.positions,
                ],
                &pc_pool,
                (pgx, pgy, 1),
                target_s,
            )?
        };
        let pc_tail = Pc::new()
            .u(t as u32)
            .i(pos0 as i32)
            .u(IDX_DIM as u32)
            .u(IDX_BLOCK as u32);
        let (tgx, tgy) = grid_for(t * IDX_DIM, 128);
        let tail = self.idx_tail.run_timed(
            &[&ws.idx_k, &state.idx_tail],
            &pc_tail,
            (tgx, tgy, 1),
            target_s,
        )?;
        let pc_sc = Pc::new()
            .u(t as u32)
            .u(ws.n_blocks as u32)
            .u(IDX_N_HEAD as u32)
            .u(IDX_DIM as u32)
            .i(pos0 as i32)
            .u(IDX_BLOCK as u32)
            .f(1.0 / (IDX_DIM as f32).sqrt());
        // Same kernels as the production path (tiled score, radix top-k unless
        // env picks serial); rank's multi-dispatch is timed as serial.
        let (score_k, sgrid) = if tiled_score() && t > 1 {
            let g = (ws.n_blocks.div_ceil(64) as u32, t.div_ceil(32) as u32, 1);
            (&self.idx_score_tiled, g)
        } else {
            let (sgx, sgy) = grid_for(t * ws.n_blocks, 1);
            (&self.idx_score, (sgx, sgy, 1))
        };
        let score = score_k.run_timed(
            &[&ws.idx_q, &state.idx_pooled, &ws.idx_scores],
            &pc_sc,
            sgrid,
            target_s,
        )?;
        let pc_tk = Pc::new()
            .u(t as u32)
            .u(ws.n_blocks as u32)
            .u(ws.topk_blocks as u32)
            .u(ws.max_ids as u32)
            .i(pos0 as i32)
            .u(0); // dense off
        let (tgx2, tgy2) = grid_for(t, 1);
        let topk_k = if topk_path() == TopkPath::Radix { &self.topk_radix } else { &self.topk };
        let tk = topk_k.run_timed(
            &[&ws.idx_scores, &ws.ids, &ws.counts],
            &pc_tk,
            (tgx2, tgy2, 1),
            target_s,
        )?;
        // `seed_selection`: give the attention stages a full selection instead of
        // whatever a zero-filled cache's top-k produced. The bench's cache holds no
        // real keys, so every indexer score ties and `counts` comes out 0 — which
        // made the long-context decode cell time *launch overhead* (2.29 us) rather
        // than the 2051-cell window it exists to measure. Both V layouts run the
        // same seeded selection, so the f16/q8_0 ratio is comparable.
        if seed_selection {
            let blocks = ws.max_ids.min(ws.n_blocks);
            let ids: Vec<u32> = (0..ws.max_ids)
                .map(|i| (i % ws.n_blocks.max(1)) as u32)
                .collect();
            crate::gemv::upload_into(self.q_norm.gpu(), &ws.ids, &crate::ops::u32_to_bytes(&ids))?;
            crate::gemv::upload_into(
                self.q_norm.gpu(),
                &ws.counts,
                &crate::ops::u32_to_bytes(&[blocks as u32]),
            )?;
        }

        let pc_at = Pc::new()
            .u(t as u32)
            .u(MODEL_N_HEAD as u32)
            .u(MODEL_HEAD_DIM as u32)
            .u(ws.slots as u32)
            .u(ws.max_ids as u32)
            .i(pos0 as i32)
            .u(PAGE_CELLS as u32)
            .u(MODEL_N_KV_HEAD as u32);
        let (agx, agy) = grid_for(t * MODEL_N_HEAD, 1);
        let mut attn = self.scores_targets(state).0.run_timed(
            &[
                &ws.q,
                &state.k_pool,
                &state.page_table,
                &ws.ids,
                &ws.counts,
                &ws.attn,
            ],
            &pc_at,
            (agx, agy, 1),
            target_s,
        )?;
        let sm =
            self.softmax
                .run_timed(&ws.attn, &ws.attn, MODEL_N_HEAD * t, ws.slots, target_s)?;
        attn.per_iter_s += sm.per_iter_s;
        let (apx, apy) = grid_for(t * ATTN_OUT_DIM, 256);
        let apply = self.attn_apply_target(state).run_timed(
            &[
                &ws.attn,
                &state.v_pool,
                &state.page_table,
                &ws.ids,
                &ws.counts,
                &ws.q_full,
                &ws.attn_out,
            ],
            &pc_at,
            (apx, apy, 1),
            target_s,
        )?;
        let o = proj
            .out
            .run_timed(proj.w_out, &ws.attn_out, out, t, target_s)?;
        Ok([
            q, k, v, iq, ik, qn, kn, iqn, rq, rk, ri, kva, pool, tail, score, tk, attn, apply, o,
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::kv_host_matches;

    #[test]
    fn kv_host_spec_matches_cards() {
        let (rx, mi) = ("0000:03:00.0", "0000:0e:00.0");
        assert!(kv_host_matches("1", rx) && kv_host_matches("1", mi));
        assert!(!kv_host_matches("", rx) && !kv_host_matches("0", rx));
        assert!(kv_host_matches("03:00.0", rx) && !kv_host_matches("03:00.0", mi));
        assert!(kv_host_matches("03:00.0, 0e:00.0", mi));
    }

    use super::sizes::*;

    /// Text 0..5, a 3x2 image at cell 5, text after: the image tokens are
    /// `(5, 5 + i/3, 5 + i%3)` (llama.cpp `.pos` dump layout, t y x) and the next
    /// text position is 5 + max(3, 2) = 8 at cell 11.
    #[test]
    fn mrope_triples_follow_mtmd() {
        let t = super::mrope_triples(14, &[(5, 3, 2)]).unwrap();
        let cell = |c: usize| [t[3 * c], t[3 * c + 1], t[3 * c + 2]];
        assert_eq!(cell(4), [4, 4, 4]);
        assert_eq!(cell(5), [5, 5, 5]);
        assert_eq!(cell(7), [5, 5, 7]);
        assert_eq!(cell(8), [5, 6, 5]);
        assert_eq!(cell(10), [5, 6, 7]);
        assert_eq!(cell(11), [8, 8, 8]);
        assert_eq!(cell(13), [10, 10, 10]);
        assert_eq!(super::mrope_triples(4, &[]).unwrap(), super::identity_triples(4));
        assert!(super::mrope_triples(8, &[(5, 3, 2)]).is_err());
    }

    /// The capacity argument of §8bf turns on these two numbers, so they are pinned:
    /// a q8_0 **V** pool is ~0.56x the f16 pair's V half, and the pair of pools is
    /// `2 x f16_pool_bytes` exactly.
    #[test]
    fn the_q8v_pool_is_the_ratio_the_capacity_argument_uses() {
        // 32 x i8 + f32 scale = 36 B per 32 elements vs 64 B for f16
        assert_eq!(Q8_BLOCK_BYTES, 36);
        assert_eq!(q8_blocks_per_row(), 8);
        let pages = 8;
        // `f16_pool_bytes(pages)` is **one** pool (K or V): 16 KiB per page.
        let f16_v = f16_pool_bytes(pages);
        let q8v = q8v_pool_bytes(pages);
        assert!(q8v < f16_v, "q8_0 V must be smaller: {q8v} vs {f16_v}");
        let r = q8v_vs_f16();
        assert!((0.56..0.57).contains(&r), "V ratio {r}");
        // The whole-pool (K+V) ratio is what the agent count sees: K f16 + V q8_0.
        let pair = (f16_v + q8v) as f64 / (2 * f16_v) as f64;
        assert!((0.78..0.79).contains(&pair), "K+V ratio {pair}");
        // And the per-agent arithmetic at 32K that §8bf quotes: V saves 14 MiB per block,
        // 12 blocks -> 168 MiB/agent, 933 -> 765 MiB (4 x 765 = 3.06 GiB beside 3.3).
        let pages_32k = 32768 / PAGE_CELLS;
        let save_mib =
            (f16_pool_bytes(pages_32k) - q8v_pool_bytes(pages_32k)) as f64 / (1 << 20) as f64;
        assert!(
            (13.9..14.1).contains(&save_mib),
            "per-block V saving {save_mib} MiB"
        );
        assert!(
            (167.0..169.0).contains(&(save_mib * 12.0)),
            "per-agent V saving"
        );
    }

    /// q6_0 K+V (the default): 13 words per 64 elements = 208 B per 256-dim row vs
    /// 512 B f16, so each pool is 0.40625x and the K+V pair shrinks 2.46x.
    #[test]
    fn the_q6_pool_is_0_40625_of_f16() {
        assert_eq!(q6_row_words(), 52);
        let pages = 8;
        assert_eq!(q6_pool_bytes(pages) * 32, f16_pool_bytes(pages) * 13);
    }

    /// Positional lists are checked against the index counts they are consumed by.
    #[test]
    fn the_indexer_sizes_follow_their_index_counts() {
        assert_eq!(idx_tail_bytes(), ((IDX_BLOCK - 1) * IDX_DIM * 4) as u64);
        let cells = 32768;
        let blocks = pooled_blocks(cells);
        assert_eq!(blocks, cells / 4 + 1);
        assert_eq!(pooled_idx_bytes(blocks), (blocks * 128 * 4) as u64);
    }

    #[test]
    fn a_head_dim_that_is_not_a_block_multiple_is_refused() {
        assert_eq!(MODEL_HEAD_DIM % Q8_BLOCK_ELEMS, 0);
    }

    /// The pool byte counts are consumed by index arithmetic, so they are held to
    /// the row counts the kernels compute (`row = (page*n_kv_head + kvh)*page_cells
    /// + in_page`, one row per cell per head) rather than to a literal.
    #[test]
    fn both_pool_layouts_hold_the_rows_the_kernels_index() {
        for pages in [1usize, 8, 128, 2048] {
            let rows = pool_rows(pages);
            assert_eq!(rows, pages * MODEL_N_KV_HEAD * PAGE_CELLS);
            assert_eq!(
                f16_pool_bytes(pages) as usize,
                rows * f16_row_words() * 4,
                "f16 pool, {pages} pages"
            );
            assert_eq!(
                q8v_pool_bytes(pages) as usize,
                rows * q8v_row_words() * 4,
                "q8_0 V pool, {pages} pages"
            );
            // One row of one layout, spelled out: 128 words vs 72 words.
            assert_eq!(f16_row_words(), 128);
            assert_eq!(q8v_row_words(), 72);
            assert_eq!(q8v_row_words() * 4, Q8_BLOCK_BYTES * q8_blocks_per_row());
        }
    }

    /// The 32K numbers the capacity table in §8bf/§8bg quotes, from the same
    /// functions `QsaState::new` allocates with.
    #[test]
    fn the_q8v_saving_at_32k_is_the_capacity_table() {
        let pages = 32768 / PAGE_CELLS;
        let per_block_mib =
            (f16_pool_bytes(pages) - q8v_pool_bytes(pages)) as f64 / (1 << 20) as f64;
        assert!(
            (13.9..14.1).contains(&per_block_mib),
            "per-block V saving {per_block_mib} MiB"
        );
        // 12 QSA layers: 933 MiB -> 765 MiB per agent.
        let per_agent = per_block_mib * 12.0;
        assert!(
            (167.0..169.0).contains(&per_agent),
            "per-agent V saving {per_agent} MiB"
        );
    }

    /// `to_q8_0` is the store rule, so its error is bounded by the block scale — and
    /// an all-zero block must round-trip exactly (a zero scale would make the
    /// device's `float -> int` conversion undefined).
    #[test]
    fn the_q8_0_reference_is_bounded_and_exact_on_zeros() {
        let zeros = vec![0.0f32; 64];
        assert_eq!(crate::qsa::cpu::to_q8_0(&zeros), zeros);
        let mut s = 12345u64;
        let x: Vec<f32> = (0..256)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 8.0
            })
            .collect();
        let q = crate::qsa::cpu::to_q8_0(&x);
        assert_eq!(q.len(), x.len());
        for b in 0..x.len() / Q8_BLOCK_ELEMS {
            let lo = b * Q8_BLOCK_ELEMS;
            let amax = x[lo..lo + Q8_BLOCK_ELEMS]
                .iter()
                .fold(0.0f32, |m, v| m.max(v.abs()));
            let half_lsb = amax / 127.0 / 2.0;
            for i in lo..lo + Q8_BLOCK_ELEMS {
                assert!(
                    (q[i] - x[i]).abs() <= half_lsb * 1.001,
                    "element {i}: {} vs {} (half lsb {half_lsb})",
                    q[i],
                    x[i]
                );
            }
            // A block with one value at amax must land exactly on the q8_0 grid.
            let at_max = q[lo..lo + Q8_BLOCK_ELEMS]
                .iter()
                .map(|v| v.abs())
                .fold(0.0f32, f32::max);
            assert!((at_max - amax).abs() <= half_lsb * 1.001);
        }
    }

    use super::*;

    #[test]
    fn completed_blocks_follow_the_trigger_cell() {
        // One token at a time: cell 3 completes block 0, cell 7 block 1, ...
        assert_eq!(blocks_completed(0, 1), (0, 0));
        assert_eq!(blocks_completed(3, 1), (0, 1));
        assert_eq!(blocks_completed(4, 1), (1, 0));
        assert_eq!(blocks_completed(7, 1), (1, 1));
        // A 4-token chunk starting on a block boundary completes exactly one.
        assert_eq!(blocks_completed(0, 4), (0, 1));
        // A chunk straddling a boundary completes the block it finishes.
        assert_eq!(blocks_completed(2, 4), (0, 1)); // cells 5 is the trigger
                                                    // 16 tokens from 0 complete blocks 0..3.
        assert_eq!(blocks_completed(0, 16), (0, 4));
        // 5 tokens from 0 complete block 0 only (cell 3) — 4 is not a trigger.
        assert_eq!(blocks_completed(0, 5), (0, 1));
        // Continuing at 5: cells 5..9 -> cell 7 completes block 1.
        assert_eq!(blocks_completed(5, 5), (1, 1));
    }

    #[test]
    fn selection_budget_is_the_spec_width() {
        assert_eq!(SELECT_WIDTH, 2051);
        assert_eq!(IDX_TOPK_BLOCKS * IDX_BLOCK, 2048);
        assert_eq!(page_bytes(), 32768);
    }
}
