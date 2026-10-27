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
/// The spec's selection width, `min(n_kv, idx_top_k + idx_block - 1)`.
pub const SELECT_WIDTH: usize = IDX_TOP_K + IDX_BLOCK - 1;
/// Cells per KV page (this module's choice, spec §7 / `docs/phase-3/05` §7).
pub const PAGE_CELLS: usize = 16;
/// `attn_q` output width: 24 heads x (256 q + 256 gate).
pub const QGATE_DIM: usize = MODEL_N_HEAD * 2 * MODEL_HEAD_DIM;
/// `attn_output` input width: 24 x 256.
pub const ATTN_OUT_DIM: usize = MODEL_N_HEAD * MODEL_HEAD_DIM;

const _: () = assert!(SELECT_WIDTH == 2051);
const _: () = assert!(QGATE_DIM == 12288);
const _: () = assert!(ATTN_OUT_DIM == 6144);

/// Bytes of one 16-cell page of one layer, K and V together (32 KiB).
pub fn page_bytes() -> u64 {
    (2 * MODEL_N_KV_HEAD * PAGE_CELLS * MODEL_HEAD_DIM * 2) as u64
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
    /// Identity position table for the whole window (`pos[i] = i`), so no call
    /// ever re-uploads positions.
    pub positions: Buf,
    /// Cells the cache can hold (a multiple of `PAGE_CELLS`).
    pub capacity_cells: usize,
    /// Position table length (the rope table's `max_pos`).
    pub max_pos: usize,
}

impl QsaState {
    /// Allocate for `capacity_cells` cells (rounded up to whole pages) and a rope
    /// table of `max_pos` positions.
    pub fn new(gpu: &Gpu, capacity_cells: usize, max_pos: usize) -> Result<Self> {
        if capacity_cells == 0 || max_pos == 0 {
            return Err(anyhow!(
                "qsa state needs capacity_cells > 0 and max_pos > 0"
            ));
        }
        let pages = capacity_cells.div_ceil(PAGE_CELLS);
        let cells = pages * PAGE_CELLS;
        let pool_words = (pages * MODEL_N_KV_HEAD * PAGE_CELLS * MODEL_HEAD_DIM / 2) as u64;
        let max_blocks = cells / IDX_BLOCK + 1;

        let sizes = [
            (pool_words * 4, "kv k_pool"),
            (pool_words * 4, "kv v_pool"),
            (pages as u64 * 4, "kv page_table"),
            (((IDX_BLOCK - 1) * IDX_DIM * 4) as u64, "indexer tail"),
            ((max_blocks * IDX_DIM * 4) as u64, "indexer pooled"),
            (max_pos as u64 * 4, "position table"),
        ];
        let mut bufs: Vec<Buf> = Vec::with_capacity(sizes.len());
        for (bytes, what) in sizes.iter() {
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
        let mut it = bufs.into_iter();
        let mut take = || it.next().expect("seven qsa state buffers were allocated");
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
        let pos: Vec<u32> = (0..max_pos as u32).collect();
        let zeros = vec![0u8; pool_words as usize * 4];
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
            if let Err(e) = crate::gemv::upload_into(gpu, dst, &zeros) {
                for b in all {
                    gpu.destroy_buf(b);
                }
                return Err(anyhow!("zeroing {what}: {e}"));
            }
        }
        Ok(QsaState {
            k_pool,
            v_pool,
            page_table,
            idx_tail,
            idx_pooled,
            positions,
            capacity_cells: cells,
            max_pos,
        })
    }

    pub fn bytes(&self) -> u64 {
        [
            &self.k_pool,
            &self.v_pool,
            &self.page_table,
            &self.idx_tail,
            &self.idx_pooled,
            &self.positions,
        ]
        .iter()
        .map(|b| b.size)
        .sum()
    }

    /// Pooled-key rows the state can hold.
    pub fn max_blocks(&self) -> usize {
        self.capacity_cells / IDX_BLOCK + 1
    }

    pub fn release(self, gpu: &Gpu) {
        for b in [
            &self.k_pool,
            &self.v_pool,
            &self.page_table,
            &self.idx_tail,
            &self.idx_pooled,
            &self.positions,
        ] {
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
    /// `[24*T, slots]` attention scores, softmaxed in place.
    pub attn: Buf,
    /// `[T, 24, 256]` gated attention output (the `attn_output` GEMV's input).
    pub attn_out: Buf,
    pub n_blocks: usize,
    pub slots: usize,
    pub max_ids: usize,
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
            &self.attn,
            &self.attn_out,
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
    idx_tail: Pipe<'g>,
    idx_pool: Pipe<'g>,
    idx_score: Pipe<'g>,
    topk: Pipe<'g>,
    attn_scores: Pipe<'g>,
    attn_apply: Pipe<'g>,
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
        let idx_tail = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_idx_tail.spv")),
            2,
            "qsa indexer tail",
        )?;
        let idx_pool = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_idx_pool.spv")),
            5,
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
        let attn_scores = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_attn_scores.spv")),
            6,
            "qsa attention scores",
        )?;
        let attn_apply = Pipe::new(
            gpu,
            include_bytes!(concat!(env!("OUT_DIR"), "/qsa_attn_apply.spv")),
            7,
            "qsa attention apply",
        )?;
        let softmax = Softmax::new(gpu)?;
        let rope_hd = Rope::new(gpu, MODEL_HEAD_DIM, MODEL_N_ROT, max_pos)?;
        let rope_idx = Rope::new(gpu, IDX_DIM, MODEL_N_ROT, max_pos)?;
        let rms_k = RmsNorm::new(gpu, MODEL_HEAD_DIM, 1)?;
        let rms_idx = RmsNorm::new(gpu, IDX_DIM, 1)?;
        let table = RopeTable::model(max_pos)?;
        let table_buf = table.upload(gpu)?;
        Ok(Qsa {
            q_norm,
            kv_append,
            idx_tail,
            idx_pool,
            idx_score,
            topk,
            attn_scores,
            attn_apply,
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
        let max_ids = n_blocks.min(IDX_TOPK_BLOCKS) + 1;
        let slots = 4 * max_ids;
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
            MODEL_N_HEAD * tokens * slots,
            tokens * ATTN_OUT_DIM,
        ];
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
        let mut take = || it.next().expect("eleven qsa buffers were allocated");
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
            attn: take(),
            attn_out: take(),
            n_blocks,
            slots,
            max_ids,
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
            &ws.attn,
            &ws.attn_out,
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

    /// Append this call's K/V to the paged f16 cache.
    pub fn kv_append(&self, state: &QsaState, ws: &QsaWorkspace, pos0: usize) -> Result<()> {
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
            .u(PAGE_CELLS as u32);
        let (gx, gy) = grid_for(ws.tokens * MODEL_N_KV_HEAD * (MODEL_HEAD_DIM / 2), 128);
        self.kv_append.run(
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
            ],
            &pc,
            (gx, gy, 1),
        )
    }

    /// Store this call's raw indexer keys into the rolling 3-slot tail.
    pub fn idx_tail(&self, state: &QsaState, ws: &QsaWorkspace, pos0: usize) -> Result<()> {
        let pc = Pc::new()
            .u(ws.tokens as u32)
            .i(pos0 as i32)
            .u(IDX_DIM as u32)
            .u(IDX_BLOCK as u32);
        let (gx, gy) = grid_for(ws.tokens * IDX_DIM, 128);
        self.idx_tail
            .run(&[&ws.idx_k, &state.idx_tail], &pc, (gx, gy, 1))
    }

    /// Score every completed block against every token's indexer query.
    pub fn idx_score(&self, state: &QsaState, ws: &QsaWorkspace, pos0: usize) -> Result<()> {
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
    pub fn topk(&self, ws: &QsaWorkspace, pos0: usize) -> Result<()> {
        let pc = Pc::new()
            .u(ws.tokens as u32)
            .u(ws.n_blocks as u32)
            .u(IDX_TOPK_BLOCKS as u32)
            .u(ws.max_ids as u32)
            .i(pos0 as i32);
        let (gx, gy) = grid_for(ws.tokens, 1);
        self.topk
            .run(&[&ws.idx_scores, &ws.ids, &ws.counts], &pc, (gx, gy, 1))
    }

    /// Scores over the selected cells, then softmax in place.
    pub fn attend(&self, state: &QsaState, ws: &QsaWorkspace, pos0: usize) -> Result<()> {
        let pc = Pc::new()
            .u(ws.tokens as u32)
            .u(MODEL_N_HEAD as u32)
            .u(MODEL_HEAD_DIM as u32)
            .u(ws.slots as u32)
            .u(ws.max_ids as u32)
            .i(pos0 as i32)
            .u(PAGE_CELLS as u32)
            .u(MODEL_N_KV_HEAD as u32);
        let units = ws.tokens * MODEL_N_HEAD;
        let (gx, gy) = grid_for(units, 1);
        self.attn_scores.run(
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

    /// Probability-weighted V, gated by `sigmoid(gate)` from the `attn_q` half.
    pub fn apply(&self, state: &QsaState, ws: &QsaWorkspace, pos0: usize) -> Result<()> {
        let pc = Pc::new()
            .u(ws.tokens as u32)
            .u(MODEL_N_HEAD as u32)
            .u(MODEL_HEAD_DIM as u32)
            .u(ws.slots as u32)
            .u(ws.max_ids as u32)
            .i(pos0 as i32)
            .u(PAGE_CELLS as u32)
            .u(MODEL_N_KV_HEAD as u32);
        let (gx, gy) = grid_for(ws.tokens * ATTN_OUT_DIM, 256);
        self.attn_apply.run(
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
        self.idx_score(state, ws, pos0)?;
        self.topk(ws, pos0)?;
        crate::batch_cmd::probe::enter(gpu, crate::batch_cmd::probe::ATTN);
        self.attend(state, ws, pos0)?;
        self.apply(state, ws, pos0)?;
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
            .u(PAGE_CELLS as u32);
        let (kx, ky) = grid_for(t * MODEL_N_KV_HEAD * (MODEL_HEAD_DIM / 2), 128);
        let kva = self.kv_append.run_timed(
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
        let (sgx, sgy) = grid_for(t * ws.n_blocks, 1);
        let score = self.idx_score.run_timed(
            &[&ws.idx_q, &state.idx_pooled, &ws.idx_scores],
            &pc_sc,
            (sgx, sgy, 1),
            target_s,
        )?;
        let pc_tk = Pc::new()
            .u(t as u32)
            .u(ws.n_blocks as u32)
            .u(IDX_TOPK_BLOCKS as u32)
            .u(ws.max_ids as u32)
            .i(pos0 as i32);
        let (tgx2, tgy2) = grid_for(t, 1);
        let tk = self.topk.run_timed(
            &[&ws.idx_scores, &ws.ids, &ws.counts],
            &pc_tk,
            (tgx2, tgy2, 1),
            target_s,
        )?;

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
        let mut attn = self.attn_scores.run_timed(
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
        let apply = self.attn_apply.run_timed(
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
