//! One projection, two dispatch shapes: the decode GEMV and the batched prefill
//! form.
//!
//! The GDN and QSA sublayers both need the same thing — "run `W·x` for this
//! layer's activation" — at `T = 1` (decode, one [`crate::Gemv`] dispatch) and at
//! `T > 1` (a prefill chunk, one [`crate::GemvBatch`] dispatch with the tokens as
//! the batch of "experts"). Wrapping the choice here keeps the sublayer code
//! shape-agnostic and makes the `T > 1` path explicit rather than a per-token
//! loop that would re-read the weights `T` times per dispatch.
//!
//! The batched form's output layout is `out[t*rows + r]`, i.e. `[T, rows]`
//! token-major — exactly what every consuming kernel here expects.

use anyhow::{anyhow, Result};
use breccia_vk::ctx::Buf;

use crate::gemv::GemvTiming;
use crate::{Gemv, GemvBatch, XLayout};

/// A projection, resolved for the batch size the caller is about to use.
pub enum Proj<'a, 'g> {
    /// Decode GEMV (`T = 1`): `out[rows] = W·x`.
    Single(&'a Gemv<'g>),
    /// Batched GEMV: `out[t*rows + r] = W_r · x[t]`, one dispatch for all tokens.
    Batch {
        pipe: &'a GemvBatch<'g>,
        map: &'a Buf,
    },
}

impl Proj<'_, '_> {
    pub fn run(&self, w: &Buf, x: &Buf, out: &Buf, tokens: usize) -> Result<()> {
        match self {
            Proj::Single(g) => {
                if tokens != 1 {
                    return Err(anyhow!(
                        "Proj::Single is the decode path (T = 1); pass Proj::Batch for T = {tokens}"
                    ));
                }
                g.run(w, x, out)
            }
            Proj::Batch { pipe, map } => pipe.run(w, x, out, map, tokens),
        }
    }

    pub fn run_timed(
        &self,
        w: &Buf,
        x: &Buf,
        out: &Buf,
        tokens: usize,
        target_s: f64,
    ) -> Result<GemvTiming> {
        match self {
            Proj::Single(g) => {
                if tokens != 1 {
                    return Err(anyhow!(
                        "Proj::Single is the decode path (T = 1); pass Proj::Batch for T = {tokens}"
                    ));
                }
                g.run_timed(w, x, out, target_s)
            }
            Proj::Batch { pipe, map } => pipe.run_timed(w, x, out, map, tokens, target_s),
        }
    }

    /// Floats written per token (`out` stride).
    pub fn rows(&self) -> usize {
        match self {
            Proj::Single(g) => g.rows(),
            Proj::Batch { pipe, .. } => pipe.rows_per_expert(),
        }
    }
}

/// The per-token activation map a [`Proj::Batch`] needs: every "expert" shares
/// weight base 0 and owns one activation row group, so the map is a function of
/// `T` alone and can be built once per batch shape.
pub fn batch_map(pipe: &GemvBatch<'_>, tokens: usize) -> Vec<u32> {
    pipe.map_for_blobs(&vec![0u32; tokens], XLayout::PerExpert)
}

/// Build and upload the map for `tokens` tokens, ready for [`Proj::Batch`].
pub fn batch_map_buf(
    gpu: &breccia_vk::ctx::Gpu,
    pipe: &GemvBatch<'_>,
    tokens: usize,
) -> Result<Buf> {
    let map = batch_map(pipe, tokens);
    crate::gemv::upload(gpu, &crate::batch::map_to_bytes(&map))
}
