//! GDN / SSM step — the linear-attention sublayer of the 36 non-QSA layers.
//!
//! Everything between the projections, in the order
//! `docs/specs/qwen4exp-architecture.md` §4.2 lists it:
//!
//! ```text
//!  x (2560) --GEMV--> qkv [10240] = q(2048) | k(2048) | v(6144)   (caller: Gemv/GemvBatch)
//!                   --> z   [6144]
//!                   --> beta_raw [48], alpha [48]
//!  conv:      per-channel depthwise causal conv (kernel 4) over [state(3) | qkv], then SiLU
//!             over the WHOLE 10240 before the split                      -> gdn_conv.comp
//!  L2 norm:   q and k only (v is never normalised), q also x 1/S         -> gdn_qk_l2.comp
//!  scalars:   beta = sigmoid(beta_raw), gate = softplus(alpha + ssm_dt) * ssm_a
//!                                                                        -> gdn_gate.comp
//!  recurrence:decay, rank-1 delta rule, output — state updated in place   -> gdn_step.comp
//!  tail:      rmsnorm_128(y, ssm_norm) * sigmoid(z)                      -> gdn_norm_gate.comp
//!  o = ssm_out @ tail                                                     (caller)
//! ```
//!
//! Five dispatches, no host round-trips, and the only state that survives a call
//! is [`GdnState`]: `(S=128, h_v=48, S=128)` f32 (3 MiB) plus the 3-row conv
//! history (120 KiB) — the same "pseudo-KV" the spec describes in §12.3, and
//! context-length independent.
//!
//! # Layouts (the contract an integrator needs)
//!
//! | buffer | shape | flat index |
//! |---|---|---|
//! | `state` | `(S, h_v, S)` = `(i, h, j)` | `(i*h_v + h)*S + j` |
//! | `conv_state` | `(d_conv-1, channels)` oldest first | `k*channels + c` |
//! | `qkv` | `[T, channels]` | `t*channels + q/k/v region` |
//! | `beta`/`gate` | `[T, h_v]` | `t*h_v + h` |
//! | `step out`, `z` | `[T, h_v*S]` | `t*(h_v*S) + h*S + j` |
//!
//! The state layout is spec open question Q3: **strata's** `(S, h_v, S)` with `j`
//! fastest, because a warp walking consecutive `j` reads contiguous memory for the
//! axes both the output and the update stream along. llama.cpp's `(S, S, H_v)`
//! holds the same three numbers with different strides, so a mismatch is silent —
//! [`cpu`] indexes the same formula and the conformance test pins it.
//!
//! # Per-token cost
//!
//! The recurrence reads and writes the whole state once per token (2 reads + 1
//! write of 3 MiB, `docs/phase-3/06` §latency). A prefill chunk of `T` tokens is
//! therefore `T` identical steps inside one invocation — the equality
//! `ST/include/strata/core/verify.hpp:3-11` asserts between strata's chunked and
//! single-token GDN kernels.

pub mod bench;
pub mod cpu;

use anyhow::{anyhow, Result};
use breccia_vk::ctx::{Buf, Gpu};

use crate::gemv::alloc;
use crate::ops::{grid_for, Pc, Pipe, RmsNorm};
use crate::proj::Proj;

/// `ssm.state_size` — the head width and the state's contracted axis.
pub const MODEL_S: usize = 128;
/// `ssm.group_count` — key/query heads (`h_k`).
pub const MODEL_H_K: usize = 16;
/// `ssm.time_step_rank` — value heads (`h_v`).
pub const MODEL_H_V: usize = 48;
/// `ssm.conv_kernel`.
pub const MODEL_D_CONV: usize = 4;
/// `2*h_k*S + h_v*S` = 10240: the width of `attn_qkv` and of the conv.
pub const CONV_CHANNELS: usize = 2 * MODEL_H_K * MODEL_S + MODEL_H_V * MODEL_S;
/// `h_v * S` = 6144 = `ssm.inner_size`.
pub const VALUE_DIM: usize = MODEL_H_V * MODEL_S;

const _: () = assert!(CONV_CHANNELS == 10240);
const _: () = assert!(VALUE_DIM == 6144);

/// The persistent per-layer GDN state: the delta-rule state plus the conv history.
///
/// Both live in VRAM for the life of a sequence and are updated **in place** by
/// every call; nothing else about a GDN layer survives a token.
pub struct GdnState {
    /// `(S, h_v, S)` f32, 3 MiB.
    pub state: Buf,
    /// `(d_conv-1, channels)` f32, oldest first, 120 KiB.
    pub conv: Buf,
}

impl GdnState {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        let state = alloc(gpu, (MODEL_S * MODEL_H_V * MODEL_S * 4) as u64)
            .map_err(|e| anyhow!("allocating the GDN state: {e}"))?;
        let conv = match alloc(gpu, ((MODEL_D_CONV - 1) * CONV_CHANNELS * 4) as u64) {
            Ok(b) => b,
            Err(e) => {
                gpu.destroy_buf(&state);
                return Err(anyhow!("allocating the GDN conv history: {e}"));
            }
        };
        Ok(GdnState { state, conv })
    }

    /// Bytes held (state + conv history), for the buffer plan.
    pub fn bytes(&self) -> u64 {
        self.state.size + self.conv.size
    }

    /// Zero both states: what a sequence start does (`ST/src/core/layer.cpp:220`).
    pub fn reset(&self, gpu: &Gpu) -> Result<()> {
        let z_state = vec![0u8; MODEL_S * MODEL_H_V * MODEL_S * 4];
        crate::gemv::upload_into(gpu, &self.state, &z_state)?;
        let z_conv = vec![0u8; (MODEL_D_CONV - 1) * CONV_CHANNELS * 4];
        crate::gemv::upload_into(gpu, &self.conv, &z_conv)?;
        Ok(())
    }

    pub fn release(self, gpu: &Gpu) {
        gpu.destroy_buf(&self.state);
        gpu.destroy_buf(&self.conv);
    }
}

/// The per-call scratch of one GDN layer at a given `T`.
pub struct GdnWorkspace {
    pub tokens: usize,
    /// `[T, 10240]` the projection output, convolved **in place**.
    pub qkv: Buf,
    /// `[T, 6144]` the `attn_gate` projection (the sigmoid gate `z`).
    pub z: Buf,
    /// `[T, 48]` raw `ssm_alpha`.
    pub alpha: Buf,
    /// `[T, 48]` raw `ssm_beta`.
    pub beta_raw: Buf,
    /// `[T, 48]` `sigmoid(beta_raw)`.
    pub beta: Buf,
    /// `[T, 48]` `softplus(alpha + dt) * a`.
    pub gate: Buf,
    /// `[T, 6144]` the recurrence output.
    pub step: Buf,
    /// `[T, 6144]` `rmsnorm(step) * sigmoid(z)` — the `ssm_out` GEMV's input.
    pub normed: Buf,
}

impl GdnWorkspace {
    pub fn bytes(&self) -> u64 {
        [
            &self.qkv,
            &self.z,
            &self.alpha,
            &self.beta_raw,
            &self.beta,
            &self.gate,
            &self.step,
            &self.normed,
        ]
        .iter()
        .map(|b| b.size)
        .sum()
    }
}

/// The four GDN-only weight tensors (§6.2). The projections are GEMVs the caller
/// owns; these are the ones these kernels read directly.
pub struct GdnWeights<'a> {
    /// `ssm_conv1d.weight` f32, GGML `ne = [4, 10240]`.
    pub conv: &'a Buf,
    /// `ssm_dt.bias` f32 `[48]`.
    pub dt: &'a Buf,
    /// `ssm_a` f32 `[48]`, negative.
    pub a: &'a Buf,
    /// `ssm_norm.weight` f32 `[128]`.
    pub ssm_norm: &'a Buf,
}

/// The five GDN-only dispatches, compiled once per device.
pub struct Gdn<'g> {
    conv: Pipe<'g>,
    qk_l2: Pipe<'g>,
    gate: Pipe<'g>,
    step: Pipe<'g>,
    norm_gate: Pipe<'g>,
}

impl<'g> Gdn<'g> {
    pub fn new(gpu: &'g Gpu) -> Result<Self> {
        Ok(Gdn {
            conv: Pipe::new(
                gpu,
                include_bytes!(concat!(env!("OUT_DIR"), "/gdn_conv.spv")),
                4,
                "gdn conv",
            )?,
            qk_l2: Pipe::new(
                gpu,
                include_bytes!(concat!(env!("OUT_DIR"), "/gdn_qk_l2.spv")),
                1,
                "gdn q/k L2 norm",
            )?,
            gate: Pipe::new(
                gpu,
                include_bytes!(concat!(env!("OUT_DIR"), "/gdn_gate.spv")),
                6,
                "gdn beta/gate",
            )?,
            step: Pipe::new(
                gpu,
                include_bytes!(concat!(env!("OUT_DIR"), "/gdn_step.spv")),
                5,
                "gdn delta rule",
            )?,
            norm_gate: Pipe::new(
                gpu,
                include_bytes!(concat!(env!("OUT_DIR"), "/gdn_norm_gate.spv")),
                4,
                "gdn norm+gate",
            )?,
        })
    }

    /// Allocate the per-call scratch for `tokens` tokens.
    pub fn workspace(&self, gpu: &Gpu, tokens: usize) -> Result<GdnWorkspace> {
        if tokens == 0 {
            return Err(anyhow!("gdn workspace needs tokens >= 1"));
        }
        let sizes = [
            tokens * CONV_CHANNELS,
            tokens * VALUE_DIM,
            tokens * MODEL_H_V,
            tokens * MODEL_H_V,
            tokens * MODEL_H_V,
            tokens * MODEL_H_V,
            tokens * VALUE_DIM,
            tokens * VALUE_DIM,
        ];
        let mut bufs: Vec<Buf> = Vec::with_capacity(sizes.len());
        for (i, n) in sizes.iter().enumerate() {
            match alloc(gpu, (*n * 4) as u64) {
                Ok(b) => bufs.push(b),
                Err(e) => {
                    for b in &bufs {
                        gpu.destroy_buf(b);
                    }
                    return Err(anyhow!("allocating gdn workspace slot {i}: {e}"));
                }
            }
        }
        let mut it = bufs.into_iter();
        let mut take = || it.next().expect("eight gdn buffers were allocated");
        Ok(GdnWorkspace {
            tokens,
            qkv: take(),
            z: take(),
            alpha: take(),
            beta_raw: take(),
            beta: take(),
            gate: take(),
            step: take(),
            normed: take(),
        })
    }

    pub fn release(&self, gpu: &Gpu, ws: GdnWorkspace) {
        for b in [
            &ws.qkv,
            &ws.z,
            &ws.alpha,
            &ws.beta_raw,
            &ws.beta,
            &ws.gate,
            &ws.step,
            &ws.normed,
        ] {
            gpu.destroy_buf(b);
        }
    }

    /// Depthwise causal conv (kernel 4) over `[conv_state | qkv]` + SiLU, in
    /// place on `ws.qkv`, updating `state.conv`.
    pub fn conv(&self, state: &GdnState, qkv: &Buf, w_conv: &Buf, tokens: usize) -> Result<()> {
        let pc = Pc::new()
            .u(CONV_CHANNELS as u32)
            .u(tokens as u32)
            .u(MODEL_D_CONV as u32);
        let (gx, gy) = grid_for(CONV_CHANNELS, 128);
        self.conv
            .run(&[&state.conv, qkv, w_conv, qkv], &pc, (gx, gy, 1))
    }

    pub fn conv_timed(
        &self,
        state: &GdnState,
        qkv: &Buf,
        w_conv: &Buf,
        tokens: usize,
        target_s: f64,
    ) -> Result<crate::GemvTiming> {
        let pc = Pc::new()
            .u(CONV_CHANNELS as u32)
            .u(tokens as u32)
            .u(MODEL_D_CONV as u32);
        let (gx, gy) = grid_for(CONV_CHANNELS, 128);
        self.conv
            .run_timed(&[&state.conv, qkv, w_conv, qkv], &pc, (gx, gy, 1), target_s)
    }

    /// L2-normalise q and k in place (v untouched), with the delta rule's `1/S`
    /// folded into q.
    pub fn qk_l2_norm(&self, qkv: &Buf, tokens: usize) -> Result<()> {
        let pc = Pc::new()
            .u(tokens as u32)
            .u(MODEL_H_K as u32)
            .u(MODEL_S as u32)
            .f(crate::ops::MODEL_RMS_EPS)
            .u(CONV_CHANNELS as u32);
        let (gx, gy) = grid_for(tokens * 2 * MODEL_H_K, 1);
        self.qk_l2.run(&[qkv], &pc, (gx, gy, 1))
    }

    /// `beta = sigmoid(beta_raw)`, `gate = softplus(alpha + dt) * a`.
    pub fn beta_gate(&self, ws: &GdnWorkspace, w: &GdnWeights<'_>) -> Result<()> {
        let pc = Pc::new().u(ws.tokens as u32).u(MODEL_H_V as u32);
        let (gx, gy) = grid_for(ws.tokens * MODEL_H_V, 128);
        self.gate.run(
            &[&ws.alpha, &ws.beta_raw, w.dt, w.a, &ws.beta, &ws.gate],
            &pc,
            (gx, gy, 1),
        )
    }

    /// The delta-rule recurrence for `ws.tokens` tokens, state updated in place.
    pub fn delta_step(&self, state: &GdnState, ws: &GdnWorkspace, out: &Buf) -> Result<()> {
        let pc = Pc::new()
            .u(ws.tokens as u32)
            .u(MODEL_H_K as u32)
            .u(MODEL_H_V as u32)
            .u(MODEL_S as u32)
            .u(CONV_CHANNELS as u32);
        let (gx, gy) = grid_for(MODEL_H_V * MODEL_S, 128);
        self.step.run(
            &[&state.state, &ws.qkv, &ws.gate, &ws.beta, out],
            &pc,
            (gx, gy, 1),
        )
    }

    pub fn delta_step_timed(
        &self,
        state: &GdnState,
        ws: &GdnWorkspace,
        out: &Buf,
        target_s: f64,
    ) -> Result<crate::GemvTiming> {
        let pc = Pc::new()
            .u(ws.tokens as u32)
            .u(MODEL_H_K as u32)
            .u(MODEL_H_V as u32)
            .u(MODEL_S as u32)
            .u(CONV_CHANNELS as u32);
        let (gx, gy) = grid_for(MODEL_H_V * MODEL_S, 128);
        self.step.run_timed(
            &[&state.state, &ws.qkv, &ws.gate, &ws.beta, out],
            &pc,
            (gx, gy, 1),
            target_s,
        )
    }

    /// `out = rmsnorm_128(step, ssm_norm) * sigmoid(z)`.
    pub fn norm_gate(&self, step: &Buf, gamma: &Buf, z: &Buf, out: &Buf) -> Result<()> {
        let rows = (out.size as usize / 4) / MODEL_S;
        let pc = Pc::new()
            .u(rows as u32)
            .f(crate::ops::MODEL_RMS_EPS)
            .u(MODEL_S as u32);
        let (gx, gy) = grid_for(rows, 1);
        self.norm_gate.run(&[step, gamma, z, out], &pc, (gx, gy, 1))
    }

    /// One whole GDN sublayer: the five GEMVs and the five kernels, in the
    /// order §4.2 lists them.
    ///
    /// `x` is `[T, 2560]` (the HC_READ `mixed` vector for each token) and `out`
    /// receives `[T, 2560]` (`ssm_out @ tail`), i.e. the block output the
    /// HC_WRITE consumes. Everything else is scratch inside `ws`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        state: &GdnState,
        ws: &GdnWorkspace,
        w: &GdnWeights<'_>,
        proj: &GdnProjections<'_, '_, '_>,
        x: &Buf,
        out: &Buf,
    ) -> Result<()> {
        let t = ws.tokens;
        let gpu = self.conv.gpu();
        // GPU-time probe stage boundaries (`BRECCIA_GPU_TIME`, no-op when off).
        crate::batch_cmd::probe::enter(gpu, crate::batch_cmd::probe::GDN_PROJ);
        // Projections (§4.2 first block). Marked separately (§8ax): `beta` and
        // `alpha` are 48-row matrices, i.e. 6 workgroups each, so they are a
        // latency floor per block rather than arithmetic.
        proj.qkv.run(proj.w_qkv, x, &ws.qkv, t)?;
        crate::batch_cmd::probe::enter(gpu, crate::batch_cmd::probe::GDN_Z);
        proj.z.run(proj.w_z, x, &ws.z, t)?;
        crate::batch_cmd::probe::enter(gpu, crate::batch_cmd::probe::GDN_BETA);
        proj.beta.run(proj.w_beta, x, &ws.beta_raw, t)?;
        crate::batch_cmd::probe::enter(gpu, crate::batch_cmd::probe::GDN_ALPHA);
        proj.alpha.run(proj.w_alpha, x, &ws.alpha, t)?;

        crate::batch_cmd::probe::enter(gpu, crate::batch_cmd::probe::GDN_SCAN);
        self.conv(state, &ws.qkv, w.conv, t)?;
        self.qk_l2_norm(&ws.qkv, t)?;
        self.beta_gate(ws, w)?;
        self.delta_step(state, ws, &ws.step)?;
        self.norm_gate(&ws.step, w.ssm_norm, &ws.z, &ws.normed)?;
        crate::batch_cmd::probe::enter(gpu, crate::batch_cmd::probe::GDN_OUT);
        proj.out.run(proj.w_out, &ws.normed, out, t)?;
        Ok(())
    }

    /// The `ssm_out` projection is `[6144 -> 2560]`; exposed so a caller that
    /// keeps the tail in its own layout can run it separately.
    pub fn ssm_out_rows() -> usize {
        2560
    }
}

/// The five projections of one GDN layer, with their weight buffers.
pub struct GdnProjections<'a, 'g, 'w> {
    pub qkv: Proj<'a, 'g>,
    pub w_qkv: &'w Buf,
    pub z: Proj<'a, 'g>,
    pub w_z: &'w Buf,
    pub beta: Proj<'a, 'g>,
    pub w_beta: &'w Buf,
    pub alpha: Proj<'a, 'g>,
    pub w_alpha: &'w Buf,
    pub out: Proj<'a, 'g>,
    pub w_out: &'w Buf,
}

/// `RmsNorm` pipeline shape for the GDN q/k L2 normalisation, kept for callers
/// that prefer the op over [`Gdn::qk_l2_norm`] (identical maths, one dispatch
/// per half instead of one per layer).
pub fn l2_norm_op<'g>(gpu: &'g Gpu) -> Result<RmsNorm<'g>> {
    RmsNorm::with_eps(gpu, MODEL_S, 0, crate::ops::MODEL_RMS_EPS / MODEL_S as f32)
}

// ---------------------------------------------------------------------------
// Benchmark support (`breccia bench ops --op gdn`)
// ---------------------------------------------------------------------------

/// Names of [`Gdn::stage_timings`]'s stages, in dispatch order.
pub const STAGES: [&str; 10] = [
    "qkv-gemv",
    "z-gemv",
    "beta-gemv",
    "alpha-gemv",
    "conv+silu",
    "qk-l2-norm",
    "beta-gate",
    "delta-rule",
    "norm+gate",
    "ssm-out-gemv",
];

impl Gdn<'_> {
    /// Time every stage of one GDN sublayer on its own, in dispatch order.
    ///
    /// The stages are the ones [`Gdn::forward`] issues, each measured as its own
    /// back-to-back chain with a full barrier between dispatches — the same
    /// protocol `HcMixer::stage_timings` uses, so the numbers compose with
    /// `docs/phase-3/04`'s. The **layer total is the sum**: a real graph pays the
    /// same barriers between the same dispatches, so the sum is what one GDN
    /// sublayer costs at that `T` (there is no cross-stage overlap to lose).
    #[allow(clippy::too_many_arguments)]
    pub fn stage_timings(
        &self,
        state: &GdnState,
        ws: &GdnWorkspace,
        w: &GdnWeights<'_>,
        proj: &GdnProjections<'_, '_, '_>,
        x: &Buf,
        out: &Buf,
        target_s: f64,
    ) -> Result<[crate::GemvTiming; 10]> {
        let t = ws.tokens;
        let qkv = proj.qkv.run_timed(proj.w_qkv, x, &ws.qkv, t, target_s)?;
        let z = proj.z.run_timed(proj.w_z, x, &ws.z, t, target_s)?;
        let beta = proj
            .beta
            .run_timed(proj.w_beta, x, &ws.beta_raw, t, target_s)?;
        let alpha = proj
            .alpha
            .run_timed(proj.w_alpha, x, &ws.alpha, t, target_s)?;
        let conv = self.conv_timed(state, &ws.qkv, w.conv, t, target_s)?;
        let pc_l2 = Pc::new()
            .u(t as u32)
            .u(MODEL_H_K as u32)
            .u(MODEL_S as u32)
            .f(crate::ops::MODEL_RMS_EPS)
            .u(CONV_CHANNELS as u32);
        let (gx, gy) = grid_for(t * 2 * MODEL_H_K, 1);
        let l2 = self
            .qk_l2
            .run_timed(&[&ws.qkv], &pc_l2, (gx, gy, 1), target_s)?;
        let pc_gate = Pc::new().u(t as u32).u(MODEL_H_V as u32);
        let (ggx, ggy) = grid_for(t * MODEL_H_V, 128);
        let gate = self.gate.run_timed(
            &[&ws.alpha, &ws.beta_raw, w.dt, w.a, &ws.beta, &ws.gate],
            &pc_gate,
            (ggx, ggy, 1),
            target_s,
        )?;
        let step = self.delta_step_timed(state, ws, &ws.step, target_s)?;
        let rows = t * MODEL_H_V;
        let pc_ng = Pc::new()
            .u(rows as u32)
            .f(crate::ops::MODEL_RMS_EPS)
            .u(MODEL_S as u32);
        let (ngx, ngy) = grid_for(rows, 1);
        let ng = self.norm_gate.run_timed(
            &[&ws.step, w.ssm_norm, &ws.z, &ws.normed],
            &pc_ng,
            (ngx, ngy, 1),
            target_s,
        )?;
        let outg = proj
            .out
            .run_timed(proj.w_out, &ws.normed, out, t, target_s)?;
        Ok([qkv, z, beta, alpha, conv, l2, gate, step, ng, outg])
    }
}
