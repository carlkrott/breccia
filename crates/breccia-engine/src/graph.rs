//! The forward-pass graph: 48 blocks, one token at a time.
//!
//! ```text
//! token ──▶ emb gather (4 rows = the hc copies) ──▶ R [hc, n_embd]
//! for L in 0..48:
//!   if PLE(L):  R = PLE(R, ple_emb)                       host block, GPU projections
//!   HC_READ(attn) ──▶ mixed
//!   GDN(L) | QSA(L) ──▶ block_out
//!   HC_WRITE(attn)  R += block_out · 2σ(inject/hc)
//!   HC_READ(ffn)  ──▶ mixed
//!   router ──▶ ids, w   ·   shared expert ──▶ sh[n_embd] · scalar gate
//!   ▼ one readback                     ▲ one upload
//!   host expert lane: Σ_j w_j down_j(silu(gate_j x) · up_j x)
//!   HC_WRITE(ffn)   R += block_out · 2σ(inject/hc)
//! HC_READ_FINAL ──▶ cur [n_embd] ──▶ lm_head GEMV ──▶ logits [vocab] ──▶ sampler
//! ```
//!
//! # Why the host round trip is on the critical path
//!
//! The expert lane is `breccia-kernels::cpu`'s, whose entry point takes `&[f32]`
//! and `&mut [f32]`: the MoE activation is a host vector by contract
//! (`docs/phase-3/08` §1). One block therefore costs exactly one 10 KiB readback
//! and one 10 KiB upload — two `vkWaitForFences` per block, which is what
//! [`crate::xfer::Staging`] exists to keep at two rather than ten.
//!
//! # Dispatch discipline
//!
//! Every kernel entry point in `breccia-kernels` records, submits and waits
//! (`ops::Pipe::run` → `Gpu::wall_run`), so the host is synchronous with the GPU
//! at every call boundary. That is what makes the per-stage timers honest without
//! inserting a single extra sync: each delta is the real time that stage occupies
//! on the critical path, submit overhead included.

use anyhow::{anyhow, bail, Context, Result};
use breccia_vk::ctx::Buf;

use breccia_kernels::cpu::{self, ExpertPool};
use breccia_kernels::ops::{
    f32_from_bytes, f32_to_bytes, EmbGather, HcMixer, HcWeights, HcWorkspace, Router, SwiGlu,
};
use breccia_kernels::{
    Gdn, GdnProjections, GdnState, GdnWeights, GdnWorkspace, Proj, Qsa, QsaProjections, QsaState,
    QsaWeights, QsaWorkspace, QuantType,
};

use crate::config::LayerKind;
use crate::model::{HcW, Model, PipelineSet};
use crate::ple::PleHost;
use crate::sampler::{Sampler, Sampling};
use crate::stats::{Checksum, LayerTrace};
use crate::xfer::Staging;

/// Per-stage timer slots, in graph order.
pub const STAGES: [&str; 15] = [
    "embed",
    "ple",
    "hc-read(attn)",
    "mixer",
    "hc-write(attn)",
    "hc-read(ffn)",
    "router",
    "shared-exp",
    "xfer-down",
    "experts(host)",
    "xfer-up",
    "hc-write(ffn)",
    "handoff",
    "final-hc",
    "head",
];

const EMBED: usize = 0;
const PLE: usize = 1;
const HC_READ_A: usize = 2;
const MIXER: usize = 3;
const HC_WRITE_A: usize = 4;
const HC_READ_F: usize = 5;
const ROUTER: usize = 6;
const SHARED: usize = 7;
const XFER_DOWN: usize = 8;
const EXPERTS: usize = 9;
const XFER_UP: usize = 10;
const HC_WRITE_F: usize = 11;
const HANDOFF: usize = 12;
const FINAL_HC: usize = 13;
const HEAD: usize = 14;

/// Wall-clock accounting: host-side deltas around synchronous dispatch groups.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Timers {
    pub on: bool,
    pub ns: [u64; STAGES.len()],
    #[serde(skip)]
    last: Option<std::time::Instant>,
    #[serde(skip)]
    stage: usize,
}

impl Timers {
    fn start(&mut self) {
        if self.on {
            self.stage = EMBED;
            self.last = Some(std::time::Instant::now());
        }
    }
    /// Attribute the time since the last mark to the current stage, then advance.
    fn mark(&mut self, stage: usize) {
        if !self.on {
            return;
        }
        let now = std::time::Instant::now();
        if let Some(last) = self.last {
            self.ns[self.stage] += now.duration_since(last).as_nanos() as u64;
        }
        self.last = Some(now);
        self.stage = stage;
    }
    fn stop(&mut self) {
        if !self.on {
            return;
        }
        if let Some(last) = self.last.take() {
            self.ns[self.stage] += last.elapsed().as_nanos() as u64;
        }
    }
    fn clear(&mut self) {
        self.ns = [0; STAGES.len()];
        self.last = None;
        self.stage = 0;
    }
    pub fn total_s(&self) -> f64 {
        self.ns.iter().sum::<u64>() as f64 / 1e9
    }
    /// `(stage, milliseconds per token)`, for the report table.
    pub fn table(&self, tokens: usize) -> Vec<(&'static str, f64)> {
        let n = tokens.max(1) as f64;
        STAGES
            .iter()
            .zip(self.ns.iter())
            .map(|(name, ns)| (*name, *ns as f64 / 1e6 / n))
            .collect()
    }
}

/// Everything one device owns during a session.
pub struct Venue<'m> {
    pub pipes: PipelineSet<'m>,
    pub hc: HcMixer<'m>,
    pub final_hc: HcMixer<'m>,
    /// `Some` for the whole session; taken only by `Drop`, which is why it is
    /// an `Option` (`HcMixer::release` consumes the workspace).
    pub hc_ws: Option<HcWorkspace>,
    pub gdn: Option<Gdn<'m>>,
    pub gdn_ws: Option<GdnWorkspace>,
    pub qsa: Option<Qsa<'m>>,
    pub router: Router<'m>,
    pub swiglu: SwiGlu<'m>,
    pub emb: EmbGather<'m>,
    pub emb_ple: EmbGather<'m>,
    /// The residual stack `[hc, n_embd]`.
    pub r: Buf,
    /// The sublayer + MoE input. **This is `HcWorkspace::mixed`**, i.e. what
    /// `HcMixer::read` wrote — see [`Venue::mixed`].
    pub block_out: Buf,
    pub sh: Buf,
    pub sh_gate_logit: Buf,
    pub shexp_gate: Buf,
    pub shexp_up: Buf,
    pub shexp_act: Buf,
    pub router_logits: Buf,
    pub router_ids: Buf,
    pub router_w: Buf,
    pub ple_key_out: Buf,
    pub ple_value_out: Buf,
    pub ple_emb: Buf,
    pub logits: Option<Buf>,
    pub name: String,
}

/// Session setup.
#[derive(Clone, Debug)]
pub struct SessionCfg {
    /// KV cells each QSA layer may use.
    pub ctx_capacity: usize,
    /// Host expert-lane threads (`0` = the lane's default).
    pub threads: usize,
    /// Collect per-stage timings.
    pub timers: bool,
}

impl Default for SessionCfg {
    fn default() -> Self {
        SessionCfg {
            ctx_capacity: 256,
            threads: 0,
            timers: false,
        }
    }
}

/// A loaded, ready-to-run session: pipelines, state, and the host expert pool.
pub struct Session<'m> {
    pub model: &'m Model,
    venues: Vec<Venue<'m>>,
    /// One staging buffer per device — kept out of [`Venue`] so a transfer can
    /// borrow it mutably while reading the device's own buffers.
    stagings: Vec<Staging>,
    gdn_state: Vec<Option<GdnState>>,
    qsa_state: Vec<Option<QsaState>>,
    pool: ExpertPool,
    ple: Option<PleHost>,
    sampler: Sampler,
    pub sampling: Sampling,
    pub timers: Timers,
    /// Forward passes run since the last reset (the decode denominator).
    pub forwards: usize,
    pub expert_bytes: u64,
}

impl<'m> Session<'m> {
    /// Compile every pipeline, allocate every buffer, zero the state.
    pub fn new(model: &'m Model, cfg: &SessionCfg) -> Result<Self> {
        let mc = &model.cfg;
        let head_dev = model.device_of(mc.n_layers - 1);
        let emb_ty = ty_of(model, "token_embd.weight")?;
        let ple_ty = ty_of(model, "per_layer_token_embd.weight")?;
        let mut venues = Vec::with_capacity(model.devices.len());
        let mut stagings = Vec::with_capacity(model.devices.len());
        for (i, dev) in model.devices.iter().enumerate() {
            let gpu = dev.gpu();
            let pipes = PipelineSet::build(gpu, &model.keys)?;
            let hc = HcMixer::new(gpu, QuantType::Bf16, mc.n_embd, mc.hc, mc.hc_lr, true)
                .context("compiling the hyper-connection mixer")?;
            let final_hc = HcMixer::new(gpu, QuantType::Bf16, mc.n_embd, mc.hc, mc.hc_lr, false)
                .context("compiling the final hyper-connection mixer")?;
            let hc_ws = Some(hc.workspace(gpu, 1)?);
            let mut needs_gdn = false;
            let mut needs_qsa = false;
            for b in 0..mc.n_layers {
                if model.device_of(b) != i {
                    continue;
                }
                match mc.layer_kind(b) {
                    LayerKind::Gdn => needs_gdn = true,
                    LayerKind::Qsa => needs_qsa = true,
                }
            }
            let (gdn, gdn_ws) = if needs_gdn {
                let g = Gdn::new(gpu)?;
                let ws = g.workspace(gpu, 1)?;
                (Some(g), Some(ws))
            } else {
                (None, None)
            };
            let qsa = if needs_qsa {
                Some(Qsa::new(gpu, cfg.ctx_capacity.max(8))?)
            } else {
                None
            };
            let router = Router::new(gpu, mc.n_expert, mc.n_expert_used)?;
            let swiglu = SwiGlu::new(gpu)?;
            let emb = EmbGather::new(gpu, emb_ty, mc.n_embd)?;
            let emb_ple = EmbGather::new(gpu, ple_ty, mc.ple.dim)?;
            // Big enough for the largest single readback: on the head device the
            // logits row (vocab × 4 = 993 KiB), elsewhere R 40 KiB + PLE key
            // 40 KiB + value 10 KiB.
            let stage_cap = if head_dev == i {
                mc.vocab as u64 * 4 + (64 << 10)
            } else {
                192 << 10
            };
            stagings.push(Staging::new(gpu, stage_cap)?);
            let alloc = |bytes: u64, what: &str| -> Result<Buf> {
                breccia_kernels::gemv::alloc(gpu, bytes)
                    .with_context(|| format!("allocating {what} on device {i}"))
            };
            let n = mc.n_embd as u64;
            let r = alloc(mc.hc_dim() as u64 * 4, "the residual stack")?;
            let block_out = alloc(n * 4, "block_out")?;
            let sh = alloc(n * 4, "shared-expert out")?;
            let sh_gate_logit = alloc(4, "shared-expert gate")?;
            let shexp_gate = alloc(mc.n_ff as u64 * 4, "shexp gate")?;
            let shexp_up = alloc(mc.n_ff as u64 * 4, "shexp up")?;
            let shexp_act = alloc(mc.n_ff as u64 * 4, "shexp act")?;
            let router_logits = alloc(mc.n_expert as u64 * 4, "router logits")?;
            let router_ids = alloc(router.ids_bytes(1), "router ids")?;
            let router_w = alloc(router.weights_bytes(1), "router weights")?;
            let ple_key_out = alloc(mc.hc_dim() as u64 * 4, "PLE key")?;
            let ple_value_out = alloc(n * 4, "PLE value")?;
            let ple_emb = alloc(mc.n_embd as u64 * 4, "PLE embed")?;
            let logits = if head_dev == i {
                Some(alloc(mc.vocab as u64 * 4, "logits")?)
            } else {
                None
            };
            venues.push(Venue {
                pipes,
                hc,
                final_hc,
                hc_ws,
                gdn,
                gdn_ws,
                qsa,
                router,
                swiglu,
                emb,
                emb_ple,
                r,
                block_out,
                sh,
                sh_gate_logit,
                shexp_gate,
                shexp_up,
                shexp_act,
                router_logits,
                router_ids,
                router_w,
                ple_key_out,
                ple_value_out,
                ple_emb,
                logits,
                name: dev.info.name.clone(),
            });
        }

        // Per-layer recurrent / KV state, on the device that owns the block.
        let mut gdn_state = Vec::with_capacity(mc.n_layers);
        let mut qsa_state = Vec::with_capacity(mc.n_layers);
        for b in 0..mc.n_layers {
            let gpu = model.devices[model.device_of(b)].gpu();
            match mc.layer_kind(b) {
                LayerKind::Gdn => {
                    let st = GdnState::new(gpu)
                        .with_context(|| format!("allocating the GDN state of block {b}"))?;
                    st.reset(gpu)?;
                    gdn_state.push(Some(st));
                    qsa_state.push(None);
                }
                LayerKind::Qsa => {
                    let st = QsaState::new(gpu, cfg.ctx_capacity, cfg.ctx_capacity.max(8))
                        .with_context(|| format!("allocating the QSA state of block {b}"))?;
                    gdn_state.push(None);
                    qsa_state.push(Some(st));
                }
            }
        }

        let threads = if cfg.threads == 0 {
            default_threads()
        } else {
            cfg.threads
        };
        let pool = ExpertPool::new(std::sync::Arc::clone(&model.shards), threads, cpu::path());
        let ple = match &model.ple {
            Some(p) => Some(PleHost::new(
                mc,
                &p.norm_key,
                &p.norm_query,
                &p.norm_conv,
                &p.conv,
            )?),
            None => None,
        };
        Ok(Session {
            model,
            venues,
            stagings,
            gdn_state,
            qsa_state,
            pool,
            ple,
            sampler: Sampler::new(Sampling::default().seed),
            sampling: Sampling::default(),
            timers: Timers {
                on: cfg.timers,
                ..Default::default()
            },
            forwards: 0,
            expert_bytes: 0,
        })
    }

    /// Threads the host expert lane is running.
    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    /// Zero every recurrent state and the PLE history — a new sequence.
    pub fn reset(&mut self) -> Result<()> {
        let m = self.model;
        for b in 0..m.cfg.n_layers {
            if let Some(st) = &self.gdn_state[b] {
                st.reset(m.devices[m.device_of(b)].gpu())?;
            }
        }
        if let Some(p) = self.ple.as_mut() {
            p.reset();
        }
        self.expert_bytes = 0;
        self.forwards = 0;
        self.timers.clear();
        Ok(())
    }

    /// Bytes of VRAM the session itself allocated (state + scratch).
    pub fn session_bytes(&self) -> u64 {
        let mut total = 0u64;
        for v in &self.venues {
            for b in [
                &v.r,
                &v.block_out,
                &v.sh,
                &v.sh_gate_logit,
                &v.shexp_gate,
                &v.shexp_up,
                &v.shexp_act,
                &v.router_logits,
                &v.router_ids,
                &v.router_w,
                &v.ple_key_out,
                &v.ple_value_out,
                &v.ple_emb,
            ] {
                total += b.size;
            }
            if let Some(l) = &v.logits {
                total += l.size;
            }
            if let Some(ws) = &v.hc_ws {
                total += ws.bytes();
            }
            if let Some(ws) = &v.gdn_ws {
                total += ws.bytes();
            }
        }
        total += self.stagings.iter().map(|s| s.bytes()).sum::<u64>();
        total += self
            .gdn_state
            .iter()
            .flatten()
            .map(|s| s.bytes())
            .sum::<u64>();
        total += self
            .qsa_state
            .iter()
            .flatten()
            .map(|s| s.bytes())
            .sum::<u64>();
        total
    }

    /// Run one token at `pos`; returns its logits.
    pub fn token(
        &mut self,
        tok: u32,
        pos: usize,
        debug: Option<&mut Vec<LayerTrace>>,
    ) -> Result<Vec<f32>> {
        let m = self.model;
        let mc = &m.cfg;
        if pos >= mc.n_ctx_max {
            bail!(
                "position {pos} is past the model's context limit {}",
                mc.n_ctx_max
            );
        }
        let mut trace = debug;

        // ---- embedding: 4 gathered rows ARE the hc copies of the vector ------
        self.timers.start();
        {
            let dev = m.device_of(0);
            let gpu = m.devices[dev].gpu();
            let v = &self.venues[dev];
            let table = m.token_embd()?;
            let rows = vec![tok; mc.hc];
            let staged = v.emb.gather_staged(gpu, table, &rows, &v.r)?;
            v.emb.release(gpu, &staged);
        }
        self.timers.mark(PLE);

        for block in 0..mc.n_layers {
            self.block(block, pos, tok, &mut trace)?;
        }

        // ---- the final mixer (the output norm) and the head ------------------
        self.timers.mark(FINAL_HC);
        let last = m.device_of(mc.n_layers - 1);
        let raw = {
            let gpu = m.devices[last].gpu();
            let v = &self.venues[last];
            let hw = HcWeights {
                norm: &m.out_hc_norm,
                down: &m.out_hc_down.buf,
                up: &m.out_hc_up.buf,
                inject: None,
            };
            let ws = v.hc_ws.as_ref().expect("the session's hc workspace");
            v.final_hc.read(ws, &v.r, &hw)?;
            self.timers.mark(HEAD);
            let out = v
                .logits
                .as_ref()
                .ok_or_else(|| anyhow!("the last device has no logits buffer"))?;
            v.pipes.get(&m.head.key)?.run(&m.head.buf, &ws.mixed, out)?;
            self.stagings[last].read_batch(gpu, &[(out, 0, mc.vocab * 4)])?
        };
        self.timers.stop();
        self.forwards += 1;
        let values = f32_from_bytes(&raw[0]);
        if values.len() != mc.vocab {
            bail!(
                "the head produced {} logits, expected {}",
                values.len(),
                mc.vocab
            );
        }
        // A non-finite logit is a numerical failure, not a sampling preference.
        if let Some(i) = values.iter().position(|v| !v.is_finite()) {
            bail!("logit {i} is not finite ({})", values[i]);
        }
        Ok(values)
    }

    /// One block, start to finish.
    fn block(
        &mut self,
        b: usize,
        pos: usize,
        tok: u32,
        trace: &mut Option<&mut Vec<LayerTrace>>,
    ) -> Result<()> {
        let m = self.model;
        let mc = &m.cfg;
        let dev = m.device_of(b);

        // ---- move the residual stack if the block lives elsewhere -----------
        if b > 0 && m.device_of(b - 1) != dev {
            self.timers.mark(HANDOFF);
            let prev = m.device_of(b - 1);
            let bytes = self.stagings[prev].read_batch(
                m.devices[prev].gpu(),
                &[(&self.venues[prev].r, 0, mc.hc_dim() * 4)],
            )?;
            self.stagings[dev].write(m.devices[dev].gpu(), &self.venues[dev].r, &bytes[0])?;
        }

        // ---- (A) PLE insertion (block 1 only) --------------------------------
        if mc.is_ple_layer(b) {
            self.timers.mark(PLE);
            self.ple_step(b, tok)?;
        }

        // ---- (B) HC_READ(attn) -----------------------------------------------
        self.timers.mark(HC_READ_A);
        {
            let v = &self.venues[dev];
            hc_read(v, &m.layers[b].hc_attn, &v.r)?;
        }
        let mixed_attn = self.maybe_checksum(trace.is_some(), dev, BufKind::Mixed, mc.n_embd)?;

        // ---- (C) the token mixer --------------------------------------------
        self.timers.mark(MIXER);
        match m.layers[b].kind {
            LayerKind::Gdn => self.run_gdn(b, dev)?,
            LayerKind::Qsa => self.run_qsa(b, dev, pos)?,
        }
        let mixer_out = self.maybe_checksum(trace.is_some(), dev, BufKind::BlockOut, mc.n_embd)?;

        // ---- (D) HC_WRITE(attn) ----------------------------------------------
        self.timers.mark(HC_WRITE_A);
        {
            let v = &self.venues[dev];
            let ws = v.hc_ws.as_ref().expect("the session's hc workspace");
            v.hc.write(ws, &v.r, &v.block_out, &v.r)?;
        }

        // ---- (E) HC_READ(ffn) ------------------------------------------------
        self.timers.mark(HC_READ_F);
        {
            let v = &self.venues[dev];
            hc_read(v, &m.layers[b].hc_ffn, &v.r)?;
        }
        let mixed_ffn = self.maybe_checksum(trace.is_some(), dev, BufKind::Mixed, mc.n_embd)?;

        // ---- (F) router ------------------------------------------------------
        self.timers.mark(ROUTER);
        {
            let lw = &m.layers[b];
            let v = &self.venues[dev];
            let ws = v.hc_ws.as_ref().expect("the session's hc workspace");
            v.pipes.get(&lw.moe.router.key)?.run(
                &lw.moe.router.buf,
                &ws.mixed,
                &v.router_logits,
            )?;
            v.router
                .run(&v.router_logits, &v.router_ids, &v.router_w, 1)?;
        }

        // ---- (G) shared expert + its scalar sigmoid gate ---------------------
        self.timers.mark(SHARED);
        {
            let lw = &m.layers[b];
            let v = &self.venues[dev];
            let pipes = &v.pipes;
            let ws = v.hc_ws.as_ref().expect("the session's hc workspace");
            pipes
                .get(&lw.moe.gate.key)?
                .run(&lw.moe.gate.buf, &ws.mixed, &v.shexp_gate)?;
            pipes
                .get(&lw.moe.up.key)?
                .run(&lw.moe.up.buf, &ws.mixed, &v.shexp_up)?;
            v.swiglu
                .run(&v.shexp_gate, &v.shexp_up, &v.shexp_act, mc.n_ff)?;
            pipes
                .get(&lw.moe.down.key)?
                .run(&lw.moe.down.buf, &v.shexp_act, &v.sh)?;
            pipes.get(&lw.moe.shexp_gate.key)?.run(
                &lw.moe.shexp_gate.buf,
                &ws.mixed,
                &v.sh_gate_logit,
            )?;
        }

        // ---- one readback: mixed, ids, weights, shared out, shared gate ------
        self.timers.mark(XFER_DOWN);
        let k = mc.n_expert_used;
        let ws_mixed = &self.venues[dev]
            .hc_ws
            .as_ref()
            .expect("the session's hc workspace")
            .mixed;
        let raw = self.stagings[dev].read_batch(
            m.devices[dev].gpu(),
            &[
                (ws_mixed, 0, mc.n_embd * 4),
                (&self.venues[dev].router_ids, 0, k * 4),
                (&self.venues[dev].router_w, 0, k * 4),
                (&self.venues[dev].sh, 0, mc.n_embd * 4),
                (&self.venues[dev].sh_gate_logit, 0, 4),
            ],
        )?;
        let mixed = f32_from_bytes(&raw[0]);
        let ids: Vec<usize> = raw[1]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
            .collect();
        let weights = f32_from_bytes(&raw[2]);
        let shared = f32_from_bytes(&raw[3]);
        let g = f32::from_le_bytes([raw[4][0], raw[4][1], raw[4][2], raw[4][3]]);
        let gate = 1.0 / (1.0 + (-g).exp());

        // ---- host expert lane -------------------------------------------------
        self.timers.mark(EXPERTS);
        let mut out = vec![0f32; mc.n_embd];
        let bytes = self
            .pool
            .moe_topk(&m.layers[b].experts, &ids, &weights, &mixed, &mut out)
            .with_context(|| format!("the expert lane on block {b}"))?;
        self.expert_bytes += bytes as u64;
        for i in 0..mc.n_embd {
            out[i] += shared[i] * gate;
        }
        let moe_out = if trace.is_some() {
            Some(Checksum::of(&out, 0))
        } else {
            None
        };

        // ---- upload the FFN result -------------------------------------------
        self.timers.mark(XFER_UP);
        self.stagings[dev].write(
            m.devices[dev].gpu(),
            &self.venues[dev].block_out,
            &f32_to_bytes(&out),
        )?;

        // ---- (H) HC_WRITE(ffn) -----------------------------------------------
        self.timers.mark(HC_WRITE_F);
        {
            let v = &self.venues[dev];
            let ws = v.hc_ws.as_ref().expect("the session's hc workspace");
            v.hc.write(ws, &v.r, &v.block_out, &v.r)?;
        }

        if let Some(t) = trace.as_mut() {
            let residual = self.checksum(dev, BufKind::R, mc.hc_dim())?;
            t.push(LayerTrace {
                block: b,
                kind: match m.layers[b].kind {
                    LayerKind::Gdn => "gdn".to_string(),
                    LayerKind::Qsa => "qsa".to_string(),
                },
                device: dev,
                residual,
                mixed_attn: mixed_attn.unwrap_or_default(),
                mixed_ffn: mixed_ffn.unwrap_or_default(),
                mixer_out: mixer_out.unwrap_or_default(),
                moe_out: moe_out.unwrap_or_default(),
                router_ids: ids.iter().map(|&i| i as u32).collect(),
                router_weights: weights,
                ple_gate: None,
                ple_conv: None,
            });
        }
        Ok(())
    }

    fn run_gdn(&mut self, b: usize, dev: usize) -> Result<()> {
        let m = self.model;
        let gw = m.layers[b]
            .gdn
            .as_ref()
            .ok_or_else(|| anyhow!("block {b} is GDN but has no weights"))?;
        let v = &self.venues[dev];
        let proj = GdnProjections {
            qkv: Proj::Single(v.pipes.get(&gw.qkv.key)?),
            w_qkv: &gw.qkv.buf,
            z: Proj::Single(v.pipes.get(&gw.z.key)?),
            w_z: &gw.z.buf,
            beta: Proj::Single(v.pipes.get(&gw.beta.key)?),
            w_beta: &gw.beta.buf,
            alpha: Proj::Single(v.pipes.get(&gw.alpha.key)?),
            w_alpha: &gw.alpha.buf,
            out: Proj::Single(v.pipes.get(&gw.out.key)?),
            w_out: &gw.out.buf,
        };
        let weights = GdnWeights {
            conv: &gw.conv,
            dt: &gw.dt,
            a: &gw.a,
            ssm_norm: &gw.ssm_norm,
        };
        let gdn = v
            .gdn
            .as_ref()
            .ok_or_else(|| anyhow!("device {dev} has no GDN pipelines"))?;
        let ws = v
            .gdn_ws
            .as_ref()
            .ok_or_else(|| anyhow!("device {dev} has no GDN workspace"))?;
        let state = self.gdn_state[b]
            .as_ref()
            .ok_or_else(|| anyhow!("block {b} has no GDN state"))?;
        let mixed = &v.hc_ws.as_ref().expect("the session's hc workspace").mixed;
        gdn.forward(state, ws, &weights, &proj, mixed, &v.block_out)
    }

    fn run_qsa(&mut self, b: usize, dev: usize, pos: usize) -> Result<()> {
        let m = self.model;
        let qw = m.layers[b]
            .qsa
            .as_ref()
            .ok_or_else(|| anyhow!("block {b} is QSA but has no weights"))?;
        let gpu = m.devices[dev].gpu();
        let v = &self.venues[dev];
        let qsa = v
            .qsa
            .as_ref()
            .ok_or_else(|| anyhow!("device {dev} has no QSA pipelines"))?;
        let state = self.qsa_state[b]
            .as_ref()
            .ok_or_else(|| anyhow!("block {b} has no QSA state"))?;
        let n_kv_after = pos + 1;
        if n_kv_after > state.capacity_cells {
            bail!(
                "block {b}: position {pos} needs {n_kv_after} KV cells but the cache holds {}",
                state.capacity_cells
            );
        }
        let ws: QsaWorkspace = qsa.workspace(gpu, 1, n_kv_after)?;
        let weights = QsaWeights {
            q_norm: &qw.q_norm,
            k_norm: &qw.k_norm,
            idx_q_norm: &qw.idx_q_norm,
            idx_k_norm: &qw.idx_k_norm,
        };
        let proj = QsaProjections {
            q: Proj::Single(v.pipes.get(&qw.q.key)?),
            w_q: &qw.q.buf,
            k: Proj::Single(v.pipes.get(&qw.k.key)?),
            w_k: &qw.k.buf,
            v: Proj::Single(v.pipes.get(&qw.v.key)?),
            w_v: &qw.v.buf,
            idx_q: Proj::Single(v.pipes.get(&qw.idx_q.key)?),
            w_idx_q: &qw.idx_q.buf,
            idx_k: Proj::Single(v.pipes.get(&qw.idx_k.key)?),
            w_idx_k: &qw.idx_k.buf,
            out: Proj::Single(v.pipes.get(&qw.out.key)?),
            w_out: &qw.out.buf,
        };
        let hc = v.hc_ws.as_ref().expect("the session's hc workspace");
        let res = qsa.forward(state, &ws, &weights, &proj, &hc.mixed, &v.block_out, pos);
        qsa.release(gpu, ws);
        res
    }

    /// The PLE insertion: host hash → GPU projections → host block → write back.
    fn ple_step(&mut self, b: usize, tok: u32) -> Result<()> {
        let m = self.model;
        let mc = &m.cfg;
        let raw = m
            .ple
            .as_ref()
            .ok_or_else(|| anyhow!("block {b} runs PLE but the model has no PLE tensors"))?;
        let table = m.ple_table()?;
        let dev = m.device_of(b);
        let hc_dim = mc.hc_dim();
        let n_embd = mc.n_embd;

        // 1. the host hash (§9.1): 16 table rows, head-major.
        let rows = self
            .ple
            .as_mut()
            .ok_or_else(|| anyhow!("no PLE host state"))?
            .rows_for(tok);

        // 2. fault the rows in concurrently, then gather them on the GPU.
        //
        // The PLE table is 26.8 GiB of lazily-mapped IQ4_NL, and one token needs
        // 16 scattered 90-byte rows = 16 page faults. Serialised, those cost
        // ~28 ms per token here whenever the page cache has evicted them (the
        // expert lane streams ~800 MiB/token through the same NVMe), against
        // 0.65 ms once hot. llama.cpp prefetches the same rows for the same
        // reason (`llama_prefetch_rows`, `qwen4exp.cpp:1283-1288`).
        prefetch_rows(table, &rows, m.ple_row_bytes());
        {
            let gpu = m.devices[dev].gpu();
            let v = &self.venues[dev];
            let staged = v.emb_ple.gather_staged(gpu, table, &rows, &v.ple_emb)?;
            v.emb_ple.release(gpu, &staged);
            v.pipes
                .get(&raw.key.key)?
                .run(&raw.key.buf, &v.ple_emb, &v.ple_key_out)?;
            v.pipes
                .get(&raw.value.key)?
                .run(&raw.value.buf, &v.ple_emb, &v.ple_value_out)?;
        }

        // 3. readback R, key, value; run the block on the host; write R back.
        let gpu = m.devices[dev].gpu();
        let parts = {
            let v = &self.venues[dev];
            self.stagings[dev].read_batch(
                gpu,
                &[
                    (&v.r, 0, hc_dim * 4),
                    (&v.ple_key_out, 0, hc_dim * 4),
                    (&v.ple_value_out, 0, n_embd * 4),
                ],
            )?
        };
        let r_host = f32_from_bytes(&parts[0]);
        let key = f32_from_bytes(&parts[1]);
        let value = f32_from_bytes(&parts[2]);
        let mut out = vec![0f32; hc_dim];
        self.ple
            .as_mut()
            .ok_or_else(|| anyhow!("no PLE host state"))?
            .block(&r_host, &key, &value, &mut out)?;
        self.stagings[dev].write(gpu, &self.venues[dev].r, &f32_to_bytes(&out))?;
        Ok(())
    }

    fn maybe_checksum(
        &mut self,
        wanted: bool,
        dev: usize,
        what: BufKind,
        n: usize,
    ) -> Result<Option<Checksum>> {
        if !wanted {
            return Ok(None);
        }
        Ok(Some(self.checksum(dev, what, n)?))
    }

    /// Download one of a venue's scratch buffers and digest it.
    fn checksum(&mut self, dev: usize, what: BufKind, n: usize) -> Result<Checksum> {
        let gpu = self.model.devices[dev].gpu();
        let v = &self.venues[dev];
        let buf = match what {
            BufKind::R => &v.r,
            BufKind::Mixed => &v.hc_ws.as_ref().expect("hc workspace").mixed,
            BufKind::BlockOut => &v.block_out,
        };
        let raw = self.stagings[dev].read_batch(gpu, &[(buf, 0, n * 4)])?;
        Ok(Checksum::of(&f32_from_bytes(&raw[0]), 8))
    }

    /// Prompt + generation. Decode-only timing: the prompt is fed one token at a
    /// time (there is no batched prefill path yet) and `decode_seconds` covers
    /// only the forwards that produced generated tokens.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        n: usize,
        debug: Option<&mut Vec<LayerTrace>>,
    ) -> Result<Generation> {
        if prompt.is_empty() {
            bail!("the prompt is empty: the n-gram window and the first token both need history");
        }
        self.reset()?;
        let mut trace = debug;
        let mut pos = 0usize;
        let mut logits = Vec::new();
        for &t in prompt {
            logits = self.token(t, pos, None)?;
            pos += 1;
        }
        // The prompt is not part of the decode measurement.
        self.forwards = 0;
        self.expert_bytes = 0;
        self.timers.clear();

        let mut out = Vec::with_capacity(n);
        let mut first_step_logits = Vec::new();
        let mut last_logits = logits.clone();
        let t_decode = std::time::Instant::now();
        for i in 0..n {
            let next = self
                .sampler
                .sample(&logits, &self.sampling)
                .ok_or_else(|| anyhow!("the sampler produced no token"))?;
            out.push(next);
            if i + 1 == n {
                break;
            }
            let dbg = if i == 0 { trace.take() } else { None };
            logits = self.token(next, pos, dbg)?;
            pos += 1;
            if first_step_logits.is_empty() {
                first_step_logits = logits.clone();
            }
            last_logits = logits.clone();
        }
        let decode_seconds = t_decode.elapsed().as_secs_f64();
        Ok(Generation {
            tokens: out,
            last_logits,
            first_step_logits,
            decode_seconds,
            forwards: self.forwards,
            expert_bytes: self.expert_bytes,
        })
    }
}

impl Drop for Session<'_> {
    /// Free the scratch, workspaces and state this session allocated.
    ///
    /// The kernel crate's types free what *they* own on `Drop` (pipelines,
    /// quant tables), but a `HcWorkspace` / `GdnWorkspace` / `GdnState` /
    /// `QsaState` is a plain bag of `Buf`s, and every scratch buffer is a bare
    /// `Buf`: those are the caller's to release, and a leak here shows up as a
    /// `VUID-vkDestroyDevice-device-05137` at teardown.
    fn drop(&mut self) {
        let m = self.model;
        for (i, v) in self.venues.iter_mut().enumerate() {
            let gpu = m.devices[i].gpu();
            for b in [
                &v.r,
                &v.block_out,
                &v.sh,
                &v.sh_gate_logit,
                &v.shexp_gate,
                &v.shexp_up,
                &v.shexp_act,
                &v.router_logits,
                &v.router_ids,
                &v.router_w,
                &v.ple_key_out,
                &v.ple_value_out,
                &v.ple_emb,
            ] {
                gpu.destroy_buf(b);
            }
            if let Some(l) = &v.logits {
                gpu.destroy_buf(l);
            }
            if let Some(ws) = v.hc_ws.take() {
                v.hc.release(gpu, ws);
            }
            if let Some(ws) = &v.gdn_ws {
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
        }
        for b in 0..m.cfg.n_layers {
            let gpu = m.devices[m.device_of(b)].gpu();
            if let Some(st) = &self.gdn_state[b] {
                gpu.destroy_buf(&st.state);
                gpu.destroy_buf(&st.conv);
            }
            if let Some(st) = &self.qsa_state[b] {
                for buf in [
                    &st.k_pool,
                    &st.v_pool,
                    &st.page_table,
                    &st.idx_tail,
                    &st.idx_pooled,
                    &st.positions,
                ] {
                    gpu.destroy_buf(buf);
                }
            }
        }
        for (i, s) in self.stagings.iter().enumerate() {
            s.release(m.devices[i].gpu());
        }
    }
}

/// A finished generation.
#[derive(Clone, Debug)]
pub struct Generation {
    pub tokens: Vec<u32>,
    pub last_logits: Vec<f32>,
    pub first_step_logits: Vec<f32>,
    pub decode_seconds: f64,
    /// Forward passes run after the prompt (the decode denominator).
    pub forwards: usize,
    pub expert_bytes: u64,
}

impl Generation {
    pub fn tokens_per_s(&self) -> f64 {
        if self.decode_seconds <= 0.0 || self.forwards == 0 {
            return 0.0;
        }
        self.forwards as f64 / self.decode_seconds
    }
    pub fn ms_per_token(&self) -> f64 {
        if self.forwards == 0 {
            return 0.0;
        }
        self.decode_seconds * 1e3 / self.forwards as f64
    }
    /// Expert weight bytes read per generated token.
    pub fn bytes_per_token(&self) -> u64 {
        if self.forwards == 0 {
            0
        } else {
            self.expert_bytes / self.forwards as u64
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum BufKind {
    R,
    Mixed,
    BlockOut,
}

/// Touch one byte of each of `rows` from up to four threads, so 16 page faults
/// cost one fault latency instead of sixteen.
fn prefetch_rows(table: &[u8], rows: &[u32], row_bytes: usize) {
    if rows.len() < 2 || table.is_empty() {
        return;
    }
    let threads = rows.len().min(4);
    let per = rows.len().div_ceil(threads);
    std::thread::scope(|s| {
        for chunk in rows.chunks(per) {
            s.spawn(move || {
                let mut sink = 0u8;
                for r in chunk {
                    let at = (*r as usize).saturating_mul(row_bytes);
                    if at < table.len() {
                        // SAFETY: `at` is inside the mapping; a volatile read
                        // only stops the optimiser from dropping the touch.
                        sink ^= unsafe { std::ptr::read_volatile(table.as_ptr().add(at)) };
                    }
                }
                std::hint::black_box(sink);
            });
        }
    });
}

fn hc_read(v: &Venue<'_>, w: &HcW, r: &Buf) -> Result<()> {
    let hw = HcWeights {
        norm: &w.norm,
        down: &w.down.buf,
        up: &w.up.buf,
        inject: w.inject.as_ref(),
    };
    let ws = v.hc_ws.as_ref().expect("the session's hc workspace");
    v.hc.read(ws, r, &hw)
}

/// Physical cores / 2, capped at 16 — the same default the host lane's own
/// benchmark uses (`breccia-kernels::cpu::bench::default_threads`, which is not
/// exported, so the rule is restated here).
fn default_threads() -> usize {
    let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") else {
        return 8;
    };
    let mut physical = String::new();
    let mut seen: std::collections::BTreeSet<(String, String)> = std::collections::BTreeSet::new();
    for line in text.lines() {
        if let Some((k, v)) = line.split_once(':') {
            match k.trim() {
                "physical id" => physical = v.trim().to_string(),
                "core id" => {
                    seen.insert((physical.clone(), v.trim().to_string()));
                }
                _ => {}
            }
        }
    }
    (seen.len().max(1) / 2).clamp(1, 16)
}

/// The quant type of one named tensor.
fn ty_of(m: &Model, name: &str) -> Result<QuantType> {
    let t = m
        .shards
        .tensor(name)
        .ok_or_else(|| anyhow!("missing tensor `{name}`"))?;
    QuantType::from_ggml(t.ty())
        .ok_or_else(|| anyhow!("{name}: ggml type {} has no kernel", t.ty().id()))
}

impl Venue<'_> {
    /// One line for the banner.
    pub fn summary(&self) -> String {
        format!("{} · {} compiled pipelines", self.name, self.pipes.len())
    }
}

/// A `Gpu`-independent check that the head is the shape the graph assumes.
pub fn assert_head(m: &Model) -> Result<()> {
    let want = m.cfg.vocab;
    if m.head.key.rows != want {
        bail!(
            "the head has {} rows, expected {want} (= token_embd.ne1)",
            m.head.key.rows
        );
    }
    if m.head.key.cols != m.cfg.n_embd {
        bail!(
            "the head has {} columns, expected {}",
            m.head.key.cols,
            m.cfg.n_embd
        );
    }
    Ok(())
}
