//! `inkling2_loop` forward pass (S11 of `docs/specs/inkling2-loop-port.md`).
//!
//! A straight port of llama.cpp `src/models/inkling2.cpp` `build_layer` /
//! `build_output` and `inkling2-loop.cpp`'s pass driver onto the generic
//! Breccia ops (S4-S10). Prefill only: one fresh sequence, positions `0..T`,
//! f32 K/V held for the call (KV append for decode is S13).
//!
//! Every shape comes from [`InklingConfig`] or the tensor dims, so a sibling
//! checkpoint with other widths, loop bounds or block size runs unchanged.

use std::collections::HashMap;
use std::ops::Deref;

use anyhow::{anyhow, ensure, Context, Result};
use breccia_format::ShardSet;
use breccia_kernels::ops::{
    self, Act, ActMode, Add, AttnDense, AttnShape, ControlWeights, DepthMix, EmbGather, Glu,
    GluMode, LowRankControl, RelBias, RmsNorm, Router, RowClip, ShortConv, Sink,
};
use breccia_kernels::{gemv, proj, GemvBatch, MoeFanIn, QuantType, XLayout};
use breccia_vk::ctx::{Buf, Gpu};

use crate::inkling::InklingConfig;

/// Called with `(node name, slot, values)` for every golden-named node (slot
/// -1 = global). Returning `Some` replaces the node (teacher forcing).
pub type Tap<'t> = dyn FnMut(&str, i32, &[f32]) -> Option<Vec<f32>> + 't;

/// A device buffer freed on drop.
pub struct Owned<'g>(&'g Gpu, Buf);

impl Deref for Owned<'_> {
    type Target = Buf;
    fn deref(&self) -> &Buf {
        &self.1
    }
}

impl Drop for Owned<'_> {
    fn drop(&mut self) {
        self.0.destroy_buf(&self.1);
    }
}

const CTRL: [&str; 4] = ["q", "k", "v", "ffn"];

pub struct InklingForward<'g> {
    gpu: &'g Gpu,
    shards: &'g ShardSet,
    cfg: InklingConfig,
    /// Batched GEMV per `(type, rows, cols)`.
    gemvs: HashMap<(QuantType, usize, usize), GemvBatch<'g>>,
    norm: RmsNorm<'g>,
    head_norm: RmsNorm<'g>,
    mix: DepthMix<'g>,
    ctrl: LowRankControl<'g>,
    conv: ShortConv<'g>,
    attn: AttnDense<'g>,
    add: Add<'g>,
    act: Act<'g>,
    glu: Glu<'g>,
    router: Router<'g>,
    fanin: MoeFanIn<'g>,
    emb: EmbGather<'g>,
    clip_q: RowClip<'g>,
    clip_q_tau: RowClip<'g>,
    clip_k: RowClip<'g>,
    clip_r: RowClip<'g>,
    clip_r_tau: RowClip<'g>,
}

impl<'g> InklingForward<'g> {
    pub fn new(gpu: &'g Gpu, shards: &'g ShardSet, cfg: InklingConfig) -> Result<Self> {
        let (n, hd, h, hkv) = (cfg.n_embd, cfg.head_dim, cfg.n_head, cfg.n_head_kv);
        let tau = Some((cfg.log_scale_floor, cfg.log_scale_alpha));
        let emb_ty = Self::ty_of(shards, "token_embd.weight")?;
        Ok(InklingForward {
            gpu,
            shards,
            gemvs: HashMap::new(),
            norm: RmsNorm::with_eps(gpu, n, 1, cfg.eps)?,
            head_norm: RmsNorm::with_eps(gpu, hd, 0, cfg.eps)?,
            mix: DepthMix::new(gpu, n, 1e-6)?,
            ctrl: LowRankControl::new(gpu, n, cfg.loop_rank, 1e-6)?,
            conv: ShortConv::new(gpu)?,
            attn: AttnDense::new(gpu)?,
            add: Add::new(gpu)?,
            act: Act::new(gpu, ActMode::TanhCap)?,
            glu: Glu::new(
                gpu,
                GluMode::Situ {
                    beta: cfg.situ_beta,
                    lb: cfg.situ_linear_beta,
                },
            )?,
            router: Router::sigmoid(
                gpu,
                cfg.n_expert,
                cfg.n_expert_used,
                1e-20,
                cfg.expert_weights_scale,
            )?,
            fanin: MoeFanIn::new(gpu)?,
            emb: EmbGather::new(gpu, emb_ty, n)?,
            clip_q: RowClip::new(gpu, hd, h, cfg.qk_clip, None)?,
            clip_q_tau: RowClip::new(gpu, hd, h, cfg.qk_clip, tau)?,
            clip_k: RowClip::new(gpu, hd, hkv, cfg.qk_clip, None)?,
            clip_r: RowClip::new(gpu, cfg.n_rel, h, cfg.rel_clip, None)?,
            clip_r_tau: RowClip::new(gpu, cfg.n_rel, h, cfg.rel_clip, tau)?,
            cfg,
        })
    }

    pub fn config(&self) -> &InklingConfig {
        &self.cfg
    }

    fn ty_of(shards: &ShardSet, name: &str) -> Result<QuantType> {
        let t = shards
            .tensor(name)
            .ok_or_else(|| anyhow!("missing tensor `{name}`"))?;
        QuantType::from_ggml(t.ty()).ok_or_else(|| anyhow!("`{name}`: no kernel for its type"))
    }

    fn buf(&self, floats: usize) -> Result<Owned<'g>> {
        Ok(Owned(self.gpu, gemv::alloc(self.gpu, (floats * 4) as u64)?))
    }

    fn bytes(&self, b: &[u8]) -> Result<Owned<'g>> {
        Ok(Owned(self.gpu, gemv::upload(self.gpu, b)?))
    }

    fn f32s(&self, v: &[f32]) -> Result<Owned<'g>> {
        Ok(Owned(self.gpu, ops::upload_f32(self.gpu, v)?))
    }

    fn u32s(&self, v: &[u32]) -> Result<Owned<'g>> {
        self.bytes(&ops::u32_to_bytes(v))
    }

    /// Raw GGUF bytes of `name` on the device (dims, type).
    fn weight(&self, name: &str) -> Result<(Owned<'g>, Vec<usize>, QuantType)> {
        let t = self
            .shards
            .tensor(name)
            .ok_or_else(|| anyhow!("missing tensor `{name}`"))?;
        let ty = QuantType::from_ggml(t.ty()).ok_or_else(|| anyhow!("`{name}`: no kernel"))?;
        let bytes = t.as_slice();
        ensure!(bytes.len() as u64 == t.n_bytes(), "`{name}` is truncated");
        let dims = t.dims().iter().map(|&d| d as usize).collect();
        Ok((
            self.bytes(bytes)
                .with_context(|| format!("uploading {name}"))?,
            dims,
            ty,
        ))
    }

    fn has(&self, name: &str) -> bool {
        self.shards.tensor(name).is_some()
    }

    fn gemv(&mut self, ty: QuantType, rows: usize, cols: usize) -> Result<&GemvBatch<'g>> {
        use std::collections::hash_map::Entry;
        match self.gemvs.entry((ty, rows, cols)) {
            Entry::Occupied(e) => Ok(e.into_mut()),
            Entry::Vacant(e) => {
                let g = GemvBatch::new(self.gpu, ty, rows, cols)?;
                // activations are packed [T, cols]; the map's row stride is padded_cols
                ensure!(
                    g.padded_cols() == cols,
                    "{ty:?} {rows}x{cols}: padded cols {} != cols",
                    g.padded_cols()
                );
                Ok(e.insert(g))
            }
        }
    }

    /// `out[T, rows] = W . x[T, cols]` for a dense 2-D tensor.
    fn proj(&mut self, name: &str, x: &Buf, tokens: usize) -> Result<(Owned<'g>, usize)> {
        let (w, dims, ty) = self.weight(name)?;
        ensure!(dims.len() == 2, "`{name}` is not 2-D");
        let (cols, rows) = (dims[0], dims[1]);
        let out = self.buf(tokens * rows)?;
        let gpu = self.gpu;
        let g = self.gemv(ty, rows, cols)?;
        let map = Owned(gpu, proj::batch_map_buf(gpu, g, tokens)?);
        g.run(&w, x, &out, &map, tokens)?;
        Ok((out, rows))
    }

    /// Download for the tap, and upload its replacement if it returns one.
    fn tap(
        &self,
        tap: &mut Option<&mut Tap>,
        name: &str,
        slot: i32,
        b: &Buf,
        n: usize,
    ) -> Result<()> {
        if let Some(t) = tap.as_mut() {
            let v = ops::download_f32(self.gpu, b, n)?;
            if let Some(r) = t(name, slot, &v) {
                ensure!(
                    r.len() == n,
                    "tap {name}-{slot}: {} values, want {n}",
                    r.len()
                );
                gemv::upload_into(self.gpu, b, bytemuck_f32(&r))?;
            }
        }
        Ok(())
    }

    fn copy(&self, dst: &Buf, dst_floats: usize, src: &Buf, floats: usize) -> Result<()> {
        gemv::copy_at(
            self.gpu,
            dst,
            (dst_floats * 4) as u64,
            src,
            0,
            (floats * 4) as u64,
        )
    }

    fn dup(&self, src: &Buf, floats: usize) -> Result<Owned<'g>> {
        let b = self.buf(floats)?;
        self.copy(&b, 0, src, floats)?;
        Ok(b)
    }

    /// Depth mix of `count` stack slots from `first` plus `cur` with `query`.
    #[allow(clippy::too_many_arguments)]
    fn depth_mix(
        &self,
        stack: &Buf,
        first: usize,
        count: usize,
        cur: &Buf,
        query: &str,
        tokens: usize,
    ) -> Result<(Owned<'g>, usize)> {
        let (q, dims, _) = self.weight(query)?;
        let nq = dims.get(1).copied().unwrap_or(1);
        let out = self.buf(tokens * nq * self.cfg.n_embd)?;
        self.mix.run(
            stack,
            first,
            count,
            tokens * self.cfg.n_embd,
            cur,
            &q,
            nq,
            &out,
            tokens,
        )?;
        Ok((out, nq))
    }

    /// Low-rank control of a `[T, n]` view by message column `c` of `msgs` `[T, 4, n]`.
    fn control(
        &self,
        layer: usize,
        c: usize,
        cur: &Buf,
        cv: (usize, usize),
        msgs: &Buf,
        tokens: usize,
    ) -> Result<Owned<'g>> {
        let n = self.cfg.n_embd;
        let p = format!("blk.{layer}.loop_{}_ctrl_", CTRL[c]);
        let (down, ..) = self.weight(&format!("{p}down.weight"))?;
        let (gain, ..) = self.weight(&format!("{p}gain.weight"))?;
        let (forget, ..) = self.weight(&format!("{p}forget.weight"))?;
        let out = self.buf(tokens * n)?;
        let ws = self.ctrl.workspace(self.gpu, tokens)?;
        let r = self.ctrl.run(
            &ws,
            cur,
            cv,
            msgs,
            (CTRL.len() * n, c * n),
            &ControlWeights {
                down: &down,
                gain: &gain,
                forget: &forget,
            },
            &out,
        );
        self.ctrl.release(self.gpu, ws);
        r?;
        Ok(out)
    }

    /// Prefill `ids` and return the last token's logits (`vocab` values).
    pub fn prefill(&mut self, ids: &[u32], mut tap: Option<&mut Tap>) -> Result<Vec<f32>> {
        let t = ids.len();
        ensure!(t > 0, "empty prompt");
        let cfg = self.cfg.clone();
        let n = cfg.n_embd;
        let plan = cfg.plan();
        let tn = t * n;

        // input: embedding gather + token_embd_norm
        let raw = self.buf(tn)?;
        let emb_table = self
            .shards
            .tensor("token_embd.weight")
            .ok_or_else(|| anyhow!("missing token_embd"))?;
        let stage = self
            .emb
            .gather_staged(self.gpu, emb_table.as_slice(), ids, &raw)?;
        self.emb.release(self.gpu, &stage);
        let (g, ..) = self.weight("token_embd_norm.weight")?;
        let mut cur = self.buf(tn)?;
        self.norm.run(&raw, Some(&g), &cur, t)?;
        drop((raw, g));
        self.tap(&mut tap, "inp_norm", -1, &cur, tn)?;

        let pos: Vec<i32> = (0..t as i32).collect();
        let pos = self.bytes(bytemuck_i32(&pos))?;
        let state_stride = 2 * cfg.kv_dim() + 2 * n;
        let zeros = vec![0f32; (cfg.n_conv - 1) * state_stride];
        let mut states = Vec::with_capacity(plan.n_slots());
        for _ in 0..plan.n_slots() {
            states.push(self.f32s(&zeros)?);
        }

        // block stack (oldest first) and the previous pass's bank
        let max_blocks = DepthMix::MAX_STATES - 1;
        let stack = self.buf(max_blocks * tn)?;
        let mut nblocks = 0usize;
        let bank = self.buf(max_blocks * tn)?;
        let mut bank_count = 0usize;
        let mut bank_out: Option<Owned<'g>> = None;
        let mut prelude: Option<(Owned<'g>, usize)> = None;

        for st in plan.steps() {
            let in_core = (plan.start..plan.end).contains(&st.layer);
            if in_core && st.layer == plan.start {
                if st.pass == 0 {
                    prelude = Some((self.dup(&cur, tn)?, nblocks));
                } else {
                    let (p, pblocks) = prelude.as_ref().expect("prelude saved at pass 0");
                    bank_count = nblocks - pblocks - 1;
                    for i in 0..bank_count {
                        gemv::copy_at(
                            self.gpu,
                            &bank,
                            (i * tn * 4) as u64,
                            &stack,
                            ((pblocks + 1 + i) * tn * 4) as u64,
                            (tn * 4) as u64,
                        )?;
                    }
                    bank_out = Some(std::mem::replace(&mut cur, self.dup(p, tn)?));
                    nblocks = *pblocks;
                }
            }
            let msgs = if in_core && st.pass > 0 {
                let (m, _) = self.depth_mix(
                    &bank,
                    0,
                    bank_count,
                    bank_out.as_ref().expect("bank"),
                    &format!("blk.{}.loop_msg_query.weight", st.layer),
                    t,
                )?;
                self.tap(&mut tap, "loop_msgs", st.slot as i32, &m, CTRL.len() * tn)?;
                Some(m)
            } else {
                None
            };
            cur = self.layer(
                st.layer,
                st.slot,
                cur,
                &stack,
                &mut nblocks,
                msgs.as_deref(),
                &states[st.slot],
                &pos,
                t,
                &mut tap,
            )?;
        }

        // output head on the last token
        let (mixed, _) = self.depth_mix(&stack, 0, nblocks, &cur, "output_res_query.weight", t)?;
        self.tap(&mut tap, "out_res_mix", -1, &mixed, tn)?;
        let (g, ..) = self.weight("output_norm.weight")?;
        let last = self.buf(n)?;
        self.norm
            .run_view(&mixed, (n, (t - 1) * n), Some(&g), &last, (n, 0), 1)?;
        self.tap(&mut tap, "result_norm", -1, &last, n)?;
        let (logits, rows) = self.proj("output.weight", &last, 1)?;
        if cfg.softcap != 0.0 {
            self.act.run(&logits, &logits, rows, cfg.softcap)?;
        }
        self.tap(&mut tap, "result_output", -1, &logits, cfg.vocab)?;
        ops::download_f32(self.gpu, &logits, cfg.vocab)
    }

    /// llama.cpp `build_layer(il = slot, ip = layer, ...)`.
    #[allow(clippy::too_many_arguments)]
    fn layer(
        &mut self,
        layer: usize,
        slot: usize,
        inp_sa: Owned<'g>,
        stack: &Buf,
        nblocks: &mut usize,
        msgs: Option<&Buf>,
        state: &Buf,
        pos: &Buf,
        t: usize,
        tap: &mut Option<&mut Tap>,
    ) -> Result<Owned<'g>> {
        let cfg = self.cfg.clone();
        let (n, hd, h, hkv) = (cfg.n_embd, cfg.head_dim, cfg.n_head, cfg.n_head_kv);
        let (kv, qd) = (cfg.kv_dim(), cfg.q_dim());
        let tn = t * n;
        let s = slot as i32;
        let b = |x: &str| format!("blk.{layer}.{x}.weight");
        let swa = cfg.swa[layer];
        let state_stride = 2 * kv + 2 * n;

        // attention inputs: views (buffer, stride, offset) of cur or the depth mix
        let mixed = if self.has(&b("attn_res_query")) {
            let (m, nq) = self.depth_mix(stack, 0, *nblocks, &inp_sa, &b("attn_res_query"), t)?;
            self.tap(tap, "attn_res_mix", s, &m, nq * tn)?;
            Some((m, nq))
        } else {
            None
        };
        let view = |j: usize| -> (&Buf, (usize, usize)) {
            match &mixed {
                Some((m, nq)) => (m, (nq * n, if *nq == 1 { 0 } else { j * n })),
                None => (&inp_sa, (n, 0)),
            }
        };
        let mut ctrl_out: Vec<Owned<'g>> = Vec::new();
        if let Some(msgs) = msgs {
            for j in 0..3 {
                let (buf, cv) = view(j);
                let o = self.control(layer, j, buf, cv, msgs, t)?;
                self.tap(
                    tap,
                    ["loop_ctrl_q", "loop_ctrl_k", "loop_ctrl_v"][j],
                    s,
                    &o,
                    tn,
                )?;
                ctrl_out.push(o);
            }
        }
        let input = |j: usize| -> (&Buf, (usize, usize)) {
            if ctrl_out.is_empty() {
                view(j)
            } else {
                (&ctrl_out[j], (n, 0))
            }
        };
        // input norms: SWA shares attn_norm (one input unless messages split it)
        let norm_names: [String; 3] = if swa {
            [b("attn_norm"), b("attn_norm"), b("attn_norm")]
        } else {
            [
                b("attn_q_inp_norm"),
                b("attn_k_inp_norm"),
                b("attn_v_inp_norm"),
            ]
        };
        let shared = swa && msgs.is_none();
        let mut ins: Vec<Owned<'g>> = Vec::new();
        for (j, name) in norm_names
            .iter()
            .enumerate()
            .take(if shared { 1 } else { 3 })
        {
            let (g, ..) = self.weight(name)?;
            let (x, xv) = input(j);
            let o = self.buf(tn)?;
            self.norm.run_view(x, xv, Some(&g), &o, (n, 0), t)?;
            ins.push(o);
        }
        drop(ctrl_out);
        drop(mixed);
        let in_at = |j: usize| if shared { 0 } else { j };
        self.tap(tap, "attn_in_q", s, &ins[0], tn)?;
        if !shared {
            self.tap(tap, "attn_in_k", s, &ins[1], tn)?;
            self.tap(tap, "attn_in_v", s, &ins[2], tn)?;
        }
        let (in_q, in_k, in_v) = (&ins[in_at(0)], &ins[in_at(1)], &ins[in_at(2)]);

        // projections + short convolutions (dynamic taps on full layers)
        let (q, _) = self.proj(&b("attn_q"), in_q, t)?;
        let (kp, _) = self.proj(&b("attn_k"), in_k, t)?;
        let (vp, _) = self.proj(&b("attn_v"), in_v, t)?;
        let dynk = if swa {
            None
        } else {
            Some(self.proj(&b("attn_k_conv_proj"), in_k, t)?)
        };
        let dynv = if swa {
            None
        } else {
            Some(self.proj(&b("attn_v_conv_proj"), in_v, t)?)
        };
        let (kw, ..) = self.weight(&b("attn_k_conv"))?;
        let (vw, ..) = self.weight(&b("attn_v_conv"))?;
        let k = cfg.n_conv;
        self.tap(tap, "Kproj", s, &kp, t * kv)?;
        self.tap(tap, "Vproj", s, &vp, t * kv)?;
        let hk = dynk.as_ref().map(|(d, rows)| (&**d, rows / k));
        let hv = dynv.as_ref().map(|(d, rows)| (&**d, rows / k));
        self.conv
            .run(state, state_stride, 0, &kp, &kw, k, hk, &kp, kv, t)?;
        self.conv
            .run(state, state_stride, kv, &vp, &vw, k, hv, &vp, kv, t)?;
        drop((dynk, dynv, kw, vw));

        // clips: Q/R by channel scale (+ tau on full layers), K plain, V per-head RMS
        let (qc, ..) = self.weight(&b("attn_q_clip"))?;
        let qcur = self.buf(t * qd)?;
        if swa {
            self.clip_q.run(&q, Some(&qc), None, &qcur, t)?;
        } else {
            self.clip_q_tau.run(&q, Some(&qc), Some(pos), &qcur, t)?;
        }
        let kcur = self.buf(t * kv)?;
        self.clip_k.run(&kp, None, None, &kcur, t)?;
        let vcur = self.buf(t * kv)?;
        self.head_norm.run(&vp, None, &vcur, t * hkv)?;
        let (r, _) = self.proj(&b("attn_rel"), in_q, t)?;
        let (rc, ..) = self.weight(&b("attn_rel_clip"))?;
        let rcur = self.buf(t * cfg.n_rel * h)?;
        if swa {
            self.clip_r.run(&r, Some(&rc), None, &rcur, t)?;
        } else {
            self.clip_r_tau.run(&r, Some(&rc), Some(pos), &rcur, t)?;
        }
        drop((q, kp, vp, r, qc, rc));
        self.tap(tap, "Qcur", s, &qcur, t * qd)?;
        self.tap(tap, "Kcur", s, &kcur, t * kv)?;
        self.tap(tap, "Vcur", s, &vcur, t * kv)?;
        self.tap(tap, "Rcur", s, &rcur, t * cfg.n_rel * h)?;

        // attention: rel bias + per-token sink logit
        let (sinks, _) = self.proj(&b("attn_sink_proj"), in_q, t)?;
        self.tap(tap, "attn_sink", s, &sinks, t * h)?;
        let (bias, bdims, _) = self.weight(&b("attn_rel_bias"))?;
        let kqv = self.buf(t * qd)?;
        self.attn.run(
            &AttnShape {
                heads: h,
                kv_heads: hkv,
                head_dim: hd,
                window: if swa { cfg.sliding_window } else { 0 },
                scale: 1.0 / (hd as f32).sqrt(),
            },
            &qcur,
            pos,
            t,
            &kcur,
            &vcur,
            pos,
            t,
            Some(&RelBias {
                r: &rcur,
                bias: &bias,
                rel_dim: cfg.n_rel,
                extent: bdims[1],
            }),
            Some(&Sink {
                sink: &sinks,
                per_token: true,
                scale: 1.0,
            }),
            &kqv,
        )?;
        drop((qcur, kcur, vcur, rcur, sinks, bias));
        self.tap(tap, "kqv_out", s, &kqv, t * qd)?;

        let (gate, _) = self.proj(&b("attn_gate"), in_q, t)?;
        self.add.run_gated(&kqv, &gate, &kqv, t * qd)?;
        drop(gate);
        drop(ins);
        self.tap(tap, "attn_gated", s, &kqv, t * qd)?;
        let (wo, _) = self.proj(&b("attn_output"), &kqv, t)?;
        drop(kqv);
        self.tap(tap, "attn_wo", s, &wo, tn)?;
        let (pw, ..) = self.weight(&b("attn_post_conv"))?;
        self.conv
            .run(state, state_stride, 2 * kv, &wo, &pw, k, None, &wo, n, t)?;
        drop(pw);
        self.tap(tap, "attn_out", s, &wo, tn)?;

        // a block start pushes the old stream; otherwise the residual add
        if layer.is_multiple_of(cfg.attn_res_block) {
            ensure!(*nblocks < DepthMix::MAX_STATES - 1, "block stack full");
            self.copy(stack, *nblocks * tn, &inp_sa, tn)?;
            *nblocks += 1;
        } else {
            self.add.run(&inp_sa, &wo, &wo, tn, 1.0)?;
        }
        let ffn_inp = wo;
        drop(inp_sa);
        self.tap(tap, "ffn_inp", s, &ffn_inp, tn)?;

        // MoE
        let (mut x, _) = self.depth_mix(stack, 0, *nblocks, &ffn_inp, &b("ffn_res_query"), t)?;
        self.tap(tap, "ffn_res_mix", s, &x, tn)?;
        if let Some(msgs) = msgs {
            x = self.control(layer, 3, &x, (n, 0), msgs, t)?;
            self.tap(tap, "loop_ctrl_ffn", s, &x, tn)?;
        }
        let (g, ..) = self.weight(&b("ffn_norm"))?;
        let fin = self.buf(tn)?;
        self.norm.run(&x, Some(&g), &fin, t)?;
        drop((x, g));
        self.tap(tap, "ffn_in", s, &fin, tn)?;
        let moe = self.moe(layer, &fin, t, slot, tap)?;
        drop(fin);
        self.tap(tap, "ffn_moe_out", s, &moe, tn)?;
        let (fw, ..) = self.weight(&b("ffn_post_conv"))?;
        self.conv.run(
            state,
            state_stride,
            2 * kv + n,
            &moe,
            &fw,
            k,
            None,
            &moe,
            n,
            t,
        )?;
        drop(fw);
        self.tap(tap, "ffn_out", s, &moe, tn)?;
        self.add.run(&ffn_inp, &moe, &ffn_inp, tn, 1.0)?;
        self.tap(tap, "l_out", s, &ffn_inp, tn)?;
        Ok(ffn_inp)
    }

    /// Sigmoid-routed SITU experts with the merged gate_up tensor.
    fn moe(
        &mut self,
        layer: usize,
        x: &Buf,
        t: usize,
        slot: usize,
        tap: &mut Option<&mut Tap>,
    ) -> Result<Owned<'g>> {
        let (ne, kx, n) = (self.cfg.n_expert, self.cfg.n_expert_used, self.cfg.n_embd);
        let b = |s: &str| format!("blk.{layer}.{s}");
        let (logits, _) = self.proj(&b("ffn_gate_inp.weight"), x, t)?;
        self.tap(tap, "ffn_moe_logits", slot as i32, &logits, t * ne)?;
        let (bias, ..) = self.weight(&b("exp_probs_b.bias"))?;
        let ids = Owned(self.gpu, gemv::alloc(self.gpu, self.router.ids_bytes(t))?);
        let wts = Owned(
            self.gpu,
            gemv::alloc(self.gpu, self.router.weights_bytes(t))?,
        );
        self.router.run_biased(&logits, &bias, &ids, &wts, t)?;
        drop((logits, bias));
        let pairs = t * kx;
        let sel = ops::download_u32(self.gpu, &ids, pairs)?;
        ensure!(
            sel.iter().all(|&e| (e as usize) < ne),
            "router id out of range"
        );

        // gate_up: pair p = t*k + j reads activation row t
        let (gu_w, gu_dims, gu_ty) = self.weight(&b("ffn_gate_up_exps.weight"))?;
        let (gu_rows, cols) = (gu_dims[1], gu_dims[0]);
        let gpu = self.gpu;
        let gu = self.buf(pairs * gu_rows)?;
        {
            let g = self.gemv(gu_ty, gu_rows, cols)?;
            let words: Vec<u32> = sel.iter().map(|&e| e * g.blob_words()).collect();
            let mut map = g.map_for_blobs(&words, XLayout::PerExpert);
            let (rows, stride) = (g.rows(pairs), (g.padded_cols() / 4) as u32);
            for p in 0..pairs {
                for r in 0..gu_rows {
                    map[rows + p * gu_rows + r] = stride * (p / kx) as u32;
                }
            }
            let map = Owned(
                gpu,
                gemv::upload(gpu, &breccia_kernels::batch::map_to_bytes(&map))?,
            );
            g.run(&gu_w, x, &gu, &map, pairs)?;
        }
        drop(gu_w);
        let half = gu_rows / 2;
        let situ = self.buf(pairs * half)?;
        self.glu.run_fused(&gu, &situ, pairs, half)?;
        drop(gu);

        // down: pair p reads SITU row p
        let (dn_w, dn_dims, dn_ty) = self.weight(&b("ffn_down_exps.weight"))?;
        ensure!(
            dn_dims[0] == half && dn_dims[1] == n,
            "ffn_down_exps shape {dn_dims:?}"
        );
        let down = self.buf(pairs * n)?;
        {
            let g = self.gemv(dn_ty, n, half)?;
            let words: Vec<u32> = sel.iter().map(|&e| e * g.blob_words()).collect();
            let map = g.map_for_blobs(&words, XLayout::PerExpert);
            let map = Owned(
                gpu,
                gemv::upload(gpu, &breccia_kernels::batch::map_to_bytes(&map))?,
            );
            g.run(&dn_w, &situ, &down, &map, pairs)?;
        }
        drop((dn_w, situ));

        let slots = self.u32s(&(0..pairs as u32).collect::<Vec<_>>())?;
        let shared = self.f32s(&vec![0.0; t * n])?;
        let gate0 = self.f32s(&vec![0.0; t])?;
        let out = self.buf(t * n)?;
        self.fanin
            .run(&down, &slots, &wts, &shared, &gate0, &out, t, kx, n)?;
        Ok(out)
    }
}

fn bytemuck_f32(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding or invalid bit patterns as bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}

fn bytemuck_i32(v: &[i32]) -> &[u8] {
    // SAFETY: as above.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}
