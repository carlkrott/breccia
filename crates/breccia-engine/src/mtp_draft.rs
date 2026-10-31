//! MTP draft layer: one transformer layer (QSA attention + HC + 512-expert MoE)
//! plus a head restricted to the draft vocabulary, run on the same GPU as the
//! trunk. Numerics only steer *acceptance*; the trunk's verify decides output.
//!
//! Front (per row): `R' = fc_hidden(norm(R)) + broadcast(fc_embedding(norm(embed(tok))))`,
//! then attn-HC read -> k/v -> draft KV append. Rest (last row only): q, attend,
//! o_proj, attn-HC write, mlp-HC read, router + shared + routed experts, mlp-HC
//! write, final mixer, head. Layout and order follow `strata/src/core/mtp.cpp`.
//!
//! Experts live on the device (708 MB, AoS Q2_0) so a whole step is one dispatch
//! batch with no mid-step host sync.
//!
//! shortcut: buffers are never freed (the draft lives as long as the process);
//! upgrade if a long-lived server needs to unload it.

use anyhow::{anyhow, bail, Context, Result};
use breccia_format::MtpRt;
use breccia_kernels::batch::alloc_map;
use breccia_kernels::gemv::{alloc, download, upload};
use breccia_kernels::ops::{f32_from_bytes, f32_to_bytes, Add, EmbGather, HcMixer, HcWeights, HcWorkspace, RmsNorm, Router, SwiGlu};
use breccia_kernels::proj::batch_map_buf;
use breccia_kernels::{
    batch_cmd, Gemv, GemvBatch, MoeCombine, MoeMap, Qsa, QsaState, QsaWorkspace, QuantType,
};
use breccia_vk::ctx::{Buf, Gpu};

use crate::model::Model;

const N_EMBD: usize = 2560;
const HC: usize = 4;
const HC_DIM: usize = N_EMBD * HC;
const HC_LR: usize = 320;
const N_EXPERT: usize = 512;
const K_USED: usize = 10;
const N_FF: usize = 640;
/// Bytes of one Q2_0 expert projection: 640 rows x 40 blocks x 18 B.
const PROJ_BYTES: usize = 460_800;

/// fp32 -> fp16 bits, round to nearest even, subnormals kept.
fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 0xff {
        return s | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let ne = e - 127 + 15;
    if ne >= 31 {
        return s | 0x7c00;
    }
    if ne <= 0 {
        if ne < -10 {
            return s;
        }
        let m = m | 0x80_0000;
        let sh = (14 - ne) as u32;
        let half = m >> sh;
        let rem = m & ((1 << sh) - 1);
        let mid = 1u32 << (sh - 1);
        let up = rem > mid || (rem == mid && half & 1 == 1);
        return s | (half + up as u32) as u16;
    }
    let half = ((ne as u32) << 10) | (m >> 13);
    let rem = m & 0x1fff;
    let up = rem > 0x1000 || (rem == 0x1000 && half & 1 == 1);
    s | (half + up as u32) as u16
}

fn f16_to_f32(h: u16) -> f32 {
    breccia_kernels::reference::f16_to_f32(h)
}

/// Q8_0 (34 B blocks: fp16 d + 32 x i8) -> fp16 row-major bytes.
/// shortcut: the engine has no Q8_0 GEMV, so the dense weights are widened to
/// fp16 on the host (2x VRAM, ~134 MB); upgrade if the draft's VRAM matters.
fn q8_0_to_f16(src: &[u8]) -> Vec<u8> {
    assert!(src.len() % 34 == 0, "q8_0 data is whole 34-byte blocks");
    let mut out = Vec::with_capacity(src.len() / 34 * 64);
    for blk in src.chunks_exact(34) {
        let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for &q in &blk[2..] {
            out.extend_from_slice(&f32_to_f16(d * (q as i8) as f32).to_le_bytes());
        }
    }
    out
}

/// One projection of every expert, repacked from the plane-split blob to AoS
/// Q2_0 blocks (`scale(2) + codes(16)`), experts back to back.
/// `role`: 0 gate (gu row 2r), 1 up (gu row 2r+1), 2 down.
fn repack_role(rt: &MtpRt, role: usize) -> Result<Vec<u8>> {
    const GU_CODES: usize = 0;
    const DOWN_CODES: usize = 819_200;
    const GU_SCALES: usize = 1_228_800;
    const DOWN_SCALES: usize = 1_331_200;
    let (rows, blocks) = if role == 2 { (2560, 10) } else { (640, 40) };
    let mut out = Vec::with_capacity(N_EXPERT * PROJ_BYTES);
    for e in 0..N_EXPERT {
        let blob = rt.expert_blob(e).map_err(|x| anyhow!("{x}"))?;
        for r in 0..rows {
            for b in 0..blocks {
                let (codes, scale) = if role == 2 {
                    let i = r * blocks + b;
                    (DOWN_CODES + i * 16, DOWN_SCALES + i * 2)
                } else {
                    let i = (2 * r + role) * blocks + b;
                    (GU_CODES + i * 16, GU_SCALES + i * 2)
                };
                out.extend_from_slice(&blob[scale..scale + 2]);
                out.extend_from_slice(&blob[codes..codes + 16]);
            }
        }
    }
    debug_assert_eq!(out.len(), N_EXPERT * PROJ_BYTES);
    Ok(out)
}

/// A projection weight: uploaded buffer plus its GEMV.
struct P<'g> {
    w: Buf,
    g: Gemv<'g>,
}

pub struct MtpDraft<'g> {
    gpu: &'g Gpu,
    model: &'g Model,
    emb: EmbGather<'g>,
    rms_e: RmsNorm<'g>,
    rms_h: RmsNorm<'g>,
    add: Add<'g>,
    fc: GemvBatch<'g>,
    fc_map: Buf,
    w_fc_e: Buf,
    w_fc_h: Buf,
    g_en: Buf,
    g_hn: Buf,
    hc: HcMixer<'g>,
    hc_ws: HcWorkspace,
    fin: HcMixer<'g>,
    fin_ws: HcWorkspace,
    attn_hc: [Buf; 4], // norm, down, up, inject
    mlp_hc: [Buf; 4],
    fin_hc: [Buf; 3],
    qsa: Qsa<'g>,
    qsa_state: QsaState,
    qsa_ws: QsaWorkspace,
    q_norm: Buf,
    k_norm: Buf,
    pq: P<'g>,
    pk: P<'g>,
    pv: P<'g>,
    po: P<'g>,
    prouter: P<'g>,
    psh_g: P<'g>,
    psh_u: P<'g>,
    psh_d: P<'g>,
    pshgate: P<'g>,
    router: Router<'g>,
    swiglu: SwiGlu<'g>,
    moe_map: MoeMap<'g>,
    moe_combine: MoeCombine<'g>,
    gate_b: GemvBatch<'g>,
    up_b: GemvBatch<'g>,
    down_b: GemvBatch<'g>,
    gate_all: Buf,
    up_all: Buf,
    down_all: Buf,
    map_gate: Buf,
    map_up: Buf,
    map_down: Buf,
    head: P<'g>,
    n_draft: usize,
    rt_tokens: Vec<i32>,
    // activations
    emb4: Buf,
    en4: Buf,
    e2x4: Buf,
    hn: Buf,
    h2: Buf,
    r_in: Buf,
    r_a: Buf,
    r_b: Buf,
    r_c: Buf,
    block_out: Buf,
    router_logits: Buf,
    router_ids: Buf,
    router_w: Buf,
    sh_g: Buf,
    sh_u: Buf,
    sh_act: Buf,
    sh: Buf,
    sh_gate_logit: Buf,
    moe_gate: Buf,
    moe_up: Buf,
    moe_act: Buf,
    moe_down: Buf,
    logits: Buf,
    /// Draft KV cells in use (the next append goes to cell `len`).
    pub len: usize,
    pub cells: usize,
    /// Device bytes held by this draft (for the VRAM report).
    pub bytes: u64,
}

fn fbuf(gpu: &Gpu, floats: usize) -> Result<Buf> {
    // 256-float round-up covers every kernel's padded read of its input.
    alloc(gpu, (floats.div_ceil(256) * 256 * 4) as u64)
}

impl<'g> MtpDraft<'g> {
    /// Open the MTP weights in `dir` and build the draft (what the CLI calls).
    pub fn open(model: &'g Model, dev: usize, dir: &str, cells: usize) -> Result<Self> {
        let rt = MtpRt::open(dir).map_err(|e| anyhow!("{dir}: {e}"))?;
        Self::new(model, dev, &rt, cells).context("building the MTP draft")
    }

    /// Build the draft on trunk device `dev`. `cells` is the draft KV capacity.
    pub fn new(model: &'g Model, dev: usize, rt: &MtpRt, cells: usize) -> Result<Self> {
        let gpu: &'g Gpu = model.devices[dev].gpu();
        let mut bytes = 0u64;
        let mut up = |b: &[u8]| -> Result<Buf> {
            bytes += b.len() as u64;
            upload(gpu, b)
        };
        let dense = |n: &str| -> Result<&[u8]> {
            rt.dense_bytes(n).map_err(|e| anyhow!("{n}: {e}"))
        };
        let f16w = |n: &str| -> Result<Vec<u8>> { Ok(q8_0_to_f16(dense(n)?)) };

        let emb_t = model
            .shards
            .tensor("token_embd.weight")
            .ok_or_else(|| anyhow!("no token_embd.weight"))?;
        let emb_ty = QuantType::from_ggml(emb_t.ty()).ok_or_else(|| anyhow!("token_embd type"))?;
        let emb = EmbGather::new(gpu, emb_ty, N_EMBD)?;

        let fc = GemvBatch::new(gpu, QuantType::F16, N_EMBD, N_EMBD)?;
        if fc.padded_cols() != N_EMBD {
            bail!("fc batch pads {} cols, the 4-row trick needs {N_EMBD}", fc.padded_cols());
        }
        let fc_map = batch_map_buf(gpu, &fc, HC)?;
        let w_fc_e = up(&f16w("fc_embedding.weight")?)?;
        let w_fc_h = up(&f16w("fc_hidden.weight")?)?;
        let g_en = up(dense("pre_fc_norm_embedding.weight")?)?;
        let g_hn = up(dense("pre_fc_norm_hidden.weight")?)?;

        let hc = HcMixer::new(gpu, QuantType::Bf16, N_EMBD, HC, HC_LR, true)?;
        let fin = HcMixer::new(gpu, QuantType::Bf16, N_EMBD, HC, HC_LR, false)?;
        let hc_ws = hc.workspace(gpu, 1)?;
        let fin_ws = fin.workspace(gpu, 1)?;
        let mut hc_set = |p: &str, inject: bool| -> Result<[Buf; 4]> {
            let inj = if inject {
                up(dense(&format!("{p}.block_inject_weight.weight"))?)?
            } else {
                up(&[0u8; 8])?
            };
            Ok([
                up(dense(&format!("{p}.hc_norm.weight"))?)?,
                up(dense(&format!("{p}.input_mix_weight_down.weight"))?)?,
                up(dense(&format!("{p}.input_mix_weight_up.weight"))?)?,
                inj,
            ])
        };
        let attn_hc = hc_set("attn_hyper_connection", true)?;
        let mlp_hc = hc_set("mlp_hyper_connection", true)?;
        let fin4 = hc_set("hyper_connection_mixer", false)?;
        let [f0, f1, f2, _] = fin4;
        let fin_hc = [f0, f1, f2];

        let max_pos = cells.max(8);
        let qsa = Qsa::new(gpu, max_pos)?;
        let qsa_state = QsaState::new(gpu, cells, max_pos)?;
        qsa_state.reset(gpu)?;
        let qsa_ws = qsa.workspace(gpu, 1, cells)?;
        let q_norm = up(dense("self_attn.q_norm.weight")?)?;
        let k_norm = up(dense("self_attn.k_norm.weight")?)?;

        let mut proj = |name: &str, ty: QuantType, rows: usize, cols: usize, q8: bool| -> Result<P<'g>> {
            let w = if q8 { up(&f16w(name)?)? } else { up(dense(name)?)? };
            Ok(P { w, g: Gemv::new(gpu, ty, rows, cols)? })
        };
        let f16 = QuantType::F16;
        let pq = proj("self_attn.q_proj.weight", f16, 12288, N_EMBD, true)?;
        let pk = proj("self_attn.k_proj.weight", f16, 512, N_EMBD, true)?;
        let pv = proj("self_attn.v_proj.weight", f16, 512, N_EMBD, true)?;
        let po = proj("self_attn.o_proj.weight", f16, N_EMBD, 6144, true)?;
        let prouter = proj("mlp.gate.weight", QuantType::Bf16, N_EXPERT, N_EMBD, false)?;
        let psh_g = proj("mlp.shared_expert.gate_proj.weight", f16, N_FF, N_EMBD, true)?;
        let psh_u = proj("mlp.shared_expert.up_proj.weight", f16, N_FF, N_EMBD, true)?;
        let psh_d = proj("mlp.shared_expert.down_proj.weight", f16, N_EMBD, N_FF, true)?;
        let pshgate = proj("mlp.shared_expert_gate.weight", QuantType::Bf16, 1, N_EMBD, false)?;

        let router = Router::new(gpu, N_EXPERT, K_USED)?;
        let swiglu = SwiGlu::new(gpu)?;
        let moe_map = MoeMap::new(gpu)?;
        let moe_combine = MoeCombine::new(gpu)?;
        let q2 = QuantType::Q2_0;
        let gate_b = GemvBatch::new(gpu, q2, N_FF, N_EMBD)?;
        let up_b = GemvBatch::new(gpu, q2, N_FF, N_EMBD)?;
        let down_b = GemvBatch::new(gpu, q2, N_EMBD, N_FF)?;
        for b in [&gate_b, &up_b, &down_b] {
            if b.blob_bytes() as usize != PROJ_BYTES {
                bail!("expert projection is {} B, repack assumes {PROJ_BYTES}", b.blob_bytes());
            }
        }
        // One role at a time: each repack is a 236 MB host vector.
        let gate_all = up(&repack_role(rt, 0)?)?;
        let up_all = up(&repack_role(rt, 1)?)?;
        let down_all = up(&repack_role(rt, 2)?)?;
        let map_gate = alloc_map(gpu, gate_b.rows(K_USED))?;
        let map_up = alloc_map(gpu, up_b.rows(K_USED))?;
        let map_down = alloc_map(gpu, down_b.rows(K_USED))?;

        // Head restricted to the draft vocabulary: gather those rows of
        // output.weight once, contiguous.
        let out_t = model
            .shards
            .tensor("output.weight")
            .ok_or_else(|| anyhow!("no output.weight"))?;
        let head_ty = QuantType::from_ggml(out_t.ty()).ok_or_else(|| anyhow!("output.weight type"))?;
        let row_bytes = N_EMBD / head_ty.block_elems() * head_ty.block_bytes();
        let all = out_t.as_slice();
        let rt_tokens: Vec<i32> = rt.draft_tokens().collect();
        let n_draft = rt_tokens.len();
        let mut packed = Vec::with_capacity(n_draft * row_bytes);
        for &t in &rt_tokens {
            let o = t as usize * row_bytes;
            packed.extend_from_slice(
                all.get(o..o + row_bytes)
                    .ok_or_else(|| anyhow!("draft token {t} is outside output.weight"))?,
            );
        }
        let head = P { w: up(&packed)?, g: Gemv::new(gpu, head_ty, n_draft, N_EMBD)? };
        drop(packed);

        let mut act_bytes = 0u64;
        let mut f = |n: usize| -> Result<Buf> {
            let b = fbuf(gpu, n)?;
            act_bytes += b.size;
            Ok(b)
        };
        let me = MtpDraft {
            gpu,
            model,
            emb4: alloc(gpu, emb.out_bytes(HC))?,
            emb,
            rms_e: RmsNorm::new(gpu, N_EMBD, 1)?,
            rms_h: RmsNorm::new(gpu, HC_DIM, 1)?,
            add: Add::new(gpu)?,
            fc,
            fc_map,
            w_fc_e,
            w_fc_h,
            g_en,
            g_hn,
            hc,
            hc_ws,
            fin,
            fin_ws,
            attn_hc,
            mlp_hc,
            fin_hc,
            qsa,
            qsa_state,
            qsa_ws,
            q_norm,
            k_norm,
            pq,
            pk,
            pv,
            po,
            prouter,
            psh_g,
            psh_u,
            psh_d,
            pshgate,
            router,
            swiglu,
            moe_map,
            moe_combine,
            gate_b,
            up_b,
            down_b,
            gate_all,
            up_all,
            down_all,
            map_gate,
            map_up,
            map_down,
            head,
            n_draft,
            rt_tokens,
            en4: f(HC_DIM)?,
            e2x4: f(HC_DIM)?,
            hn: f(HC_DIM)?,
            h2: f(HC_DIM)?,
            r_in: f(HC_DIM)?,
            r_a: f(HC_DIM)?,
            r_b: f(HC_DIM)?,
            r_c: f(HC_DIM)?,
            block_out: f(N_EMBD)?,
            router_logits: f(N_EXPERT)?,
            router_ids: f(K_USED)?,
            router_w: f(K_USED)?,
            sh_g: f(N_FF)?,
            sh_u: f(N_FF)?,
            sh_act: f(N_FF)?,
            sh: f(N_EMBD)?,
            sh_gate_logit: f(1)?,
            moe_gate: f(K_USED * N_FF)?,
            moe_up: f(K_USED * N_FF)?,
            moe_act: f(K_USED * N_FF)?,
            moe_down: f(K_USED * N_EMBD)?,
            logits: f(n_draft)?,
            len: 0,
            cells,
            bytes: 0,
        };
        let mut me = me;
        me.bytes = bytes + act_bytes + me.qsa_state.bytes() + me.hc_ws.bytes() + me.fin_ws.bytes();
        Ok(me)
    }

    pub fn n_draft(&self) -> usize {
        self.n_draft
    }

    /// Set the trunk hidden stack `R` (10240 floats) the next step conditions on.
    pub fn set_r(&self, r: &[f32]) -> Result<()> {
        if r.len() != HC_DIM {
            bail!("R must be {HC_DIM} floats, got {}", r.len());
        }
        breccia_kernels::gemv::upload_into(self.gpu, &self.r_in, &f32_to_bytes(r))
    }

    /// Record the front for `tok` conditioned on `r_in` into `self.r_a`, append
    /// the draft K/V at cell `pos`. Must run inside an open batch.
    fn record_front(&self, pos: usize) -> Result<()> {
        self.rms_e.run(&self.emb4, Some(&self.g_en), &self.en4, HC)?;
        self.fc.run(&self.w_fc_e, &self.en4, &self.e2x4, &self.fc_map, HC)?;
        self.rms_h.run(&self.r_in, Some(&self.g_hn), &self.hn, 1)?;
        self.fc.run(&self.w_fc_h, &self.hn, &self.h2, &self.fc_map, HC)?;
        self.add.run(&self.h2, &self.e2x4, &self.r_a, HC_DIM, 1.0)?;
        self.hc.read(
            &self.hc_ws,
            &self.r_a,
            &HcWeights {
                norm: &self.attn_hc[0],
                down: &self.attn_hc[1],
                up: &self.attn_hc[2],
                inject: Some(&self.attn_hc[3]),
            },
        )?;
        let mixed = &self.hc_ws.mixed;
        let ws = &self.qsa_ws;
        self.pk.g.run(&self.pk.w, mixed, &ws.k_cur)?;
        self.pv.g.run(&self.pv.w, mixed, &ws.v_cur)?;
        self.qsa.k_norm(ws, &self.k_norm)?;
        self.qsa.rope_k(ws, &self.qsa_state, pos)?;
        self.qsa.kv_append(&self.qsa_state, ws, pos)?;
        Ok(())
    }

    /// The rest of the layer for the row at `pos`, ending in draft logits.
    /// Reads `self.r_a` (the front's output), leaves the post-layer stack in `r_c`.
    fn record_rest(&mut self, pos: usize) -> Result<()> {
        let n_kv_after = pos + 1;
        {
            let ws = &mut self.qsa_ws;
            ws.n_blocks = n_kv_after / 4;
            ws.max_ids = ws.n_blocks.min(512) + 1;
            ws.slots = 4 * ws.max_ids;
        }
        let ws = &self.qsa_ws;
        let mixed = &self.hc_ws.mixed;
        self.pq.g.run(&self.pq.w, mixed, &ws.q_full)?;
        self.qsa.q_norm(ws, &self.q_norm)?;
        self.qsa.rope_q(ws, &self.qsa_state, pos)?;
        self.qsa.topk_dense(ws, pos)?;
        self.qsa.attend(&self.qsa_state, ws, pos)?;
        self.qsa.apply(&self.qsa_state, ws, pos)?;
        self.po.g.run(&self.po.w, &ws.attn_out, &self.block_out)?;
        self.hc.write(&self.hc_ws, &self.r_a, &self.block_out, &self.r_b)?;

        self.hc.read(
            &self.hc_ws,
            &self.r_b,
            &HcWeights {
                norm: &self.mlp_hc[0],
                down: &self.mlp_hc[1],
                up: &self.mlp_hc[2],
                inject: Some(&self.mlp_hc[3]),
            },
        )?;
        let mixed = &self.hc_ws.mixed;
        self.prouter.g.run(&self.prouter.w, mixed, &self.router_logits)?;
        self.router.run(&self.router_logits, &self.router_ids, &self.router_w, 1)?;
        self.psh_g.g.run(&self.psh_g.w, mixed, &self.sh_g)?;
        self.psh_u.g.run(&self.psh_u.w, mixed, &self.sh_u)?;
        self.swiglu.run(&self.sh_g, &self.sh_u, &self.sh_act, N_FF)?;
        self.psh_d.g.run(&self.psh_d.w, &self.sh_act, &self.sh)?;
        self.pshgate.g.run(&self.pshgate.w, mixed, &self.sh_gate_logit)?;
        // routed experts, all on device
        self.moe_map.run(&self.router_ids, &self.map_gate, &self.gate_b, K_USED, 0, false)?;
        self.gate_b.run(&self.gate_all, mixed, &self.moe_gate, &self.map_gate, K_USED)?;
        self.moe_map.run(&self.router_ids, &self.map_up, &self.up_b, K_USED, 0, false)?;
        self.up_b.run(&self.up_all, mixed, &self.moe_up, &self.map_up, K_USED)?;
        self.swiglu.run(&self.moe_gate, &self.moe_up, &self.moe_act, K_USED * N_FF)?;
        self.moe_map.run(&self.router_ids, &self.map_down, &self.down_b, K_USED, 0, true)?;
        self.down_b.run(&self.down_all, &self.moe_act, &self.moe_down, &self.map_down, K_USED)?;
        self.moe_combine.run(
            &self.moe_down,
            &self.router_w,
            &self.sh,
            &self.sh_gate_logit,
            &self.block_out,
            K_USED,
            N_EMBD,
        )?;
        self.hc.write(&self.hc_ws, &self.r_b, &self.block_out, &self.r_c)?;

        self.fin.read(
            &self.fin_ws,
            &self.r_c,
            &HcWeights {
                norm: &self.fin_hc[0],
                down: &self.fin_hc[1],
                up: &self.fin_hc[2],
                inject: None,
            },
        )?;
        self.head.g.run(&self.head.w, &self.fin_ws.mixed, &self.logits)?;
        Ok(())
    }

    /// Forget every draft cell (zero the draft KV).
    pub fn reset(&mut self) -> Result<()> {
        self.len = 0;
        self.qsa_state.reset(self.gpu)
    }

    /// Front only (K/V of cell `pos` from host `r` and `tok`), no prediction:
    /// the catch-up for a cell whose trunk residual is known.
    pub fn catchup(&mut self, r: &[f32], tok: u32, pos: usize) -> Result<()> {
        if pos >= self.cells {
            bail!("draft KV cell {pos} is past its {} cells", self.cells);
        }
        self.set_r(r)?;
        self.gather(tok)?;
        batch_cmd::begin(self.gpu)?;
        let rec = self.record_front(pos);
        let end = batch_cmd::end(self.gpu);
        rec?;
        end?;
        self.len = self.len.max(pos + 1);
        Ok(())
    }

    fn gather(&self, tok: u32) -> Result<()> {
        // The gather uploads and waits by itself, so it goes before the batch opens.
        let staged = self.emb.gather_staged(
            self.gpu,
            self.model.token_embd()?,
            &vec![tok; HC],
            &self.emb4,
        )?;
        self.emb.release(self.gpu, &staged);
        Ok(())
    }

    /// One full draft step with `r` already set ([`Self::set_r`]).
    pub fn step(&mut self, tok: u32, pos: usize) -> Result<(u32, [f64; 4])> {
        self.step_ex(tok, pos, false)
    }

    /// One full draft step: embed `tok`, run the layer at KV cell `pos`, return
    /// the argmax draft token id and the per-phase milliseconds
    /// `(gather, record, gpu, readback)`. `chain`: condition on the previous
    /// step's own output stack instead of the host-set `r`.
    pub fn step_ex(&mut self, tok: u32, pos: usize, chain: bool) -> Result<(u32, [f64; 4])> {
        if pos >= self.cells {
            bail!("draft KV cell {pos} is past its {} cells", self.cells);
        }
        let t0 = std::time::Instant::now();
        self.gather(tok)?;
        let t1 = std::time::Instant::now();
        batch_cmd::begin(self.gpu)?;
        let rec = (|| -> Result<()> {
            if chain {
                batch_cmd::record_copy(self.gpu, &self.r_in, 0, &self.r_c, 0, (HC_DIM * 4) as u64)?;
            }
            self.record_front(pos)?;
            self.record_rest(pos)
        })();
        if let Err(e) = rec {
            let _ = batch_cmd::end(self.gpu);
            return Err(e);
        }
        let t2 = std::time::Instant::now();
        batch_cmd::end(self.gpu)?;
        let t3 = std::time::Instant::now();
        let raw = download(self.gpu, &self.logits, self.n_draft * 4)?;
        let logits = f32_from_bytes(&raw);
        // shortcut: host argmax over ~425 KB; there is no GPU argmax kernel, and
        // a chained step needs the host id for the next embedding anyway.
        let mut best = 0usize;
        for (i, &v) in logits.iter().enumerate() {
            if v > logits[best] {
                best = i;
            }
        }
        let t4 = std::time::Instant::now();
        self.len = self.len.max(pos + 1);
        let ms = |a: std::time::Instant, b: std::time::Instant| (b - a).as_secs_f64() * 1e3;
        Ok((
            self.rt_tokens[best] as u32,
            [ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4)],
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_roundtrip() {
        for v in [0.0f32, 1.0, -2.5, 0.333, 6.1e-5, 1.0e-6, 65000.0, -1.0e-7] {
            let r = f16_to_f32(f32_to_f16(v));
            let tol = (v.abs() * 1e-3).max(6e-8);
            assert!((r - v).abs() <= tol, "{v} -> {r}");
        }
        assert_eq!(f32_to_f16(1.0), 0x3c00);
    }

    #[test]
    fn q8_dequant_block() {
        let mut blk = vec![0u8; 34];
        blk[..2].copy_from_slice(&f32_to_f16(0.5).to_le_bytes());
        blk[2] = 4;
        blk[3] = (-2i8) as u8;
        let o = q8_0_to_f16(&blk);
        assert_eq!(f16_to_f32(u16::from_le_bytes([o[0], o[1]])), 2.0);
        assert_eq!(f16_to_f32(u16::from_le_bytes([o[2], o[3]])), -1.0);
        assert_eq!(o.len(), 64);
    }
}
