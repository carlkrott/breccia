//! `inkling2_loop` (Inkling-2-Mini) config, loop plan and weight binder.
//!
//! Stage S1 of `docs/specs/inkling2-loop-port.md`: everything here is CPU-only and reads
//! the GGUF without touching a device. The graph itself lands in later
//! stages; until then `Model::load` refuses the arch with a clear error.

use anyhow::{anyhow, bail, ensure, Result};
use breccia_format::{GgmlType, MetadataMap, ShardSet};

pub const ARCH: &str = "inkling2_loop";
const P: &str = "inkling2_loop.";

/// Hyperparameters of one `inkling2_loop` GGUF.
#[derive(Clone, Debug, PartialEq)]
pub struct InklingConfig {
    pub n_layers: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff_exp: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub eps: f32,
    /// `true` = sliding-window layer, `false` = full attention.
    pub swa: Vec<bool>,
    pub sliding_window: usize,
    pub n_conv: usize,
    pub expert_weights_scale: f32,
    pub situ_beta: f32,
    pub situ_linear_beta: f32,
    pub attn_res_block: usize,
    pub softcap: f32,
    pub n_rel: usize,
    pub rel_extent: usize,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub log_scale_floor: f32,
    pub log_scale_alpha: f32,
    pub loop_start: usize,
    pub loop_end: usize,
    pub loop_count: usize,
    pub loop_rank: usize,
    /// Real vocabulary; logits past it are padding rows of `output.weight`.
    pub vocab: usize,
    pub vocab_padded: usize,
    pub n_ctx_max: usize,
}

fn kv<'a>(md: &'a MetadataMap<'a>, key: &str) -> Result<&'a breccia_format::MetadataValue<'a>> {
    md.get(&format!("{P}{key}"))
        .ok_or_else(|| anyhow!("metadata is missing `{P}{key}`"))
}
fn u(md: &MetadataMap<'_>, key: &str) -> Result<usize> {
    kv(md, key)?
        .as_u64()
        .map(|x| x as usize)
        .ok_or_else(|| anyhow!("`{P}{key}` is not an unsigned integer"))
}
fn f(md: &MetadataMap<'_>, key: &str) -> Result<f32> {
    kv(md, key)?
        .as_f64()
        .map(|x| x as f32)
        .ok_or_else(|| anyhow!("`{P}{key}` is not a number"))
}
/// A per-layer count that this port only supports as one uniform value.
fn uniform(md: &MetadataMap<'_>, key: &str, n: usize) -> Result<usize> {
    let full = format!("{P}{key}");
    let v = md
        .get(&full)
        .ok_or_else(|| anyhow!("metadata is missing `{full}`"))?;
    if let Some(x) = v.as_u64() {
        return Ok(x as usize);
    }
    let a = v
        .as_array()
        .ok_or_else(|| anyhow!("`{full}` is neither an integer nor an array"))?
        .to_i64_vec()?;
    ensure!(
        a.len() == n,
        "`{full}` has {} entries, expected {n}",
        a.len()
    );
    ensure!(
        a[0] > 0 && a.iter().all(|&x| x == a[0]),
        "`{full}` varies per layer or is not positive: {a:?} (not supported)"
    );
    Ok(a[0] as usize)
}

impl InklingConfig {
    pub fn from_metadata(md: &MetadataMap<'_>) -> Result<Self> {
        let arch = md.str("general.architecture").unwrap_or("");
        ensure!(arch == ARCH, "architecture is `{arch}`, expected `{ARCH}`");
        let n_layers = u(md, "block_count")?;
        let swa = md
            .array(&format!("{P}attention.sliding_window_pattern"))
            .ok_or_else(|| anyhow!("metadata is missing `{P}attention.sliding_window_pattern`"))?
            .to_bool_vec()?;
        let head_dim = u(md, "attention.key_length")?;
        let c = InklingConfig {
            n_layers,
            n_embd: u(md, "embedding_length")?,
            n_head: uniform(md, "attention.head_count", n_layers)?,
            n_head_kv: uniform(md, "attention.head_count_kv", n_layers)?,
            head_dim,
            n_ff_exp: u(md, "expert_feed_forward_length")?,
            n_expert: u(md, "expert_count")?,
            n_expert_used: u(md, "expert_used_count")?,
            eps: f(md, "attention.layer_norm_rms_epsilon")?,
            swa,
            sliding_window: u(md, "attention.sliding_window")?,
            n_conv: u(md, "shortconv.l_cache")?,
            expert_weights_scale: f(md, "expert_weights_scale")?,
            situ_beta: f(md, "activation.situ_beta")?,
            situ_linear_beta: f(md, "activation.situ_linear_beta")?,
            attn_res_block: u(md, "attn_res.block_size")?,
            softcap: f(md, "final_logit_softcapping")?,
            n_rel: u(md, "attention.relative_length")?,
            rel_extent: u(md, "attention.relative_extent")?,
            qk_clip: f(md, "attention.qk_clip")?,
            rel_clip: f(md, "attention.relative_clip")?,
            log_scale_floor: f(md, "attention.log_scale_floor")?,
            log_scale_alpha: f(md, "attention.log_scale_alpha")?,
            loop_start: u(md, "loop.start")?,
            loop_end: u(md, "loop.end")?,
            loop_count: u(md, "loop.count")?,
            loop_rank: u(md, "loop.gate_rank")?,
            vocab: u(md, "vocab_size_unpadded")?,
            vocab_padded: u(md, "vocab_size")?,
            n_ctx_max: u(md, "context_length")?,
        };
        for k in ["value_length", "key_length_swa", "value_length_swa"] {
            ensure!(
                u(md, &format!("attention.{k}"))? == head_dim,
                "`{P}attention.{k}` != key_length (not supported)"
            );
        }
        ensure!(
            u(md, "rope.dimension_count")? == 0,
            "`{P}rope.dimension_count` != 0: this port has no RoPE"
        );
        if let Some(v) = md.get(&format!("{P}loop.varying")) {
            ensure!(
                v.as_bool() == Some(false),
                "`{P}loop.varying` = true is not supported"
            );
        }
        c.validate()?;
        Ok(c)
    }

    pub fn validate(&self) -> Result<()> {
        let c = self;
        ensure!(
            c.swa.len() == c.n_layers,
            "sliding_window_pattern has {} entries, expected {}",
            c.swa.len(),
            c.n_layers
        );
        ensure!(
            c.n_head.is_multiple_of(c.n_head_kv),
            "n_head {} is not a multiple of n_head_kv {}",
            c.n_head,
            c.n_head_kv
        );
        ensure!(
            c.loop_start < c.loop_end && c.loop_end <= c.n_layers && c.loop_count >= 1,
            "bad loop range {}..{} x{}",
            c.loop_start,
            c.loop_end,
            c.loop_count
        );
        ensure!(
            c.n_expert_used <= c.n_expert,
            "expert_used {} > expert_count {}",
            c.n_expert_used,
            c.n_expert
        );
        ensure!(
            c.vocab <= c.vocab_padded,
            "vocab_size_unpadded {} > vocab_size {}",
            c.vocab,
            c.vocab_padded
        );
        Ok(())
    }

    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }
    pub fn plan(&self) -> LoopPlan {
        LoopPlan {
            n_layers: self.n_layers,
            start: self.loop_start,
            end: self.loop_end,
            count: self.loop_count,
        }
    }
}

/// Execution order of the looped stack and its KV-slot numbering.
///
/// Prelude `0..start`, then the core `start..end` `count` times, then the
/// post `end..n_layers`. Pass 0 and the prelude/post use slot = layer (as in
/// llama.cpp); pass `p >= 1` gets slots `n_layers + (p-1)*len + (l-start)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoopPlan {
    pub n_layers: usize,
    pub start: usize,
    pub end: usize,
    pub count: usize,
}

/// One executed layer: which weights (`layer`) and which KV/state slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    pub slot: usize,
    pub layer: usize,
    pub pass: usize,
}

impl LoopPlan {
    pub fn n_slots(&self) -> usize {
        self.n_layers + (self.count - 1) * (self.end - self.start)
    }
    pub fn slot(&self, pass: usize, layer: usize) -> usize {
        if pass == 0 {
            layer
        } else {
            debug_assert!((self.start..self.end).contains(&layer) && pass < self.count);
            self.n_layers + (pass - 1) * (self.end - self.start) + (layer - self.start)
        }
    }
    pub fn steps(&self) -> Vec<Step> {
        let mut v: Vec<Step> = (0..self.start)
            .map(|l| Step {
                slot: l,
                layer: l,
                pass: 0,
            })
            .collect();
        for pass in 0..self.count {
            v.extend((self.start..self.end).map(|l| Step {
                slot: self.slot(pass, l),
                layer: l,
                pass,
            }));
        }
        v.extend((self.end..self.n_layers).map(|l| Step {
            slot: l,
            layer: l,
            pass: 0,
        }));
        v
    }
    /// Physical layer behind a slot.
    pub fn layer_of(&self, slot: usize) -> usize {
        if slot < self.n_layers {
            slot
        } else {
            self.start + (slot - self.n_layers) % (self.end - self.start)
        }
    }
}

/// Check every tensor's shape and type against `cfg`; return how many matched.
///
/// Types are exact: Breccia has kernels for F32/Q4_K/Q6_K/IQ4_NL only, so the
/// stock GGUF (Q8_0/Q5_0 parts) must be requantized first (see the port doc).
pub fn bind(shards: &ShardSet, cfg: &InklingConfig) -> Result<usize> {
    use GgmlType as T;
    let (d, q, kv, nc, r) = (
        cfg.n_embd as u64,
        cfg.q_dim() as u64,
        cfg.kv_dim() as u64,
        cfg.n_conv as u64,
        cfg.loop_rank as u64,
    );
    let (h, hkv, hd, ne, ff) = (
        cfg.n_head as u64,
        cfg.n_head_kv as u64,
        cfg.head_dim as u64,
        cfg.n_expert as u64,
        cfg.n_ff_exp as u64,
    );
    let mut want: Vec<(String, Vec<u64>, GgmlType)> = vec![
        (
            "token_embd.weight".into(),
            vec![d, cfg.vocab_padded as u64],
            T::Q4_K,
        ),
        ("token_embd_norm.weight".into(), vec![d], T::F32),
        (
            "output.weight".into(),
            vec![d, cfg.vocab_padded as u64],
            T::Q6_K,
        ),
        ("output_norm.weight".into(), vec![d], T::F32),
        ("output_res_query.weight".into(), vec![d], T::F32),
    ];
    for l in 0..cfg.n_layers {
        let mut add = |n: &str, dims: &[u64], ty: GgmlType| {
            want.push((format!("blk.{l}.{n}"), dims.to_vec(), ty))
        };
        add("attn_q.weight", &[d, q], T::Q6_K);
        add("attn_gate.weight", &[d, q], T::Q6_K);
        add("attn_k.weight", &[d, kv], T::Q6_K);
        add("attn_v.weight", &[d, kv], T::Q6_K);
        add("attn_output.weight", &[q, d], T::Q6_K);
        add("attn_q_clip.weight", &[hd, h], T::F32);
        add("attn_sink_proj.weight", &[d, h], T::Q4_K);
        add("attn_rel.weight", &[d, cfg.n_rel as u64 * h], T::Q4_K);
        add(
            "attn_rel_bias.weight",
            &[cfg.n_rel as u64, cfg.rel_extent as u64],
            T::F32,
        );
        add("attn_rel_clip.weight", &[cfg.n_rel as u64, h], T::F32);
        add("attn_k_conv.weight", &[nc, kv], T::F32);
        add("attn_v_conv.weight", &[nc, kv], T::F32);
        add("attn_post_conv.weight", &[nc, d], T::F32);
        add("ffn_post_conv.weight", &[nc, d], T::F32);
        if cfg.swa[l] {
            add("attn_norm.weight", &[d], T::F32);
        } else {
            for n in [
                "attn_q_inp_norm.weight",
                "attn_k_inp_norm.weight",
                "attn_v_inp_norm.weight",
            ] {
                add(n, &[d], T::F32);
            }
            add("attn_k_conv_proj.weight", &[d, nc * hkv], T::Q6_K);
            add("attn_v_conv_proj.weight", &[d, nc * hkv], T::Q6_K);
        }
        if l > 0 {
            let dims = if cfg.swa[l] { vec![d] } else { vec![d, 3] };
            add("attn_res_query.weight", &dims, T::F32);
        }
        add("ffn_res_query.weight", &[d], T::F32);
        add("ffn_norm.weight", &[d], T::F32);
        add("ffn_gate_inp.weight", &[d, ne], T::F32);
        add("exp_probs_b.bias", &[ne], T::F32);
        add("ffn_gate_up_exps.weight", &[d, 2 * ff, ne], T::Q4_K);
        add("ffn_down_exps.weight", &[ff, d, ne], T::IQ4_NL);
        if (cfg.loop_start..cfg.loop_end).contains(&l) {
            add("loop_msg_query.weight", &[d, 4], T::F32);
            for c in ["q", "k", "v", "ffn"] {
                add(&format!("loop_{c}_ctrl_down.weight"), &[2 * d, r], T::F32);
                add(&format!("loop_{c}_ctrl_gain.weight"), &[r, d], T::F32);
                add(&format!("loop_{c}_ctrl_forget.weight"), &[r, d], T::F32);
            }
        }
    }
    for (name, dims, ty) in &want {
        let t = shards
            .tensor(name)
            .ok_or_else(|| anyhow!("missing tensor `{name}`"))?;
        ensure!(
            t.dims() == dims.as_slice(),
            "`{name}` has shape {:?}, expected {dims:?}",
            t.dims()
        );
        if t.ty() != *ty {
            bail!(
                "`{name}` is {}, expected {} (requantize the stock GGUF with the port doc's recipe)",
                t.ty().name().unwrap_or("?"),
                ty.name().unwrap_or("?")
            );
        }
    }
    let n = shards.tensor_count();
    ensure!(
        n == want.len(),
        "the file has {n} tensors, the binder knows {} (unknown extras)",
        want.len()
    );
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    const REAL: LoopPlan = LoopPlan {
        n_layers: 24,
        start: 6,
        end: 18,
        count: 3,
    };

    #[test]
    fn plan_matches_llama_cpp_slots() {
        assert_eq!(REAL.n_slots(), 48);
        let s = REAL.steps();
        assert_eq!(s.len(), 48);
        let slots: Vec<usize> = s.iter().map(|x| x.slot).collect();
        let want: Vec<usize> = (0..18).chain(24..48).chain(18..24).collect();
        assert_eq!(slots, want);
        // every slot used once, and its layer round-trips
        let mut seen = slots.clone();
        seen.sort();
        assert_eq!(seen, (0..48).collect::<Vec<_>>());
        for st in &s {
            assert_eq!(REAL.layer_of(st.slot), st.layer);
        }
        // full-attention slots (layers 5/11/17/23)
        let swa = |l: usize| l % 6 != 5;
        let full: Vec<usize> = s
            .iter()
            .filter(|x| !swa(x.layer))
            .map(|x| x.slot)
            .collect::<Vec<_>>();
        let mut full = full;
        full.sort();
        assert_eq!(full, vec![5, 11, 17, 23, 29, 35, 41, 47]);
    }

    /// Binds the real requantized GGUF when `BRECCIA_INKLING_GGUF` points at it.
    #[test]
    fn binds_the_real_file() -> Result<()> {
        let Ok(path) = std::env::var("BRECCIA_INKLING_GGUF") else {
            eprintln!("skipped: BRECCIA_INKLING_GGUF unset");
            return Ok(());
        };
        let shards = ShardSet::open_family(&path)?;
        let cfg = InklingConfig::from_metadata(&shards.metadata()?)?;
        assert_eq!(
            (cfg.n_layers, cfg.n_embd, cfg.n_head, cfg.n_head_kv),
            (24, 2304, 24, 8)
        );
        assert_eq!((cfg.vocab, cfg.vocab_padded), (200058, 201024));
        assert_eq!(cfg.plan(), REAL);
        assert_eq!(cfg.swa.iter().filter(|&&s| !s).count(), 4);
        assert_eq!(bind(&shards, &cfg)?, 704);
        Ok(())
    }
}
