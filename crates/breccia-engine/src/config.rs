//! Model geometry, read from the GGUF metadata (never hard-coded).
//!
//! Every number here is a `qwen4exp.*` KV of the real artifact
//! (`docs/specs/qwen4exp-architecture.md` §1). The derived identities the spec
//! asks the engine to assert at load time (`conv_dim`, `ssm_v_heads*state ==
//! ssm.inner_size`, `hc_lr == hc_dim/32`, `ple_n_heads*ple_dim == n_embd`) are
//! checked in [`ModelConfig::validate`], so a different artifact fails loudly
//! instead of producing plausible-looking garbage.

use anyhow::{anyhow, bail, Result};
use breccia_format::metadata::{MetadataArray, MetadataMap, MetadataValue};

/// Which token mixer a block runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// Gated delta net (36 of 48 blocks: `(i+1) % 4 != 0`).
    Gdn,
    /// Full attention + lightning indexer (12 blocks: `i % 4 == 3`).
    Qsa,
}

/// The `ssm.*` geometry of a GDN block.
#[derive(Clone, Copy, Debug)]
pub struct SsmCfg {
    /// `ssm.state_size` — the delta-rule state edge (128).
    pub state: usize,
    /// `ssm.group_count` — key/value head count (16).
    pub h_k: usize,
    /// `ssm.time_step_rank` — value head count (48).
    pub h_v: usize,
    /// `ssm.conv_kernel` (4).
    pub conv: usize,
    /// `ssm.inner_size` = `h_v * state` (6144).
    pub inner: usize,
}

impl SsmCfg {
    /// `2*h_k*state + h_v*state` — the `attn_qkv` / conv width (10240).
    pub fn conv_channels(&self) -> usize {
        2 * self.h_k * self.state + self.h_v * self.state
    }
    /// `2*h_k*state` — the q|k half of `attn_qkv` (4096).
    pub fn qk_channels(&self) -> usize {
        2 * self.h_k * self.state
    }
}

/// The lightning-indexer geometry of a QSA block.
#[derive(Clone, Copy, Debug)]
pub struct IdxCfg {
    /// `attention.indexer.head_count` — query heads only (4).
    pub n_head: usize,
    /// `attention.indexer.key_length` (128).
    pub dim: usize,
    /// `attention.indexer.top_k` (2048).
    pub top_k: usize,
    /// `attention.compress_ratios` non-zero entry (4).
    pub block: usize,
}

/// The PLE (per-layer embedding) geometry — block 1 only.
#[derive(Clone, Debug)]
pub struct PleCfg {
    /// `ple.layers` (0-based; `[1]`).
    pub layers: Vec<usize>,
    /// `ple.ngram_size` (3).
    pub ngram: usize,
    /// `ple.heads_per_ngram` (8).
    pub heads_per_ngram: usize,
    /// `ple.conv_kernel` (4).
    pub conv: usize,
    /// `ple.eos_token_id` (248044) — cuts the n-gram window.
    pub eos: u32,
    /// `embedding_length_per_layer_input` (160).
    pub dim: usize,
    /// `ple.layer_multipliers` — the three u64 hash multipliers.
    pub mults: Vec<u64>,
    /// `ple.head_offsets` (16).
    pub offsets: Vec<u64>,
    /// `ple.head_vocab_sizes` (16).
    pub vocab: Vec<u64>,
}

impl PleCfg {
    /// `(ngram_size - 1) * heads_per_ngram` (16).
    pub fn n_head(&self) -> usize {
        (self.ngram - 1) * self.heads_per_ngram
    }
    /// Rows of conv history a sequence carries (`(conv-1) * ngram`, 9).
    pub fn hist(&self) -> usize {
        (self.conv - 1) * self.ngram
    }
}

/// Everything the graph needs to know before it touches a weight.
#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub arch: String,
    pub n_layers: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_rot: usize,
    pub rope_base: f64,
    pub eps: f32,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_ff: usize,
    pub n_ff_shared: usize,
    pub hc: usize,
    pub hc_lr: usize,
    pub qsa_interval: usize,
    pub ssm: SsmCfg,
    pub idx: IdxCfg,
    pub ple: PleCfg,
    pub vocab: usize,
    pub n_ctx_max: usize,
}

/// Read one `qwen4exp.<name>` (or `<arch>.<name>`) integer KV.
fn u64_kv(md: &MetadataMap<'_>, prefix: &str, key: &str) -> Result<usize> {
    let full = format!("{prefix}{key}");
    let v = md
        .get(&full)
        .ok_or_else(|| anyhow!("metadata is missing `{full}`"))?;
    v.as_u64()
        .map(|x| x as usize)
        .ok_or_else(|| anyhow!("`{full}` is not an unsigned integer"))
}

fn f64_kv(md: &MetadataMap<'_>, prefix: &str, key: &str) -> Result<f64> {
    let full = format!("{prefix}{key}");
    let v = md
        .get(&full)
        .ok_or_else(|| anyhow!("metadata is missing `{full}`"))?;
    v.as_f64()
        .ok_or_else(|| anyhow!("`{full}` is not a real number"))
}

fn array_kv<'a>(md: &'a MetadataMap<'a>, prefix: &str, key: &str) -> Result<MetadataArray<'a>> {
    let full = format!("{prefix}{key}");
    md.array(&full)
        .copied()
        .ok_or_else(|| anyhow!("metadata is missing the array `{full}`"))
}

/// `u64` view of an integer array, whichever integer type it declares.
fn arr_u64(a: &MetadataArray<'_>, what: &str) -> Result<Vec<u64>> {
    let mut out = Vec::with_capacity(a.len());
    for (i, v) in a.iter().enumerate() {
        let x = match v {
            MetadataValue::U8(x) => x as u64,
            MetadataValue::U16(x) => x as u64,
            MetadataValue::U32(x) => x as u64,
            MetadataValue::U64(x) => x,
            MetadataValue::I8(x) if x >= 0 => x as u64,
            MetadataValue::I16(x) if x >= 0 => x as u64,
            MetadataValue::I32(x) if x >= 0 => x as u64,
            MetadataValue::I64(x) if x >= 0 => x as u64,
            other => bail!(
                "{what}[{i}] is not a non-negative integer ({})",
                other.type_name()
            ),
        };
        out.push(x);
    }
    Ok(out)
}

impl ModelConfig {
    /// Build the geometry from an opened shard set's metadata.
    ///
    /// `vocab` is the output width of `token_embd` (the tensor is authoritative;
    /// the tokenizer KV list can be huge and is not read here).
    pub fn from_metadata(md: &MetadataMap<'_>, vocab: usize) -> Result<Self> {
        let arch = md
            .str("general.architecture")
            .ok_or_else(|| anyhow!("metadata has no `general.architecture`"))?
            .to_string();
        let prefix = format!("{arch}.");

        let ple_layers = arr_u64(&array_kv(md, &prefix, "ple.layers")?, "ple.layers")?
            .into_iter()
            .map(|x| x as usize)
            .collect::<Vec<_>>();
        let n_head = u64_kv(md, &prefix, "attention.indexer.head_count")?;
        let ple_dim = u64_kv(md, &prefix, "embedding_length_per_layer_input")?;
        let ngram = u64_kv(md, &prefix, "ple.ngram_size")?;
        let hpn = u64_kv(md, &prefix, "ple.heads_per_ngram")?;

        let cfg = ModelConfig {
            arch: arch.clone(),
            n_layers: u64_kv(md, &prefix, "block_count")?,
            n_embd: u64_kv(md, &prefix, "embedding_length")?,
            n_head: u64_kv(md, &prefix, "attention.head_count")?,
            n_head_kv: u64_kv(md, &prefix, "attention.head_count_kv")?,
            head_dim: u64_kv(md, &prefix, "attention.key_length")?,
            n_rot: u64_kv(md, &prefix, "rope.dimension_count")?,
            rope_base: f64_kv(md, &prefix, "rope.freq_base")?,
            eps: f64_kv(md, &prefix, "attention.layer_norm_rms_epsilon")? as f32,
            n_expert: u64_kv(md, &prefix, "expert_count")?,
            n_expert_used: u64_kv(md, &prefix, "expert_used_count")?,
            n_ff: u64_kv(md, &prefix, "expert_feed_forward_length")?,
            n_ff_shared: u64_kv(md, &prefix, "expert_shared_feed_forward_length")?,
            hc: u64_kv(md, &prefix, "hyper_connection.count")?,
            hc_lr: u64_kv(md, &prefix, "hyper_connection.low_rank")?,
            qsa_interval: u64_kv(md, &prefix, "full_attention_interval")?,
            ssm: SsmCfg {
                state: u64_kv(md, &prefix, "ssm.state_size")?,
                h_k: u64_kv(md, &prefix, "ssm.group_count")?,
                h_v: u64_kv(md, &prefix, "ssm.time_step_rank")?,
                conv: u64_kv(md, &prefix, "ssm.conv_kernel")?,
                inner: u64_kv(md, &prefix, "ssm.inner_size")?,
            },
            idx: IdxCfg {
                n_head,
                dim: u64_kv(md, &prefix, "attention.indexer.key_length")?,
                top_k: u64_kv(md, &prefix, "attention.indexer.top_k")?,
                block: indexer_block(&array_kv(md, &prefix, "attention.compress_ratios")?)?,
            },
            ple: PleCfg {
                layers: ple_layers,
                ngram,
                heads_per_ngram: hpn,
                conv: u64_kv(md, &prefix, "ple.conv_kernel")?,
                eos: u64_kv(md, &prefix, "ple.eos_token_id")? as u32,
                dim: ple_dim,
                mults: arr_u64(&array_kv(md, &prefix, "ple.layer_multipliers")?, "mults")?,
                offsets: arr_u64(&array_kv(md, &prefix, "ple.head_offsets")?, "offsets")?,
                vocab: arr_u64(
                    &array_kv(md, &prefix, "ple.head_vocab_sizes")?,
                    "head_vocab",
                )?,
            },
            vocab,
            n_ctx_max: u64_kv(md, &prefix, "context_length")?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// The spec's load-time identities (§2). A failure here is a wrong artifact,
    /// not a wrong runtime option.
    pub fn validate(&self) -> Result<()> {
        let want = |what: &str, got: usize, expect: usize| -> Result<()> {
            if got != expect {
                bail!("{what} is {got}, expected {expect}");
            }
            Ok(())
        };
        if self.arch != "qwen4exp" {
            bail!("architecture is `{}`, expected `qwen4exp`", self.arch);
        }
        want(
            "2*h_k*state + h_v*state (conv width)",
            self.ssm.conv_channels(),
            self.n_embd * self.hc,
        )?;
        want(
            "h_v*state (ssm.inner_size)",
            self.ssm.h_v * self.ssm.state,
            self.ssm.inner,
        )?;
        want(
            "hc*low_rank == hc_dim/32",
            self.hc_lr * 32,
            self.n_embd * self.hc,
        )?;
        want(
            "ple heads * ple dim",
            self.ple.n_head() * self.ple.dim,
            self.n_embd,
        )?;
        want(
            "ple mults (ngram_size)",
            self.ple.mults.len(),
            self.ple.ngram,
        )?;
        want("ple head count", self.ple.offsets.len(), self.ple.n_head())?;
        want("ple head count", self.ple.vocab.len(), self.ple.n_head())?;
        want(
            "rope n_rot <= head_dim",
            self.n_rot.min(self.head_dim),
            self.n_rot,
        )?;
        want(
            "qsa interval divides the block count",
            self.n_layers % self.qsa_interval,
            0,
        )?;
        for &l in &self.ple.layers {
            if l >= self.n_layers {
                bail!("ple.layers lists block {l} of {}", self.n_layers);
            }
        }
        Ok(())
    }

    /// `(i + 1) % qsa_interval == 0` ⇒ full attention (`i % 4 == 3`).
    pub fn layer_kind(&self, i: usize) -> LayerKind {
        if (i + 1).is_multiple_of(self.qsa_interval) {
            LayerKind::Qsa
        } else {
            LayerKind::Gdn
        }
    }

    /// Blocks that run the PLE insertion (`[1]` for this artifact).
    pub fn is_ple_layer(&self, i: usize) -> bool {
        self.ple.layers.contains(&i)
    }

    /// `hc * n_embd` (10240) — the residual stack width.
    pub fn hc_dim(&self) -> usize {
        self.hc * self.n_embd
    }

    /// `n_head * head_dim` (6144) — the attention output / `attn_gate` width.
    pub fn attn_out_dim(&self) -> usize {
        self.n_head * self.head_dim
    }

    /// `n_head * 2 * head_dim` (12288) — the `attn_q` q|gate width.
    pub fn qgate_dim(&self) -> usize {
        self.n_head * 2 * self.head_dim
    }

    /// Number of GDN / QSA blocks.
    pub fn layer_counts(&self) -> (usize, usize) {
        let mut gdn = 0;
        for i in 0..self.n_layers {
            if self.layer_kind(i) == LayerKind::Gdn {
                gdn += 1;
            }
        }
        (gdn, self.n_layers - gdn)
    }

    /// One-line geometry summary for the load banner.
    pub fn summary(&self) -> String {
        let (gdn, qsa) = self.layer_counts();
        format!(
            "{} · {} blocks ({} GDN + {} QSA) · n_embd {} · hc {}×{} (lr {}) · \
             experts {}/top{} ×{} · ssm S={} hv={} · idx {}×{} top{} · ple blk{:?} · vocab {}",
            self.arch,
            self.n_layers,
            gdn,
            qsa,
            self.n_embd,
            self.hc,
            self.n_embd,
            self.hc_lr,
            self.n_expert,
            self.n_expert_used,
            self.n_ff,
            self.ssm.state,
            self.ssm.h_v,
            self.idx.n_head,
            self.idx.dim,
            self.idx.top_k,
            self.ple.layers,
            self.vocab
        )
    }
}

/// The non-zero entry of `attention.compress_ratios` (4 at every QSA block).
fn indexer_block(a: &MetadataArray<'_>) -> Result<usize> {
    let vals = arr_u64(a, "compress_ratios")?;
    let mut seen = None;
    for v in vals {
        if v == 0 {
            continue;
        }
        match seen {
            None => seen = Some(v as usize),
            Some(prev) if prev == v as usize => {}
            Some(prev) => bail!("compress_ratios has two non-zero values ({prev}, {v})"),
        }
    }
    seen.ok_or_else(|| anyhow!("compress_ratios has no non-zero entry"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The artifact's geometry, as the spec records it (§1/§2).
    fn real() -> ModelConfig {
        ModelConfig {
            arch: "qwen4exp".into(),
            n_layers: 48,
            n_embd: 2560,
            n_head: 24,
            n_head_kv: 2,
            head_dim: 256,
            n_rot: 64,
            rope_base: 1e7,
            eps: 1e-6,
            n_expert: 512,
            n_expert_used: 10,
            n_ff: 640,
            n_ff_shared: 640,
            hc: 4,
            hc_lr: 320,
            qsa_interval: 4,
            ssm: SsmCfg {
                state: 128,
                h_k: 16,
                h_v: 48,
                conv: 4,
                inner: 6144,
            },
            idx: IdxCfg {
                n_head: 4,
                dim: 128,
                top_k: 2048,
                block: 4,
            },
            ple: PleCfg {
                layers: vec![1],
                ngram: 3,
                heads_per_ngram: 8,
                conv: 4,
                eos: 248044,
                dim: 160,
                mults: vec![23703573157769, 20109073645365, 8052911324071],
                offsets: (0..16).map(|i| i * 20000000).collect(),
                vocab: vec![20000003; 16],
            },
            vocab: 248320,
            n_ctx_max: 262144,
        }
    }

    #[test]
    fn layer_taxonomy_matches_the_spec() {
        let c = real();
        assert_eq!(c.layer_counts(), (36, 12));
        for i in 0..48 {
            let want = if i % 4 == 3 {
                LayerKind::Qsa
            } else {
                LayerKind::Gdn
            };
            assert_eq!(c.layer_kind(i), want, "block {i}");
        }
        assert!(c.is_ple_layer(1));
        assert_eq!(c.hc_dim(), 10240);
        assert_eq!(c.attn_out_dim(), 6144);
        assert_eq!(c.qgate_dim(), 12288);
        assert_eq!(c.ssm.conv_channels(), 10240);
        assert_eq!(c.ssm.qk_channels(), 4096);
        assert_eq!(c.ple.n_head(), 16);
        assert_eq!(c.ple.hist(), 9);
    }

    #[test]
    fn validate_rejects_a_wrong_identity() {
        let mut c = real();
        c.hc_lr = 321;
        assert!(c.validate().is_err());
        let mut c = real();
        c.ssm.h_v = 47;
        assert!(c.validate().is_err());
        let mut c = real();
        c.qsa_interval = 5;
        assert!(c.validate().is_err());
        let mut c = real();
        c.ple.layers = vec![48];
        assert!(c.validate().is_err());
        assert!(real().validate().is_ok());
    }
}
