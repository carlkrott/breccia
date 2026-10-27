//! Loading: devices, placement, dense weights → VRAM, experts → the mmap lane.
//!
//! # What is uploaded and what is not
//!
//! `docs/phase-3/05` §3 and `docs/specs/fork-flags-semantics.md` fix the split:
//!
//! | group | where | bytes |
//! |---|---|---|
//! | dense (attention/GDN/HC/router/shared expert/indexer/PLE module) | VRAM, split by [`Placement`] | 3.17 GiB |
//! | `output.weight` (the head) | VRAM, last device | 417 MiB |
//! | `token_embd` | **not uploaded** — one row is staged per token | 260 MiB on disk |
//! | routed experts (144 tensors) | host `mmap`, demand-paged | 39.97 GiB |
//! | `per_layer_token_embd` (PLE table) | host `mmap`, **16 rows per token** | 26.82 GiB |
//!
//! The two big groups are never touched by a load: the file is mapped read-only
//! and a token reads ~0.84 GB of expert bytes plus 16 × 90 B of PLE rows
//! (`docs/phase-3/07`, `docs/phase-3/08` §6.1). Uploading the dense set is the
//! only bulk copy, and it is measured (`load.upload_seconds`).

use std::collections::BTreeSet;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use breccia_format::layout::{ExpertRole, Group};
use breccia_format::ShardSet;
use breccia_kernels::cpu::LayerPlan;
use breccia_kernels::{Device, Gemv, QuantType};
use breccia_vk::ctx::{Buf, Gpu};

use crate::arena::{
    self, ArenaBlock, ArenaOpts, ArenaReport, ArenaSpec, BudgetSource, DeviceArenaRow, DeviceBudget,
};
use crate::config::{LayerKind, ModelConfig};
use crate::head::{self, HeadCandidate, HeadSpec};
use crate::placement::{self, Placement};
/// The identity of a projection: one compiled pipeline per distinct triple.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PipeKey {
    pub ty: QuantType,
    pub rows: usize,
    pub cols: usize,
}

impl PipeKey {
    fn of(t: &breccia_format::gguf::TensorView<'_>) -> Result<Self> {
        let ty = QuantType::from_ggml(t.ty())
            .ok_or_else(|| anyhow!("{}: ggml type {} has no kernel", t.name(), t.ty().id()))?;
        Ok(PipeKey {
            ty,
            rows: t.ne1() as usize,
            cols: t.ne0() as usize,
        })
    }
    /// Weight bytes of the whole projection (what one GEMV reads).
    pub fn weight_bytes(&self) -> u64 {
        (self.rows * self.cols.div_ceil(self.ty.block_elems()) * self.ty.block_bytes()) as u64
    }
}

/// One block's routed experts, resident in VRAM (`-ot`-equivalent).
///
/// The whole role tensor is uploaded — all 512 experts — exactly as the
/// production HIP config does with `-ot 'blk\.N\.ffn_(up|down|gate)_exps\.=ROCmM'`,
/// so expert `id` sits at `id * blob_words` with no per-block bookkeeping.
pub struct PlacedExperts {
    pub block: usize,
    pub device: usize,
    pub gate: W,
    pub up: W,
    pub down: W,
}

impl PlacedExperts {
    fn bytes(&self) -> u64 {
        self.gate.buf.size + self.up.buf.size + self.down.buf.size
    }
}

/// A projection weight plus the pipeline shape it needs.
pub struct W {
    pub buf: Buf,
    pub key: PipeKey,
}

/// The three hyper-connection projection weights plus the grouped gamma.
pub struct HcW {
    pub norm: Buf,
    pub down: W,
    pub up: W,
    pub inject: Option<Buf>,
}

impl HcW {
    fn bytes(&self) -> u64 {
        self.norm.size
            + self.down.buf.size
            + self.up.buf.size
            + self.inject.as_ref().map_or(0, |b| b.size)
    }
}

/// The GDN-only tensors (§6.2).
pub struct GdnW {
    /// `ssm_beta.weight` + `ssm_alpha.weight` concatenated **along rows** into one
    /// `[96, 2560]` tensor of the same quant type (§8ay), so one 96-row GEMV
    /// replaces the two 48-row dispatches. Each row is an independent dot product,
    /// so the outputs are bit-identical to the separate runs.
    pub beta_alpha: W,
    pub qkv: W,
    pub z: W,
    pub beta: W,
    pub alpha: W,
    pub out: W,
    pub conv: Buf,
    pub dt: Buf,
    pub a: Buf,
    pub ssm_norm: Buf,
}

/// The QSA-only tensors (§6.3).
pub struct QsaW {
    pub q: W,
    pub k: W,
    pub v: W,
    pub idx_q: W,
    pub idx_k: W,
    pub out: W,
    pub q_norm: Buf,
    pub k_norm: Buf,
    pub idx_q_norm: Buf,
    pub idx_k_norm: Buf,
}

/// The MoE tensors every block has (§6.1).
pub struct MoeW {
    /// `ffn_gate_inp` — the router, `[n_embd, n_expert]`.
    pub router: W,
    /// `ffn_gate_inp_shexp` — the shared expert's scalar gate, `[n_embd]`.
    pub shexp_gate: W,
    pub gate: W,
    pub up: W,
    pub down: W,
}

/// Everything one block needs on its device.
pub struct LayerWeights {
    pub block: usize,
    pub kind: LayerKind,
    pub device: usize,
    pub hc_attn: HcW,
    pub hc_ffn: HcW,
    pub moe: MoeW,
    pub gdn: Option<GdnW>,
    pub qsa: Option<QsaW>,
    /// Host-lane geometry of this block's 512 routed experts (`Copy`).
    pub experts: LayerPlan,
    pub bytes: u64,
}

impl LayerWeights {
    fn release(&self, gpu: &Gpu) {
        for b in [&self.hc_attn.norm, &self.hc_ffn.norm] {
            gpu.destroy_buf(b);
        }
        for w in [
            &self.hc_attn.down,
            &self.hc_attn.up,
            &self.hc_ffn.down,
            &self.hc_ffn.up,
            &self.moe.router,
            &self.moe.shexp_gate,
            &self.moe.gate,
            &self.moe.up,
            &self.moe.down,
        ] {
            gpu.destroy_buf(&w.buf);
        }
        for b in [&self.hc_attn.inject, &self.hc_ffn.inject]
            .into_iter()
            .flatten()
        {
            gpu.destroy_buf(b);
        }
        if let Some(g) = &self.gdn {
            for w in [&g.qkv, &g.z, &g.beta, &g.alpha, &g.out] {
                gpu.destroy_buf(&w.buf);
            }
            for b in [&g.conv, &g.dt, &g.a, &g.ssm_norm] {
                gpu.destroy_buf(b);
            }
        }
        if let Some(q) = &self.qsa {
            for w in [&q.q, &q.k, &q.v, &q.idx_q, &q.idx_k, &q.out] {
                gpu.destroy_buf(&w.buf);
            }
            for b in [&q.q_norm, &q.k_norm, &q.idx_q_norm, &q.idx_k_norm] {
                gpu.destroy_buf(b);
            }
        }
    }
}

/// The PLE tensors' raw host bytes (small: 200 KiB) — the table itself stays in
/// the mapping and is gathered row by row.
pub struct PleRaw {
    pub norm_key: Vec<u8>,
    pub norm_query: Vec<u8>,
    pub norm_conv: Vec<u8>,
    pub conv: Vec<u8>,
    /// `ple_key` / `ple_value` (GPU projections).
    pub key: W,
    pub value: W,
}

/// One device's share of the load, for the banner and the report.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct DeviceLoad {
    pub ordinal: usize,
    pub label: String,
    pub name: String,
    pub pci: String,
    pub driver: String,
    pub blocks: (usize, usize),
    pub dense_bytes: u64,
    pub vram_total: u64,
    pub upload_seconds: f64,
}

/// The measured load, for `docs/phase-4/01-forward-pass.md`.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct LoadStats {
    pub open_seconds: f64,
    pub upload_seconds: f64,
    pub dense_bytes: u64,
    pub head_bytes: u64,
    pub expert_bytes_mapped: u64,
    pub ple_bytes_mapped: u64,
    /// RSS after the load (experts and PLE are demand-paged, so this is small).
    pub rss_bytes: u64,
    pub rss_peak_bytes: u64,
    pub devices: Vec<DeviceLoad>,
    /// KV + recurrent state bytes the session will add.
    pub state_bytes: u64,
    pub plan: Vec<String>,
}

impl LoadStats {
    /// The banner `breccia bench token` prints before it runs anything.
    pub fn banner(&self, cfg: &ModelConfig) -> String {
        let mut s = String::new();
        s.push_str(&format!("model   {}\n", cfg.summary()));
        s.push_str(&format!(
            "load    {:.2} s open + {:.2} s upload · dense {} MiB + head {} MiB · \
             experts {} MiB mapped · PLE {} MiB lazy · RSS {} MiB (peak {} MiB)\n",
            self.open_seconds,
            self.upload_seconds,
            self.dense_bytes >> 20,
            self.head_bytes >> 20,
            self.expert_bytes_mapped >> 20,
            self.ple_bytes_mapped >> 20,
            self.rss_bytes >> 20,
            self.rss_peak_bytes >> 20
        ));
        for l in &self.plan {
            s.push_str(l);
            s.push('\n');
        }
        s
    }
}

/// A loaded model: devices, weights, placement, host-lane geometry.
pub struct Model {
    pub cfg: ModelConfig,
    pub shards: Arc<ShardSet>,
    pub placement: Placement,
    pub devices: Vec<Device>,
    pub layers: Vec<LayerWeights>,
    /// The device the LM head (and the final mixer) lives on. **Not** necessarily
    /// the device of the last block: `--head-device` moves it, and `submit_head`
    /// hands the residual over when they differ (`docs/phase-4` §8ar).
    pub head_device: usize,
    pub head: W,
    pub out_hc_norm: Buf,
    pub out_hc_down: W,
    pub out_hc_up: W,
    pub ple: Option<PleRaw>,
    /// Blocks whose routed experts live in VRAM, keyed by block.
    pub placed: std::collections::BTreeMap<usize, PlacedExperts>,
    /// The arena decision (rule, budget source, chosen set) for the report.
    pub arena: Option<ArenaReport>,
    pub stats: LoadStats,
    /// Distinct pipelines the graph will need, discovered at load.
    pub keys: BTreeSet<PipeKey>,
}

/// How to load.
#[derive(Clone, Debug)]
pub struct LoadOptions {
    /// Device selectors (`0000:03:00.0`, `1`, `default`); empty = first discrete.
    pub devices: Vec<String>,
    /// KV cells the session may use (the indexer selection width caps here too).
    pub ctx_capacity: usize,
    /// Explicit block counts per device (`--split-blocks 21,27`). `None` = balance
    /// the dense bytes evenly.
    pub blocks: Option<Vec<usize>>,
    /// Where the LM head lives (`--head-device`): `Auto` (the default) resolves to
    /// the profile's compute-role device, falling back to the device that owns the
    /// last block; an ordinal into `devices`; or a PCI address. See [`crate::head`].
    pub head_device: HeadSpec,
    /// The hardware profile the `Auto` rule reads. `None` = the default path
    /// (`breccia_profile::default_profile_path()`), so a machine that ran
    /// `breccia profile` gets the measured roles and one that never did keeps the
    /// historical placement.
    pub profile: Option<std::path::PathBuf>,
    /// Where the routed-expert arena comes from: nothing, an explicit block list
    /// (`--gpu-experts 0,1,…`, each on the device that owns the block), or the
    /// measured-free-VRAM rule (`--gpu-experts auto`, [`crate::arena`]).
    pub arena: ArenaSpec,
    pub verbose: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions {
            devices: Vec::new(),
            ctx_capacity: 256,
            blocks: None,
            head_device: HeadSpec::Auto,
            profile: None,
            arena: ArenaSpec::Off,
            verbose: true,
        }
    }
}

impl Model {
    /// Open the shards, pick devices, plan the placement, upload the dense set.
    pub fn load(model_glob: &str, opts: &LoadOptions) -> Result<Model> {
        let t_open = std::time::Instant::now();
        // A glob (`…-0000?-of-00002.gguf`) or a single shard path: both work.
        let shards = Arc::new(
            if model_glob.contains('*') || model_glob.contains('?') {
                ShardSet::open_glob(model_glob)
            } else {
                ShardSet::open_family(model_glob)
            }
            .with_context(|| format!("opening the shard family {model_glob}"))?,
        );
        let layout = shards.layout()?;
        let emb = layout
            .embedding
            .as_ref()
            .ok_or_else(|| anyhow!("no `token_embd.weight` in the shard set"))?;
        let cfg = ModelConfig::from_metadata(&shards.metadata()?, emb.ne1() as usize)?;
        let open_seconds = t_open.elapsed().as_secs_f64();

        // --- per-block dense bytes (drives the placement split) --------------
        let mut layer_bytes = vec![0u64; cfg.n_layers];
        for t in shards.tensors() {
            if let Some((block, _)) = breccia_format::layout::split_block_name(t.name()) {
                if block >= cfg.n_layers {
                    continue;
                }
                if breccia_format::layout::classify(t.name()) == Group::Dense {
                    layer_bytes[block] += t.n_bytes();
                }
            }
        }
        let tail_bytes = layout.output.as_ref().map_or(0, |t| t.n_bytes())
            + shards
                .tensor("output_hc_norm.weight")
                .map_or(0, |t| t.n_bytes())
            + shards
                .tensor("output_hc_down.weight")
                .map_or(0, |t| t.n_bytes())
            + shards
                .tensor("output_hc_up.weight")
                .map_or(0, |t| t.n_bytes());

        // --- devices ---------------------------------------------------------
        let specs = if opts.devices.is_empty() {
            vec![placement::DeviceSpec::first_discrete()]
        } else {
            opts.devices
                .iter()
                .map(|d| placement::DeviceSpec::parse(d))
                .collect()
        };
        let placement = match &opts.blocks {
            Some(counts) => {
                if counts.len() != specs.len() {
                    bail!(
                        "--split-blocks lists {} devices but {} were requested",
                        counts.len(),
                        specs.len()
                    );
                }
                Placement::explicit(specs, &layer_bytes, tail_bytes, counts)?
            }
            None => Placement::balanced(specs, &layer_bytes, tail_bytes),
        };
        let mut devices = Vec::with_capacity(placement.devices.len());
        for spec in &placement.devices {
            devices.push(
                Device::open(spec.selector.as_deref())
                    .with_context(|| format!("opening device {}", spec.label))?,
            );
        }
        let caps: Vec<u64> = devices.iter().map(|d| vram_bytes(d.gpu())).collect();
        co_tenant_check(&devices, opts.verbose)?;

        // --- head device (`--head-device`) -----------------------------------
        // The head is 416 MiB and its GEMV is the second-largest single item of a
        // token; where it runs is a placement decision, not a consequence of the
        // last block's. `Auto` reads the profile's compute role (see `crate::head`),
        // an ordinal indexes `--devices`, and a PCI address indexes identity.
        let last_dev = placement.device_of(cfg.n_layers - 1);
        let candidates = head_candidates(&devices, opts.profile.as_deref(), opts.verbose);
        let mut head_choice = head::resolve(&opts.head_device, &candidates, last_dev)?;
        // **The device-crossing hand-off is removed, not disabled.** It produced a
        // wrong greedy sequence in 2 of 16 exclusive processes (`scripts/
        // head-handoff-race.sh`), the residual copy itself was proven byte-exact, and
        // the perturbation was never localised — so the arm is not shippable and the
        // head may only run where block 47 runs (docs/phase-4/01-forward-pass.md §8aw).
        //
        // An *explicit* crossing request is an error (the operator asked for something
        // the engine will not do). The `auto` rule is not: it reports what the profile
        // would prefer and then keeps the head where it must be, because a default
        // that refuses to load a legal device list would be worse than a default that
        // cannot honour the role.
        if opts.head_device == HeadSpec::Auto {
            if head_choice.device != last_dev {
                head_choice.reason = format!(
                    "{} — but the head may only run where block {} runs (the crossing \
                     hand-off was removed), so it stays there; list the card you want \
                     last in `--devices` to give it the last block",
                    head_choice.reason,
                    cfg.n_layers - 1
                );
                head_choice.device = last_dev;
            }
        } else {
            head::gate(head_choice.device, last_dev, &candidates)?;
        }
        let head_dev = head_choice.device;
        if opts.verbose {
            eprintln!(
                "breccia: head: device {head_dev} ({}) — {}",
                devices[head_dev].info.name, head_choice.reason
            );
        }
        // The capacity check must charge the tail (head + final mixer) to the device
        // that will actually hold it, not to the last block's device.
        let mut planned = placement.bytes.clone();
        if head_dev != last_dev {
            planned[last_dev] = planned[last_dev].saturating_sub(tail_bytes);
            planned[head_dev] += tail_bytes;
        }
        placement.check_capacity_of(&planned, &caps)?;

        // --- weights ---------------------------------------------------------
        let mut per_device_upload = vec![0f64; devices.len()];
        let mut keys = BTreeSet::new();
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for block in 0..cfg.n_layers {
            let dev = placement.device_of(block);
            let gpu = devices[dev].gpu();
            let t0 = std::time::Instant::now();
            let kind = cfg.layer_kind(block);
            let hc_attn = hc_weights(gpu, &shards, block, "attn")?;
            let hc_ffn = hc_weights(gpu, &shards, block, "ffn")?;
            let moe = MoeW {
                router: w(gpu, &shards, &format!("blk.{block}.ffn_gate_inp.weight"))?,
                shexp_gate: w(
                    gpu,
                    &shards,
                    &format!("blk.{block}.ffn_gate_inp_shexp.weight"),
                )?,
                gate: w(gpu, &shards, &format!("blk.{block}.ffn_gate_shexp.weight"))?,
                up: w(gpu, &shards, &format!("blk.{block}.ffn_up_shexp.weight"))?,
                down: w(gpu, &shards, &format!("blk.{block}.ffn_down_shexp.weight"))?,
            };
            let gdn = match kind {
                LayerKind::Gdn => Some(GdnW {
                    qkv: w(gpu, &shards, &format!("blk.{block}.attn_qkv.weight"))?,
                    z: w(gpu, &shards, &format!("blk.{block}.attn_gate.weight"))?,
                    beta: w(gpu, &shards, &format!("blk.{block}.ssm_beta.weight"))?,
                    alpha: w(gpu, &shards, &format!("blk.{block}.ssm_alpha.weight"))?,
                    beta_alpha: w_concat(
                        gpu,
                        &shards,
                        &[
                            &format!("blk.{block}.ssm_beta.weight"),
                            &format!("blk.{block}.ssm_alpha.weight"),
                        ],
                    )?,
                    out: w(gpu, &shards, &format!("blk.{block}.ssm_out.weight"))?,
                    conv: raw(gpu, &shards, &format!("blk.{block}.ssm_conv1d.weight"))?,
                    dt: raw(gpu, &shards, &format!("blk.{block}.ssm_dt.bias"))?,
                    a: raw(gpu, &shards, &format!("blk.{block}.ssm_a"))?,
                    ssm_norm: raw(gpu, &shards, &format!("blk.{block}.ssm_norm.weight"))?,
                }),
                LayerKind::Qsa => None,
            };
            let qsa = match kind {
                LayerKind::Qsa => Some(QsaW {
                    q: w(gpu, &shards, &format!("blk.{block}.attn_q.weight"))?,
                    k: w(gpu, &shards, &format!("blk.{block}.attn_k.weight"))?,
                    v: w(gpu, &shards, &format!("blk.{block}.attn_v.weight"))?,
                    idx_q: w(gpu, &shards, &format!("blk.{block}.indexer.q_proj.weight"))?,
                    idx_k: w(gpu, &shards, &format!("blk.{block}.indexer.k_proj.weight"))?,
                    out: w(gpu, &shards, &format!("blk.{block}.attn_output.weight"))?,
                    q_norm: raw(gpu, &shards, &format!("blk.{block}.attn_q_norm.weight"))?,
                    k_norm: raw(gpu, &shards, &format!("blk.{block}.attn_k_norm.weight"))?,
                    idx_q_norm: raw(gpu, &shards, &format!("blk.{block}.indexer.q_norm.weight"))?,
                    idx_k_norm: raw(gpu, &shards, &format!("blk.{block}.indexer.k_norm.weight"))?,
                }),
                LayerKind::Gdn => None,
            };
            let expert_block = layout
                .block(block)
                .ok_or_else(|| anyhow!("block {block} has no routed-expert tensors"))?;
            let experts = LayerPlan::from_block(&shards, expert_block)?;
            let bytes = hc_attn.bytes()
                + hc_ffn.bytes()
                + [&moe.router, &moe.shexp_gate, &moe.gate, &moe.up, &moe.down]
                    .iter()
                    .map(|w| w.buf.size)
                    .sum::<u64>()
                + gdn.as_ref().map_or(0, |g| {
                    [&g.qkv, &g.z, &g.beta, &g.alpha, &g.out]
                        .iter()
                        .map(|w| w.buf.size)
                        .sum::<u64>()
                        + g.conv.size
                        + g.dt.size
                        + g.a.size
                        + g.ssm_norm.size
                })
                + qsa.as_ref().map_or(0, |q| {
                    [&q.q, &q.k, &q.v, &q.idx_q, &q.idx_k, &q.out]
                        .iter()
                        .map(|w| w.buf.size)
                        .sum::<u64>()
                        + q.q_norm.size
                        + q.k_norm.size
                        + q.idx_q_norm.size
                        + q.idx_k_norm.size
                });
            for w in [
                &moe.router,
                &moe.shexp_gate,
                &moe.gate,
                &moe.up,
                &moe.down,
                &hc_attn.down,
                &hc_attn.up,
                &hc_ffn.down,
                &hc_ffn.up,
            ] {
                keys.insert(w.key);
            }
            if let Some(g) = &gdn {
                for w in [&g.qkv, &g.z, &g.beta, &g.alpha, &g.out, &g.beta_alpha] {
                    keys.insert(w.key);
                }
            }
            if let Some(q) = &qsa {
                for w in [&q.q, &q.k, &q.v, &q.idx_q, &q.idx_k, &q.out] {
                    keys.insert(w.key);
                }
            }
            per_device_upload[dev] += t0.elapsed().as_secs_f64();
            layers.push(LayerWeights {
                block,
                kind,
                device: dev,
                hc_attn,
                hc_ffn,
                moe,
                gdn,
                qsa,
                experts,
                bytes,
            });
        }

        // --- head + final mixer ---------------------------------------------
        let gpu = devices[head_dev].gpu();
        let t0 = std::time::Instant::now();
        let head = w(gpu, &shards, "output.weight")?;
        let out_hc_norm = raw(gpu, &shards, "output_hc_norm.weight")?;
        let out_hc_down = w(gpu, &shards, "output_hc_down.weight")?;
        let out_hc_up = w(gpu, &shards, "output_hc_up.weight")?;
        keys.insert(head.key);
        keys.insert(out_hc_down.key);
        keys.insert(out_hc_up.key);
        per_device_upload[head_dev] += t0.elapsed().as_secs_f64();

        // --- PLE module (block 1 only) ---------------------------------------
        let ple = if cfg.ple.layers.is_empty() {
            None
        } else {
            let b = cfg.ple.layers[0];
            let dev = placement.device_of(b);
            let gpu = devices[dev].gpu();
            let t0 = std::time::Instant::now();
            let key = w(gpu, &shards, &format!("blk.{b}.ple_key.weight"))?;
            let value = w(gpu, &shards, &format!("blk.{b}.ple_value.weight"))?;
            keys.insert(key.key);
            keys.insert(value.key);
            let pr = PleRaw {
                norm_key: raw_bytes(&shards, &format!("blk.{b}.ple_norm_key.weight"))?,
                norm_query: raw_bytes(&shards, &format!("blk.{b}.ple_norm_query.weight"))?,
                norm_conv: raw_bytes(&shards, &format!("blk.{b}.ple_norm_conv.weight"))?,
                conv: raw_bytes(&shards, &format!("blk.{b}.ple_conv1d.weight"))?,
                key,
                value,
            };
            per_device_upload[dev] += t0.elapsed().as_secs_f64();
            Some(pr)
        };

        // --- routed experts that live in VRAM (`-ot`) -------------------------
        //
        // Two sources: an explicit block list (the hand-tuned override, kept
        // working verbatim) or the measured-free-VRAM rule (crate::arena). Both
        // end in the same upload loop, so the resident path is identical.
        //
        // Arena size per block, measured from the three role tensors' real bytes
        // (they differ by quant type: 637 MiB IQ2_XXS … 1137 MiB IQ3_S here).
        // `RolePlan::bytes()` is **one expert**; the arena holds all `n_experts`.
        let arena_sizes: Vec<u64> = layers
            .iter()
            .map(|l| {
                let e = &l.experts;
                let per = e.gate.bytes() as u64 + e.up.bytes() as u64 + e.down.bytes() as u64;
                per * e.n_experts as u64
            })
            .collect();
        let candidates: Vec<ArenaBlock> = (0..cfg.n_layers)
            .map(|b| ArenaBlock {
                block: b,
                device: placement.device_of(b),
                bytes: arena_sizes[b],
            })
            .collect();
        let (arena_choice, arena_report) = match &opts.arena {
            ArenaSpec::Off => (arena::ArenaChoice::default(), None),
            ArenaSpec::List(list) => {
                let mut blocks = Vec::with_capacity(list.len());
                for &b in list {
                    if b >= cfg.n_layers {
                        bail!("--gpu-experts lists block {b} of {}", cfg.n_layers);
                    }
                    if blocks.iter().any(|x: &ArenaBlock| x.block == b) {
                        continue;
                    }
                    blocks.push(candidates[b]);
                }
                blocks.sort_by_key(|x| x.block);
                (
                    arena::ArenaChoice {
                        blocks,
                        ..Default::default()
                    },
                    None,
                )
            }
            ArenaSpec::Auto(aopts) => {
                let budgets = probe_budgets(&devices, *aopts, opts.verbose)?;
                let choice = arena::select(&candidates, &budgets);
                let report = arena_report_of(&choice, &devices);
                (choice, Some(report))
            }
        };
        let mut placed = std::collections::BTreeMap::new();
        // A block whose allocation fails is *dropped* (it stays on the host lane,
        // which is the default path) instead of aborting the load — the rule's
        // arithmetic should prevent it, and this is the belt to that brace.
        let mut dropped: Vec<usize> = Vec::new();
        for ab in &arena_choice.blocks {
            let b = ab.block;
            let dev = ab.device;
            let gpu = devices[dev].gpu();
            let t0 = std::time::Instant::now();
            let got: Result<(W, W, W)> = (|| {
                let gate = w(gpu, &shards, &format!("blk.{b}.ffn_gate_exps.weight"))?;
                let up = w(gpu, &shards, &format!("blk.{b}.ffn_up_exps.weight"))?;
                let down = w(gpu, &shards, &format!("blk.{b}.ffn_down_exps.weight"))?;
                Ok((gate, up, down))
            })();
            match got {
                Ok((gate, up, down)) => {
                    keys.insert(gate.key);
                    keys.insert(up.key);
                    keys.insert(down.key);
                    per_device_upload[dev] += t0.elapsed().as_secs_f64();
                    placed.insert(
                        b,
                        PlacedExperts {
                            block: b,
                            device: dev,
                            gate,
                            up,
                            down,
                        },
                    );
                }
                Err(e) => {
                    eprintln!("breccia: arena: blk.{b} dropped, staying on the host lane: {e:#}");
                    dropped.push(b);
                }
            }
        }
        let mut arena_report = arena_report;
        if let Some(r) = arena_report.as_mut() {
            for b in &dropped {
                if let Some(i) = r.blocks.iter().position(|x| x == b) {
                    r.blocks.remove(i);
                }
                r.skipped
                    .push(format!("{b}: {}", arena::Skip::UploadFailed.as_str()));
            }
            r.blocks.sort_unstable();
            r.bytes = r.blocks.iter().map(|&b| arena_sizes[b]).sum();
            for row in r.devices.iter_mut() {
                let mine: Vec<usize> = r
                    .blocks
                    .iter()
                    .copied()
                    .filter(|&b| placement.device_of(b) == row.device)
                    .collect();
                row.chosen = mine.len();
                row.bytes = mine.iter().map(|&b| arena_sizes[b]).sum();
                row.left_bytes = row.budget_bytes.saturating_sub(row.bytes);
            }
        }

        let upload_seconds: f64 = per_device_upload.iter().sum();
        let dense_bytes: u64 = layers.iter().map(|l| l.bytes).sum::<u64>()
            + out_hc_norm.size
            + out_hc_down.buf.size
            + out_hc_up.buf.size
            + ple
                .as_ref()
                .map_or(0, |p| p.key.buf.size + p.value.buf.size);

        // --- the plan, printed before anything runs --------------------------
        let mut plan = Vec::new();
        let mut dev_loads = Vec::new();
        let mut state_total = 0u64;
        for (i, dev) in devices.iter().enumerate() {
            let run = placement
                .runs
                .iter()
                .find(|r| r.device == i)
                .map(|r| r.blocks)
                .unwrap_or((0, 0));
            let st = placement::state_bytes(&cfg, &placement.layer_device, i, opts.ctx_capacity);
            state_total += st;
            let has_head = head_dev == i;
            let mutable = layers
                .iter()
                .filter(|l| l.device == i)
                .map(|l| l.bytes)
                .sum::<u64>()
                + if has_head {
                    head.buf.size + out_hc_norm.size + out_hc_down.buf.size + out_hc_up.buf.size
                } else {
                    0
                }
                + if ple
                    .as_ref()
                    .is_some_and(|_| placement.device_of(cfg.ple.layers[0]) == i)
                {
                    ple.as_ref()
                        .map_or(0, |p| p.key.buf.size + p.value.buf.size)
                } else {
                    0
                };
            plan.push(format!(
                "device {i}  {:<46} blocks {:>2}..{:<2}  weights {:>6} MiB  state {:>5} MiB  \
                 of {:>6} MiB{}",
                dev.info.name,
                run.0,
                run.1,
                mutable >> 20,
                st >> 20,
                caps[i] >> 20,
                if has_head {
                    if head_dev == last_dev {
                        "  [+ head + final mixer]"
                    } else {
                        "  [+ head + final mixer, moved by --head-device]"
                    }
                } else {
                    ""
                }
            ));
            dev_loads.push(DeviceLoad {
                ordinal: i,
                label: placement.devices[i].label.clone(),
                name: dev.info.name.clone(),
                pci: dev.info.pci_address.clone(),
                driver: dev.info.driver.clone(),
                blocks: run,
                dense_bytes: mutable,
                vram_total: caps[i],
                upload_seconds: per_device_upload[i],
            });
        }
        for (dev, pe) in &placed {
            let _ = dev;
            plan.push(format!(
                "expert  blk.{:<2} -> device {}  arena {:>5} MiB (512 experts, {})",
                pe.block,
                pe.device,
                pe.bytes() >> 20,
                pe.gate.key.ty.name()
            ));
        }
        // The arena decision, as a rule with numbers rather than a block list: one
        // line per device (free → reserve → budget → spent) plus the chosen set.
        if let Some(r) = &arena_report {
            for row in &r.devices {
                plan.push(format!(
                    "arena   device {} {}  free {:>6} MiB - {} {:>4} MiB = {:>6} MiB -> {} blocks, \
                     {:>6} MiB  ({:>4} MiB left)  [{}]",
                    row.device,
                    if row.display { "display" } else { "headless" },
                    row.free_bytes >> 20,
                    if row.display { "display reserve" } else { "headroom" },
                    row.reserve_bytes >> 20,
                    row.budget_bytes >> 20,
                    row.chosen,
                    row.bytes >> 20,
                    row.left_bytes >> 20,
                    row.source,
                ));
            }
            plan.push(format!(
                "arena   rule=auto: {} of {} blocks resident ({:.1} GiB); the host lane keeps {}",
                r.blocks.len(),
                cfg.n_layers,
                r.bytes as f64 / (1u64 << 30) as f64,
                cfg.n_layers - r.blocks.len()
            ));
            plan.push(format!(
                "arena   blocks {}",
                r.blocks
                    .iter()
                    .map(|b| b.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ));
            if !r.skipped.is_empty() {
                plan.push(format!("arena   not resident: {}", r.skipped.join(", ")));
            }
        } else if !placed.is_empty() {
            plan.push(format!(
                "arena   rule=list (--gpu-experts): {} of {} blocks resident",
                placed.len(),
                cfg.n_layers
            ));
        }
        plan.push(format!(
            "host    experts {} MiB mmap (demand-paged, top-{} of {} per block) + PLE {} MiB lazy \
             + token_embd {} MiB staged",
            layout.expert_bytes() >> 20,
            cfg.n_expert_used,
            cfg.n_expert,
            layout.group_bytes(Group::Ple) >> 20,
            layout.group_bytes(Group::Embedding) >> 20
        ));

        let (rss, rss_peak) = rss_bytes();
        let stats = LoadStats {
            open_seconds,
            upload_seconds,
            dense_bytes,
            head_bytes: head.buf.size,
            expert_bytes_mapped: layout.expert_bytes(),
            ple_bytes_mapped: layout.group_bytes(Group::Ple),
            rss_bytes: rss,
            rss_peak_bytes: rss_peak,
            devices: dev_loads,
            state_bytes: state_total,
            plan,
        };

        // Silence the "unused" warning on the name list; it is kept for a future
        // per-tensor placement report (`-ot` parity, M3).

        Ok(Model {
            cfg,
            shards,
            placement,
            devices,
            layers,
            head_device: head_dev,
            head,
            out_hc_norm,
            out_hc_down,
            out_hc_up,
            ple,
            placed,
            arena: arena_report,
            stats,
            keys,
        })
    }

    /// The device a block runs on.
    pub fn device_of(&self, block: usize) -> usize {
        self.placement.device_of(block)
    }

    /// The device the LM head and the final mixer live on (`--head-device`, else
    /// the device that owns the last block).
    pub fn head_device(&self) -> usize {
        self.head_device
    }

    /// The output width (`token_embd.ne1`).
    pub fn vocab(&self) -> usize {
        self.cfg.vocab
    }

    /// Raw bytes of the `token_embd` table (staged, one row per token).
    pub fn token_embd(&self) -> Result<&[u8]> {
        let t = self
            .shards
            .tensor("token_embd.weight")
            .ok_or_else(|| anyhow!("no `token_embd.weight`"))?;
        Ok(t.as_slice())
    }

    /// The PLE table (`[dim, rows]`, IQ4_NL) — 28.8 GiB of demand-paged mapping.
    pub fn ple_table(&self) -> Result<&[u8]> {
        let t = self
            .shards
            .tensor("per_layer_token_embd.weight")
            .ok_or_else(|| anyhow!("no `per_layer_token_embd.weight`"))?;
        Ok(t.as_slice())
    }

    /// Bytes of one PLE table row (160 values of IQ4_NL = 5 blocks × 18 B).
    pub fn ple_row_bytes(&self) -> usize {
        (self.cfg.ple.dim / 32) * 18
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        // Free every buffer on the device that owns it, while the devices live.
        for l in &self.layers {
            l.release(self.devices[l.device].gpu());
        }
        let head_dev = self.head_device;
        {
            let gpu = self.devices[head_dev].gpu();
            gpu.destroy_buf(&self.head.buf);
            gpu.destroy_buf(&self.out_hc_norm);
            for w in [&self.out_hc_down, &self.out_hc_up] {
                gpu.destroy_buf(&w.buf);
            }
        }
        for pe in self.placed.values() {
            let gpu = self.devices[pe.device].gpu();
            for w in [&pe.gate, &pe.up, &pe.down] {
                gpu.destroy_buf(&w.buf);
            }
        }
        if let Some(p) = &self.ple {
            let dev = self.placement.device_of(self.cfg.ple.layers[0]);
            let gpu = self.devices[dev].gpu();
            gpu.destroy_buf(&p.key.buf);
            gpu.destroy_buf(&p.value.buf);
        }
    }
}

/// Every compiled projection pipeline of one device, keyed by shape.
pub struct PipelineSet<'g> {
    map: std::collections::BTreeMap<PipeKey, Gemv<'g>>,
    batch: std::collections::BTreeMap<PipeKey, breccia_kernels::GemvBatch<'g>>,
    /// The **multi-user** GEMM of the chunked-prefill merge (§8v), one per quant
    /// type (the kernel's shape rides in the push constants, so a type needs one
    /// pipeline however many roles use it).
    gemm:
        std::collections::BTreeMap<breccia_kernels::QuantType, breccia_kernels::gemm::GemmMany<'g>>,
}

impl<'g> PipelineSet<'g> {
    /// Compile one pipeline per distinct `(type, rows, cols)`.
    pub fn build(gpu: &'g Gpu, keys: &BTreeSet<PipeKey>) -> Result<Self> {
        let mut map = std::collections::BTreeMap::new();
        for k in keys {
            let g = Gemv::new(gpu, k.ty, k.rows, k.cols).with_context(|| {
                format!(
                    "compiling the {} {}x{} GEMV pipeline",
                    k.ty.name(),
                    k.rows,
                    k.cols
                )
            })?;
            map.insert(*k, g);
        }
        // The batched form is only needed for expert roles, and only for the
        // blocks that are resident; the caller registers those keys, so compile
        // a batch for every key that is one of the model's expert shapes.
        let mut batch = std::collections::BTreeMap::new();
        for k in keys {
            let is_expert_role =
                (k.rows == 640 && k.cols == 2560) || (k.rows == 2560 && k.cols == 640);
            if !is_expert_role {
                continue;
            }
            batch.insert(
                *k,
                breccia_kernels::GemvBatch::new(gpu, k.ty, k.rows, k.cols).with_context(|| {
                    format!(
                        "compiling the batched {} {}x{} expert pipeline",
                        k.ty.name(),
                        k.rows,
                        k.cols
                    )
                })?,
            );
        }
        // The multi-user GEMM is only needed for the resident blocks' expert roles
        // (the merged prefill pass, §8v). A type with no shader is left out and
        // the caller falls back to the per-token path for those blocks.
        let mut gemm = std::collections::BTreeMap::new();
        for k in keys {
            let is_expert_role =
                (k.rows == 640 && k.cols == 2560) || (k.rows == 2560 && k.cols == 640);
            if !is_expert_role || gemm.contains_key(&k.ty) {
                continue;
            }
            match breccia_kernels::gemm::GemmMany::new(gpu, k.ty) {
                Ok(g) => {
                    gemm.insert(k.ty, g);
                }
                Err(e) => {
                    eprintln!(
                        "merged prefill: no multi-user GEMM for {} ({e}); those blocks keep \
                         the per-token path",
                        k.ty.name()
                    );
                }
            }
        }
        Ok(PipelineSet { map, batch, gemm })
    }

    /// The multi-user GEMM for `ty`, or `None` when the shader set has no GEMM
    /// for it (the caller then keeps the per-token path for that block, §8v).
    pub fn gemm(
        &self,
        ty: breccia_kernels::QuantType,
    ) -> Option<&breccia_kernels::gemm::GemmMany<'g>> {
        self.gemm.get(&ty)
    }

    /// The batched (multi-expert) pipeline for an expert role shape.
    pub fn batch(&self, k: &PipeKey) -> Result<&breccia_kernels::GemvBatch<'g>> {
        self.batch.get(k).ok_or_else(|| {
            anyhow!(
                "no batched expert pipeline for {} {}x{}",
                k.ty.name(),
                k.rows,
                k.cols
            )
        })
    }

    pub fn get(&self, k: &PipeKey) -> Result<&Gemv<'g>> {
        self.map.get(k).ok_or_else(|| {
            anyhow!(
                "no pipeline for {} {}x{} (the load did not discover it)",
                k.ty.name(),
                k.rows,
                k.cols
            )
        })
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Weight bytes of every pipeline's shape (one dispatch's read).
    pub fn shape_bytes(&self) -> u64 {
        self.map.keys().map(|k| k.weight_bytes()).sum()
    }
}

fn w(gpu: &Gpu, shards: &ShardSet, name: &str) -> Result<W> {
    let t = shards
        .tensor(name)
        .ok_or_else(|| anyhow!("missing tensor `{name}`"))?;
    let key = PipeKey::of(&t)?;
    let buf = upload_tensor(gpu, name, &t)?;
    Ok(W { buf, key })
}

/// Several tensors of the **same quant type and column count**, concatenated along
/// rows into one allocation (§8ay).
///
/// This is what lets one GEMV cover N small matrices that share an input vector: a
/// GEMV output row is an independent dot product over that row's weights, so
/// concatenating along rows changes nothing about any row's arithmetic — the merged
/// dispatch is **bit-identical** to the N separate ones, and it replaces N
/// dependent launches (each of which is 6-80 workgroups of work) with one.
fn w_concat(gpu: &Gpu, shards: &ShardSet, names: &[&str]) -> Result<W> {
    let mut tensors = Vec::with_capacity(names.len());
    let mut key: Option<PipeKey> = None;
    let mut total = 0u64;
    for name in names {
        let t = shards
            .tensor(name)
            .ok_or_else(|| anyhow!("missing tensor `{name}`"))?;
        let k = PipeKey::of(&t)?;
        match &key {
            None => key = Some(k),
            Some(prev) if *prev != k => {
                anyhow::bail!(
                    "cannot concatenate {names:?}: `{name}` is a different quant type/layout"
                )
            }
            _ => {}
        }
        total += t.n_bytes();
        tensors.push(t);
    }
    let mut bytes = Vec::with_capacity(total as usize);
    for t in &tensors {
        bytes.extend_from_slice(t.as_slice());
    }
    let buf = breccia_kernels::gemv::upload(gpu, &bytes)
        .with_context(|| format!("uploading the concatenation of {names:?} ({total} B)"))?;
    // The pipeline key must describe the **concatenated** shape: same type and
    // columns as the parts, `rows` summed. Taking the key from the first tensor
    // would build a 48-row pipeline that silently computes only the first half.
    let mut key = key.expect("at least one tensor");
    key.rows = tensors.iter().map(|t| t.ne1() as usize).sum();
    Ok(W { buf, key })
}

fn raw(gpu: &Gpu, shards: &ShardSet, name: &str) -> Result<Buf> {
    let t = shards
        .tensor(name)
        .ok_or_else(|| anyhow!("missing tensor `{name}`"))?;
    upload_tensor(gpu, name, &t)
}

/// The number of leading tensors to truncate by `BRECCIA_FAULT_TRUNCATE_KB`
/// (fault injection for the short-read class, see [`check_uploaded`]). `0` = off.
fn fault_truncate_kb() -> Option<u64> {
    std::env::var("BRECCIA_FAULT_TRUNCATE_KB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|k| *k > 0)
}

/// Upload a whole tensor and **check the device got all of it**.
///
/// The model path has no `read()` to short-return (`breccia_format` maps the shards
/// with `memmap2` and hands out `&[u8]` subslices), so a partial upload can only
/// arise from a caller passing a truncated slice or a copy whose length disagrees
/// with the tensor. Both are silent in release — the buffer keeps whatever the
/// allocator left there and the engine produces plausible-but-wrong tokens — so the
/// declared byte length is checked here, once per tensor, at load.
///
/// `BRECCIA_FAULT_TRUNCATE_KB=k` truncates the first `upload_tensor` call by `k` KiB
/// to prove the check fires (it is the deterministic test for this class: without
/// the check the same injection loads and generates wrong tokens).
fn upload_tensor(gpu: &Gpu, name: &str, t: &breccia_format::gguf::TensorView<'_>) -> Result<Buf> {
    static FAULTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let declared = t.n_bytes();
    let mut bytes = t.as_slice();
    if let Some(kb) = fault_truncate_kb() {
        if !FAULTED.swap(true, std::sync::atomic::Ordering::SeqCst) {
            let cut = ((kb << 10) as usize).min(bytes.len().saturating_sub(1));
            bytes = &bytes[..bytes.len() - cut];
            eprintln!(
                "breccia: FAULT INJECTION: uploading {name} with {} of {declared} bytes",
                bytes.len()
            );
        }
    }
    let buf = breccia_kernels::gemv::upload(gpu, bytes)
        .with_context(|| format!("uploading {name} ({declared} B)"))?;
    check_uploaded(name, declared, bytes.len() as u64, buf.size)?;
    Ok(buf)
}

/// Fail when a tensor is uploaded with fewer bytes than the artifact declares, or when
/// the device buffer it landed in cannot hold the whole tensor.
///
/// This is the release-mode guard for the short/partial-upload class: it is *not* a
/// `debug_assert`, because the symptom it prevents (wrong tokens at ~10 % of
/// processes) is exactly what a debug build is not running when it matters. The
/// source length is what matters — `alloc` rounds a buffer up (40960 declares as a
/// 41024-byte `Buf` here), so a padded buffer is fine and a short *source* is not.
fn check_uploaded(name: &str, declared: u64, source: u64, buf: u64) -> Result<()> {
    if source != declared {
        bail!(
            "tensor `{name}` was uploaded with {source} bytes of {declared} ({} bytes short): a \
             truncated upload would silently produce wrong tokens, so the load is refused",
            declared.saturating_sub(source)
        );
    }
    if buf < declared {
        bail!(
            "tensor `{name}` landed in a {buf}-byte buffer but declares {declared}: the upload \
             could not have carried the whole tensor, so the load is refused"
        );
    }
    Ok(())
}

fn raw_bytes(shards: &ShardSet, name: &str) -> Result<Vec<u8>> {
    let t = shards
        .tensor(name)
        .ok_or_else(|| anyhow!("missing tensor `{name}`"))?;
    Ok(t.as_slice().to_vec())
}

fn hc_weights(gpu: &Gpu, shards: &ShardSet, block: usize, half: &str) -> Result<HcW> {
    Ok(HcW {
        norm: raw(gpu, shards, &format!("blk.{block}.hc_{half}_norm.weight"))?,
        down: w(gpu, shards, &format!("blk.{block}.hc_{half}_down.weight"))?,
        up: w(gpu, shards, &format!("blk.{block}.hc_{half}_up.weight"))?,
        inject: Some(raw(
            gpu,
            shards,
            &format!("blk.{block}.hc_{half}_inject.weight"),
        )?),
    })
}

/// The devices of this placement, annotated with whatever the hardware profile
/// measured about them.
///
/// Matching is by **PCI address**, never by enumeration ordinal: the profile's
/// index is its own backend's, and this engine's device order is whatever
/// `--devices` said. A card the profile does not cover keeps `role: None`, which
/// makes the `auto` rule fall back to the historical answer instead of guessing.
fn head_candidates(
    devices: &[Device],
    profile: Option<&std::path::Path>,
    verbose: bool,
) -> Vec<HeadCandidate> {
    let path = match profile {
        Some(p) => Some(p.to_path_buf()),
        None => Some(breccia_profile::default_profile_path()),
    };
    let loaded = path.as_deref().and_then(|p| {
        breccia_profile::load_profile(p).unwrap_or_else(|e| {
            if verbose {
                eprintln!("breccia: head: ignoring {} ({e:#})", p.display());
            }
            None
        })
    });
    devices
        .iter()
        .map(|d| {
            let want = breccia_vk::pci::normalize_address(&d.info.pci_address);
            let entry = loaded.as_ref().and_then(|p| {
                p.devices.iter().find(|e| {
                    !e.identity.pci_address.is_empty()
                        && breccia_vk::pci::normalize_address(&e.identity.pci_address) == want
                })
            });
            if verbose && entry.is_none() && loaded.is_some() {
                eprintln!(
                    "breccia: head: {} ({}) is not in the hardware profile",
                    d.info.name, d.info.pci_address
                );
            }
            HeadCandidate {
                pci: d.info.pci_address.clone(),
                name: d.info.name.clone(),
                role: entry.map(|e| e.role),
                fp32_gflops: entry
                    .and_then(|e| e.measurements.compute.as_ref())
                    .map(|c| c.fp32_gflops),
            }
        })
        .collect()
}

/// Refuse (or warn about) a card another process is already using.
///
/// Every wrong greedy sequence this engine has produced in a *benchmark* window
/// (5 of them, all before `scripts/head-handoff-race.sh` grew its guard) happened
/// while a second GPU process shared the cards and the same 40 GB of expert mapping;
/// in ~60 exclusive processes the rate is zero, and two arena choices are
/// numerically interchangeable (max |Δlogit| 7.4e-5), so the remaining explanation
/// for a *persistent* wrong token is the one condition that is not ours: another
/// client holding VRAM (which makes RADV evict device-local pages under a running
/// kernel) and thrashing the page cache the lane reads through.
///
/// `VK_EXT_memory_budget` answers it exactly: `heapBudget` is what *this* process may
/// allocate, so a budget more than `SLACK` below the heap size means somebody else
/// holds the difference. The default is a loud warning (a shared box is a legitimate
/// way to run the engine); `BRECCIA_REQUIRE_QUIET=1` makes it fatal.
fn co_tenant_check(devices: &[Device], verbose: bool) -> Result<()> {
    const SLACK: u64 = 512 << 20;
    let strict = std::env::var_os("BRECCIA_REQUIRE_QUIET").is_some();
    for (i, dev) in devices.iter().enumerate() {
        let Some(b) = dev.gpu().mem_budget() else {
            continue;
        };
        let held = b.heap_bytes.saturating_sub(b.budget + b.usage);
        if held > SLACK {
            let msg = format!(
                "device {i} ({}): another process holds {} MiB of its VRAM (the heap is \
                 {} MiB, this process may allocate {} MiB). Benchmarks taken now are not \
                 comparable, and a shared card is the only condition under which this \
                 engine has produced a wrong greedy sequence",
                dev.info.pci_address,
                held >> 20,
                b.heap_bytes >> 20,
                b.budget >> 20
            );
            if strict {
                bail!("{msg} (BRECCIA_REQUIRE_QUIET is set)");
            }
            if verbose {
                eprintln!("breccia: WARNING: {msg}");
            }
        }
    }
    Ok(())
}

/// Measure each device's free VRAM and turn it into an arena budget.
///
/// Called **after** the dense weights, the head and the PLE projections are
/// uploaded — those are part of the model, not the arena, and charging them to the
/// arena budget would be double counting. The source is stated per device
/// ([`BudgetSource`]): `VK_EXT_memory_budget` when advertised, amdgpu's sysfs
/// counters otherwise. When neither answers, this **fails** rather than guessing a
/// number — a wrong budget here is an `VK_ERROR_OUT_OF_DEVICE_MEMORY` at best and a
/// starved display card at worst.
fn probe_budgets(devices: &[Device], opts: ArenaOpts, verbose: bool) -> Result<Vec<DeviceBudget>> {
    let mut out = Vec::with_capacity(devices.len());
    for (i, dev) in devices.iter().enumerate() {
        let gpu = dev.gpu();
        let pci = &dev.info.pci_address;
        let (free, source) = match gpu.mem_budget() {
            Some(b) => (b.free_for_us(), BudgetSource::MemoryBudget),
            None => match breccia_vk::pci::vram_used(pci) {
                Some((total, used)) => (total.saturating_sub(used), BudgetSource::DriverSysfs),
                None => bail!(
                    "arena: device {i} ({pci}) advertises neither VK_EXT_memory_budget nor \
                     amdgpu's mem_info_vram_* — refusing to guess a VRAM budget. Pass an \
                     explicit --gpu-experts block list, or a driver that reports one"
                ),
            },
        };
        // A device that currently drives a display gets the bigger reserve; the
        // fact is read from sysfs (see `pci::drives_a_display`), never assumed.
        // `unknown` is treated as a display: the safe direction.
        let display = breccia_vk::pci::drives_a_display(pci).unwrap_or(true);
        let reserve = if display {
            opts.display_reserve
        } else {
            opts.headroom
        };
        if verbose {
            eprintln!(
                "breccia: arena: device {i} {} free {} MiB ({}) - {} {} MiB = {} MiB",
                if display {
                    "drives a display"
                } else {
                    "headless"
                },
                free >> 20,
                source.as_str(),
                if display {
                    "display reserve"
                } else {
                    "headroom"
                },
                reserve >> 20,
                free.saturating_sub(reserve) >> 20
            );
        }
        out.push(DeviceBudget {
            device: i,
            label: dev.info.pci_address.clone(),
            display,
            free,
            reserve,
            source,
        });
    }
    Ok(out)
}

/// The serialisable summary of an [`arena::ArenaChoice`], for the banner + JSON.
fn arena_report_of(choice: &arena::ArenaChoice, devices: &[Device]) -> ArenaReport {
    let mut skipped: Vec<String> = choice
        .skipped
        .iter()
        .map(|(b, why)| format!("{}: {}", b.block, why.as_str()))
        .collect();
    skipped.sort();
    ArenaReport {
        rule: "auto".to_string(),
        blocks: choice.blocks.iter().map(|b| b.block).collect(),
        bytes: choice.bytes(),
        devices: choice
            .devices
            .iter()
            .map(|s| DeviceArenaRow {
                device: s.budget.device,
                label: s.budget.label.clone(),
                name: devices[s.budget.device].info.name.clone(),
                display: s.budget.display,
                source: s.budget.source.as_str().to_string(),
                free_bytes: s.budget.free,
                reserve_bytes: s.budget.reserve,
                budget_bytes: s.budget.budget(),
                chosen: s.chosen,
                bytes: s.bytes,
                left_bytes: s.left(),
            })
            .collect(),
        skipped,
    }
}

/// The DEVICE_LOCAL heap size of a device, in bytes.
pub fn vram_bytes(gpu: &Gpu) -> u64 {
    let mut best = 0u64;
    for i in 0..gpu.mem.memory_heap_count {
        let h = gpu.mem.memory_heaps[i as usize];
        if h.flags.contains(ash::vk::MemoryHeapFlags::DEVICE_LOCAL) {
            best = best.max(h.size);
        }
    }
    best
}

/// `(current, peak)` RSS from `/proc/self/status`, in bytes.
pub fn rss_bytes() -> (u64, u64) {
    let mut cur = 0;
    let mut peak = 0;
    if let Ok(s) = std::fs::read_to_string("/proc/self/status") {
        for line in s.lines() {
            if let Some(v) = line.strip_prefix("VmRSS:") {
                cur = kb_of(v);
            } else if let Some(v) = line.strip_prefix("VmHWM:") {
                peak = kb_of(v);
            }
        }
    }
    (cur, peak)
}

fn kb_of(v: &str) -> u64 {
    v.split_whitespace()
        .next()
        .and_then(|n| n.parse::<u64>().ok())
        .map(|kb| kb * 1024)
        .unwrap_or(0)
}

/// `ExpertRole` order used by the host lane (`gate | up | down`).
pub const ROLES: [ExpertRole; 3] = [ExpertRole::Gate, ExpertRole::Up, ExpertRole::Down];

/// Fail early when a placement would not fit (called by the CLI before loading).
pub fn check_placement(p: &Placement, caps: &[u64]) -> Result<()> {
    if p.devices.is_empty() {
        bail!("the placement has no device");
    }
    p.check_capacity(caps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pipe_key_orders_by_type_then_shape() {
        let a = PipeKey {
            ty: QuantType::Iq4Xs,
            rows: 10240,
            cols: 2560,
        };
        let b = PipeKey {
            ty: QuantType::Iq4Xs,
            rows: 6144,
            cols: 2560,
        };
        assert!(b < a, "keys must be totally ordered for the BTreeMap");
        // IQ4_XS is 256/136: 10 blocks per 2560-wide row.
        assert_eq!(a.weight_bytes(), (10240 * 10 * 136) as u64);
    }

    /// The deterministic test for the short/partial-upload class: a truncated tensor
    /// must fail the load, not produce tokens. `BRECCIA_FAULT_TRUNCATE_KB` exercises
    /// the same path end-to-end (a `bench token` run with it set must exit non-zero
    /// with this message).
    #[test]
    fn a_truncated_upload_is_refused() {
        // A padded buffer (alloc rounds up) is fine; a short *source* is not.
        assert!(check_uploaded("blk.0.ffn_up_exps.weight", 1000, 1000, 1024).is_ok());
        let e = check_uploaded("blk.0.ffn_up_exps.weight", 1000, 936, 1024).unwrap_err();
        let msg = format!("{e:#}");
        assert!(msg.contains("936 bytes of 1000"), "{msg}");
        assert!(msg.contains("64 bytes short"), "{msg}");
        assert!(msg.contains("refused"), "{msg}");
        // A zero-byte upload of a non-empty tensor is the same class.
        assert!(check_uploaded("x", 4096, 0, 4096).is_err());
        // A device buffer that cannot hold the tensor is refused too.
        assert!(check_uploaded("x", 4096, 4096, 2048).is_err());
    }

    #[test]
    fn rss_is_readable() {
        let (cur, peak) = rss_bytes();
        assert!(cur > 0, "the process must have resident memory");
        assert!(peak >= cur);
    }
}
