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

use crate::config::{LayerKind, ModelConfig};
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
    pub head: W,
    pub out_hc_norm: Buf,
    pub out_hc_down: W,
    pub out_hc_up: W,
    pub ple: Option<PleRaw>,
    /// Blocks whose routed experts live in VRAM, keyed by block.
    pub placed: std::collections::BTreeMap<usize, PlacedExperts>,
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
    /// Blocks whose experts go to VRAM (`--gpu-experts 0,1,2,4,8,18,20,22,23,24,35`),
    /// each on the device that owns the block.
    pub gpu_experts: Vec<usize>,
    pub verbose: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions {
            devices: Vec::new(),
            ctx_capacity: 256,
            blocks: None,
            gpu_experts: Vec::new(),
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
        placement.check_capacity(&caps)?;

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
                for w in [&g.qkv, &g.z, &g.beta, &g.alpha, &g.out] {
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
        let head_dev = placement.device_of(cfg.n_layers - 1);
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
        let mut placed = std::collections::BTreeMap::new();
        for &b in &opts.gpu_experts {
            if b >= cfg.n_layers {
                bail!("--gpu-experts lists block {b} of {}", cfg.n_layers);
            }
            let dev = placement.device_of(b);
            let gpu = devices[dev].gpu();
            let t0 = std::time::Instant::now();
            let gate = w(gpu, &shards, &format!("blk.{b}.ffn_gate_exps.weight"))?;
            let up = w(gpu, &shards, &format!("blk.{b}.ffn_up_exps.weight"))?;
            let down = w(gpu, &shards, &format!("blk.{b}.ffn_down_exps.weight"))?;
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
                    "  [+ head + final mixer]"
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
            head,
            out_hc_norm,
            out_hc_down,
            out_hc_up,
            ple,
            placed,
            stats,
            keys,
        })
    }

    /// The device a block runs on.
    pub fn device_of(&self, block: usize) -> usize {
        self.placement.device_of(block)
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
        let head_dev = self.placement.device_of(self.cfg.n_layers - 1);
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
        Ok(PipelineSet { map, batch })
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
    let buf = breccia_kernels::gemv::upload(gpu, t.as_slice())
        .with_context(|| format!("uploading {name} ({} B)", t.n_bytes()))?;
    Ok(W { buf, key })
}

fn raw(gpu: &Gpu, shards: &ShardSet, name: &str) -> Result<Buf> {
    let t = shards
        .tensor(name)
        .ok_or_else(|| anyhow!("missing tensor `{name}`"))?;
    breccia_kernels::gemv::upload(gpu, t.as_slice())
        .with_context(|| format!("uploading {name} ({} B)", t.n_bytes()))
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

    #[test]
    fn rss_is_readable() {
        let (cur, peak) = rss_bytes();
        assert!(cur > 0, "the process must have resident memory");
        assert!(peak >= cur);
    }
}
