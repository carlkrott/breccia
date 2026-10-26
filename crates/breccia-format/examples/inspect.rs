//! Inspect and validate a GGUF shard set against
//! `docs/specs/model-artifacts-inventory.md`.
//!
//! ```bash
//! cargo run -p breccia-format --release --example inspect -- \
//!     '/srv/strata-data/models/IQ3_XXS/*-0000?-of-00002.gguf' \
//!     --pack /srv/strata-data/packs/iq3_xxs \
//!     --mtp  /srv/strata-data/mtp/rt
//! ```
//!
//! Structural invariants (tensor-data ends at EOF, alignment, split totals,
//! expert blob geometry) are always checked; the inventory's reference numbers
//! are asserted only for the IQ3_XXS `qwen4exp` two-shard model they describe.
//! Exit code is non-zero on any mismatch.

use anyhow::{bail, ensure, Context, Result};
use breccia_format::{GgmlType, Group, ModelLayout, MtpRt, Pack, ShardSet};
use std::path::PathBuf;
use std::time::Instant;

// --- Reference numbers from docs/specs/model-artifacts-inventory.md ---------
/// §2.1: shard 1 has 67 KVs and 1223 tensors; shard 2 has 3 KVs and 1 tensor.
const REF_KV: [usize; 2] = [67, 3];
const REF_TENSORS: [usize; 2] = [1223, 1];
/// §2.2 `split.tensors.count`.
const REF_SPLIT_TENSORS: u64 = 1224;
/// §2.5: the 144 expert tensors sum to exactly the pack's `experts.bin` size.
const REF_EXPERT_BYTES: u64 = 42_912_972_800;
/// §2.5/§2.6 shapes.
const REF_PLE_SHAPE: [u64; 2] = [160, 320_001_536];
const REF_ROUTER_SHAPE: [u64; 2] = [2560, 512];
/// §4.3 example layer sizes (`blob_bytes`).
const REF_BLOCK0_BLOB: u64 = 1_868_800;
const REF_BLOCK1_BLOB: u64 = 1_305_600;
/// §4.2: 512 experts per block, 48 blocks.
const REF_N_EXPERT: usize = 512;
const REF_BLOCKS: usize = 48;

struct Args {
    target: String,
    pack: Option<PathBuf>,
    mtp: Option<PathBuf>,
    all_blocks: bool,
}

fn parse_args() -> Result<Args> {
    let mut target = None;
    let mut pack = None;
    let mut mtp = None;
    let mut all_blocks = false;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--pack" => pack = Some(PathBuf::from(it.next().context("--pack needs a path")?)),
            "--mtp" => mtp = Some(PathBuf::from(it.next().context("--mtp needs a path")?)),
            "--all-blocks" => all_blocks = true,
            "-h" | "--help" => {
                println!(
                    "usage: inspect <shard-path-or-glob> [--pack DIR] [--mtp DIR] [--all-blocks]"
                );
                std::process::exit(0);
            }
            other if other.starts_with('-') => bail!("unknown flag {other}"),
            other => target = Some(other.to_string()),
        }
    }
    Ok(Args {
        target: target.context("missing shard path (or glob)")?,
        pack,
        mtp,
        all_blocks,
    })
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let started = Instant::now();

    let shards = if args.target.contains(['*', '?']) {
        ShardSet::open_glob(&args.target)?
    } else {
        ShardSet::open_family(&args.target)?
    };
    let parts = shards.parts();
    let reference = shards.architecture() == Some("qwen4exp") && shards.split().count == 2;

    println!("== container ==");
    for (i, part) in parts.iter().enumerate() {
        println!("shard {}: {}", i + 1, part.path().display());
        println!(
            "  version {}  kv {}  tensors {}  alignment {}  data_offset {:#x}  data_end {:#x} \
             (file {:#x}, trailing {} B)",
            part.version(),
            part.kv_count(),
            part.tensor_count(),
            part.alignment(),
            part.data_offset(),
            part.data_end(),
            part.file_len(),
            part.trailing_bytes()
        );
        ensure!(
            part.trailing_bytes() == 0,
            "{}: {} trailing bytes after the tensor-data section (expected data to end at EOF)",
            part.path().display(),
            part.trailing_bytes()
        );
        ensure!(
            part.alignment().is_power_of_two(),
            "alignment {} is not a power of two",
            part.alignment()
        );
    }
    let split = shards.split();
    println!(
        "split: count {}  declared tensors {:?}  per-shard tensors {:?}",
        split.count, split.declared_tensors, split.shard_tensors
    );
    println!(
        "tensors: {} total, {} bytes ({:.2} GiB)",
        shards.tensor_count(),
        shards.total_tensor_bytes(),
        shards.total_tensor_bytes() as f64 / (1u64 << 30) as f64
    );

    if reference {
        check_reference_container(&shards)?;
    }

    let layout = shards.layout()?;
    print_groups(&layout);
    print_experts(&layout, args.all_blocks);
    print_key_tensors(&shards)?;
    if reference {
        check_reference_layout(&layout, &shards)?;
    }

    if let Some(dir) = &args.pack {
        inspect_pack(dir, &shards, &layout)?;
    }
    if let Some(dir) = &args.mtp {
        inspect_mtp(dir)?;
    }

    println!(
        "\ninspect: PASS in {:.3} s",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

fn check_reference_container(shards: &ShardSet) -> Result<()> {
    let parts = shards.parts();
    ensure!(parts.len() == 2, "expected 2 shards, got {}", parts.len());
    for (i, part) in parts.iter().enumerate() {
        ensure!(
            part.kv_count() == REF_KV[i],
            "shard {}: {} KVs, inventory says {}",
            i + 1,
            part.kv_count(),
            REF_KV[i]
        );
        ensure!(
            part.tensor_count() == REF_TENSORS[i],
            "shard {}: {} tensors, inventory says {}",
            i + 1,
            part.tensor_count(),
            REF_TENSORS[i]
        );
        ensure!(
            part.alignment() == 32,
            "shard {}: alignment {}, inventory says 32",
            i + 1,
            part.alignment()
        );
    }
    ensure!(
        shards.split().declared_tensors == Some(REF_SPLIT_TENSORS),
        "split.tensors.count is {:?}, inventory says {REF_SPLIT_TENSORS}",
        shards.split().declared_tensors
    );
    ensure!(
        shards.split().shard_tensors == REF_TENSORS.to_vec(),
        "per-shard tensor counts {:?}",
        shards.split().shard_tensors
    );
    println!("[ok] container matches inventory §2.1/§2.2 (67+3 KVs, 1223+1 tensors, align 32)");
    Ok(())
}

fn check_reference_layout(layout: &ModelLayout<'_>, shards: &ShardSet) -> Result<()> {
    ensure!(
        layout.expert_bytes() == REF_EXPERT_BYTES,
        "routed experts total {} B, inventory says {REF_EXPERT_BYTES}",
        layout.expert_bytes()
    );
    ensure!(
        layout.experts.len() == REF_BLOCKS,
        "{} expert blocks, inventory says {REF_BLOCKS}",
        layout.experts.len()
    );
    for block in &layout.experts {
        ensure!(
            block.n_experts == REF_N_EXPERT,
            "{}: {} experts, inventory says {REF_N_EXPERT}",
            block.label(),
            block.n_experts
        );
    }
    let blob0 = layout
        .block(0)
        .context("missing blk.0 experts")?
        .blob_bytes();
    let blob1 = layout
        .block(1)
        .context("missing blk.1 experts")?
        .blob_bytes();
    ensure!(
        blob0 == REF_BLOCK0_BLOB && blob1 == REF_BLOCK1_BLOB,
        "per-expert blobs {blob0}/{blob1} B, inventory says {REF_BLOCK0_BLOB}/{REF_BLOCK1_BLOB}"
    );

    // §4.2: Σ blob_bytes × 512 == pack experts.bin size.
    ensure!(
        layout
            .experts
            .iter()
            .map(|b| b.blob_bytes() * REF_N_EXPERT as u64)
            .sum::<u64>()
            == REF_EXPERT_BYTES,
        "Σ blob_bytes × 512 does not match {REF_EXPERT_BYTES}"
    );

    // PLE table lives in shard 2 and keeps its own shape (§2.6).
    let ple = layout.ple.context("no per_layer_token_embd.weight")?;
    ensure!(
        ple.dims() == REF_PLE_SHAPE,
        "PLE table shape {}, inventory says {REF_PLE_SHAPE:?}",
        ple.shape()
    );
    ensure!(ple.ty() == GgmlType::IQ4_NL, "PLE table is {}", ple.ty());
    ensure!(
        shards.parts()[1]
            .tensor("per_layer_token_embd.weight")
            .is_some(),
        "PLE table is not in shard 2"
    );

    // §2.4: PLE module tensors exist only in blk.1.
    for name in [
        "blk.1.ple_key.weight",
        "blk.1.ple_value.weight",
        "blk.1.ple_conv1d.weight",
        "blk.1.ple_norm_key.weight",
        "blk.1.ple_norm_query.weight",
        "blk.1.ple_norm_conv.weight",
    ] {
        ensure!(shards.tensor(name).is_some(), "{name} missing");
    }
    ensure!(
        !shards.tensor_names().any(|n| n.starts_with("blk.0.ple_")),
        "blk.0 unexpectedly carries PLE tensors"
    );

    let router = shards
        .tensor("blk.0.ffn_gate_inp.weight")
        .context("missing blk.0.ffn_gate_inp.weight")?;
    ensure!(
        router.dims() == REF_ROUTER_SHAPE,
        "router shape {}, inventory says {REF_ROUTER_SHAPE:?}",
        router.shape()
    );
    ensure!(router.ty() == GgmlType::BF16, "router is {}", router.ty());

    // Documented block-0 quant pair: gate/up IQ2_XS, down IQ4_NL (§4.2 row 0).
    let block0 = layout.block(0).unwrap();
    ensure!(
        block0.gate.ty() == GgmlType::IQ2_XS && block0.up.ty() == GgmlType::IQ2_XS,
        "blk.0 gate/up are {}/{} (inventory: IQ2_XS/IQ2_XS)",
        block0.gate.ty(),
        block0.up.ty()
    );
    ensure!(
        block0.down.ty() == GgmlType::IQ4_NL,
        "blk.0 down is {} (inventory: IQ4_NL)",
        block0.down.ty()
    );

    println!(
        "[ok] layout matches inventory §2.4-§2.6/§4.2 (experts {REF_EXPERT_BYTES} B, \
         48 blocks × {REF_N_EXPERT}, PLE {} {}, router {}, blk.1 PLE module)",
        ple.shape(),
        ple.ty(),
        router.shape()
    );
    Ok(())
}

fn print_groups(layout: &ModelLayout<'_>) {
    println!("\n== residency groups ==");
    println!(
        "{:<10} {:>7} {:>16} {:>9}  types",
        "group", "tensors", "bytes", "GiB"
    );
    for group in Group::ALL {
        if let Some(summary) = layout.summary(group) {
            println!(
                "{:<10} {:>7} {:>16} {:>9.3}  {}",
                group.name(),
                summary.tensors,
                summary.bytes,
                summary.bytes as f64 / (1u64 << 30) as f64,
                summary.types_summary()
            );
        }
    }
    let total = layout.total_bytes();
    let tensors: usize = layout.summaries.values().map(|s| s.tensors).sum();
    assert_eq!(
        layout.summaries.values().map(|s| s.bytes).sum::<u64>(),
        total,
        "group totals must cover every tensor"
    );
    println!(
        "{:<10} {:>7} {:>16} {:>9.3}",
        "TOTAL",
        tensors,
        total,
        total as f64 / (1u64 << 30) as f64
    );
}

fn print_experts(layout: &ModelLayout<'_>, all: bool) {
    println!("\n== routed experts (per block) ==");
    println!(
        "{:<6} {:<14} {:<6} {:>12} {:>12} {:>12} {:>12}",
        "blk", "gate/up", "down", "gate/expert", "up/expert", "down/expert", "blob/expert"
    );
    let show = |block: &breccia_format::ExpertBlock<'_>| {
        println!(
            "{:<6} {:<14} {:<6} {:>12} {:>12} {:>12} {:>12}",
            block.block,
            format!("{}", block.gate.ty()),
            format!("{}", block.down.ty()),
            block.bytes_per_expert(breccia_format::ExpertRole::Gate),
            block.bytes_per_expert(breccia_format::ExpertRole::Up),
            block.bytes_per_expert(breccia_format::ExpertRole::Down),
            block.blob_bytes()
        );
    };
    let count = layout.experts.len();
    if all || count <= 4 {
        for block in &layout.experts {
            show(block);
        }
    } else {
        for block in layout.experts.iter().take(2) {
            show(block);
        }
        println!("  … {} blocks omitted (--all-blocks to show) …", count - 3);
        if let Some(last) = layout.experts.last() {
            show(last);
        }
    }
    println!(
        "total: {} blocks, {} experts each, {} B of expert weights ({:.2} GiB), Σ blob×experts = {}",
        count,
        layout.experts.first().map_or(0, |b| b.n_experts),
        layout.expert_bytes(),
        layout.expert_bytes() as f64 / (1u64 << 30) as f64,
        layout
            .experts
            .iter()
            .map(|b| b.blob_bytes() * b.n_experts as u64)
            .sum::<u64>()
    );
}

fn print_key_tensors(shards: &ShardSet) -> Result<()> {
    println!("\n== key tensors ==");
    for name in [
        "token_embd.weight",
        "output.weight",
        "output_hc_norm.weight",
        "per_layer_token_embd.weight",
        "blk.0.ffn_gate_inp.weight",
        "blk.0.ffn_gate_inp_shexp.weight",
        "blk.0.attn_qkv.weight",
        "blk.3.attn_q.weight",
        "blk.0.ssm_conv1d.weight",
        "blk.1.ple_key.weight",
        "blk.1.ple_value.weight",
        "blk.1.ple_conv1d.weight",
        "blk.47.indexer.q_proj.weight",
    ] {
        match shards.tensor(name) {
            Some(t) => println!(
                "  {:<34} {:<22} {:<8} {:>14} B",
                t.name(),
                t.shape(),
                t.ty().to_string(),
                t.n_bytes()
            ),
            None => println!("  {name:<34} MISSING"),
        }
    }
    Ok(())
}

fn inspect_pack(dir: &PathBuf, shards: &ShardSet, layout: &ModelLayout<'_>) -> Result<()> {
    println!("\n== strata pack: {} ==", dir.display());
    let pack = Pack::open(dir)?;
    let header = pack.header();
    println!(
        "index v{}  align {}  pool {} B  rows {}  source {:?}",
        header.version,
        header.align,
        header.pool_bytes,
        pack.rows().len(),
        header.source
    );
    let native = pack.native();
    println!(
        "native_experts: {} layers, n_expert {}, experts.bin {} B",
        native.layers.len(),
        native.n_expert,
        native.total_bytes
    );

    let stats = pack.validate_against(shards)?;
    println!(
        "[ok] cross-check vs GGUF: {} rows ({} native, {} dense), {} expert layers, \
         expert bytes {}",
        stats.rows, stats.native_rows, stats.dense_rows, stats.native_layers, stats.expert_bytes
    );
    ensure!(
        stats.expert_bytes == native.total_bytes,
        "GGUF expert bytes {} != experts.bin {}",
        stats.expert_bytes,
        native.total_bytes
    );

    let conversions = pack.conversions()?;
    println!(
        "conversions: schema {} tool {:?} tensors {}",
        conversions.schema,
        conversions.tool,
        conversions.tensors.len()
    );
    if let Ok(manifest) = pack.source_manifest() {
        println!(
            "provenance: experts.bin {} B, native_experts sha256 {}",
            manifest.bytes,
            manifest.native_experts_sha256.as_deref().unwrap_or("?")
        );
    }

    // Verify the documented expert-major `[gate|up|down]` interleaving by
    // comparing pack blobs against the zero-copy GGUF slices, byte for byte.
    let n_embd = layout
        .embedding
        .context("no embedding tensor for expert geometry")?
        .ne0();
    let n_ff = layout
        .experts
        .first()
        .context("no expert blocks for expert geometry")?
        .gate
        .ne1();
    let layers: Vec<u32> = if native.layers.len() > 2 {
        vec![0, 1, native.layers.len() as u32 - 1]
    } else {
        (0..native.layers.len() as u32).collect()
    };
    for &layer in &layers {
        let block = layout
            .block(layer as usize)
            .with_context(|| format!("GGUF has no expert block {layer}"))?;
        for expert in [0usize, 1, REF_N_EXPERT - 1] {
            let from_pack = pack
                .expert_slices(layer, expert as u32, n_embd, n_ff)
                .with_context(|| format!("pack layer {layer} expert {expert}"))?;
            let from_gguf = block.expert_slices(expert)?;
            ensure!(
                from_pack[0] == from_gguf.gate
                    && from_pack[1] == from_gguf.up
                    && from_pack[2] == from_gguf.down,
                "expert-major [gate|up|down] layout mismatch at layer {layer} expert {expert}"
            );
        }
        println!(
            "  layer {layer:>2}: gu_type {:<10} d_type {:<10} blob {} B × {} experts",
            block.gate.ty().to_string(),
            block.down.ty().to_string(),
            block.blob_bytes(),
            block.n_experts
        );
    }
    println!(
        "[ok] experts.bin is byte-identical to the GGUF at layers {layers:?} × experts [0, 1, 511]"
    );
    Ok(())
}

fn inspect_mtp(dir: &PathBuf) -> Result<()> {
    println!("\n== mtp rt: {} ==", dir.display());
    let mtp = MtpRt::open(dir)?;
    let mut kinds = std::collections::BTreeMap::new();
    for row in mtp.rows() {
        *kinds.entry(row.kind.name()).or_insert(0usize) += 1;
    }
    println!(
        "dense.txt: {} rows {:?}, dense.bin {} B",
        mtp.rows().len(),
        kinds,
        mtp.rows().iter().map(|r| r.bytes).sum::<u64>()
    );
    println!(
        "experts.bin: {} experts × {} B blobs",
        mtp.n_experts(),
        mtp.blob_bytes()
    );
    println!("draft_vocab: {:?} token ids", mtp.draft_vocab_len());
    ensure!(
        mtp.rows().iter().map(|r| r.bytes).sum::<u64>() == 116_099_072
            || mtp.rows().iter().map(|r| r.bytes).sum::<u64>() > 0,
        "dense rows cover no bytes"
    );
    if let Some(len) = mtp.draft_vocab_len() {
        let max_id = mtp.draft_tokens().max().unwrap_or(0);
        ensure!(
            max_id < 248_320,
            "draft token id {max_id} exceeds the 248320-token vocabulary"
        );
        println!("[ok] {len} draft token ids, max {max_id} < 248320");
    }
    println!(
        "[ok] mtp runtime readable: {} dense rows, {} expert blobs",
        mtp.rows().len(),
        mtp.n_experts()
    );
    Ok(())
}
