//! Validation against the real 75 GB IQ3_XXS shards
//! (`docs/specs/model-artifacts-inventory.md`).
//!
//! Skips silently when `/srv/strata-data` is not mounted, so it is safe in any
//! environment; on this machine it is the authoritative loader test.

use breccia_format::{GgmlType, Group, Pack, ShardSet};
use std::path::Path;
use std::time::Instant;

const SHARD1: &str =
    "/srv/strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf";
const PACK_DIR: &str = "/srv/strata-data/packs/iq3_xxs";

#[test]
fn real_shards_match_the_inventory() {
    if !Path::new(SHARD1).exists() {
        eprintln!("skipping: {SHARD1} not present");
        return;
    }

    let started = Instant::now();
    let shards = ShardSet::open_family(SHARD1).expect("open shard family");
    let layout = shards.layout().expect("classify tensors");
    let elapsed = started.elapsed();

    // §2.1 container facts.
    let parts = shards.parts();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0].kv_count(), 67);
    assert_eq!(parts[1].kv_count(), 3);
    assert_eq!(parts[0].tensor_count(), 1223);
    assert_eq!(parts[1].tensor_count(), 1);
    assert_eq!(parts[0].data_offset(), 11_024_256);
    assert_eq!(parts[1].data_offset(), 192);
    for part in parts {
        assert_eq!(part.version(), 3);
        assert_eq!(part.alignment(), 32);
        assert_eq!(part.trailing_bytes(), 0, "data must end at EOF");
        assert_eq!(part.data_end(), part.file_len());
    }

    // §2.2 split rules.
    assert_eq!(shards.split().count, 2);
    assert_eq!(shards.split().declared_tensors, Some(1224));
    assert_eq!(shards.split().shard_tensors, vec![1223, 1]);
    assert_eq!(shards.tensor_count(), 1224);

    // §2.5 group totals.
    assert_eq!(layout.expert_bytes(), 42_912_972_800);
    assert_eq!(layout.group_bytes(Group::Ple), 28_800_138_240);
    assert_eq!(layout.group_bytes(Group::Embedding), 273_152_000);
    assert_eq!(layout.group_bytes(Group::Output), 437_043_200);
    assert_eq!(layout.total_bytes(), 75_828_974_080);
    assert_eq!(shards.total_tensor_bytes(), 75_828_974_080);

    // 48 blocks × 512 experts, documented block-0/block-1 blob sizes.
    assert_eq!(layout.experts.len(), 48);
    for block in &layout.experts {
        assert_eq!(block.n_experts, 512);
    }
    let block0 = layout.block(0).unwrap();
    assert_eq!(block0.blob_bytes(), 1_868_800);
    assert_eq!(block0.gate.ty(), GgmlType::IQ2_XS);
    assert_eq!(block0.down.ty(), GgmlType::IQ4_NL);
    let block1 = layout.block(1).unwrap();
    assert_eq!(block1.blob_bytes(), 1_305_600);
    assert_eq!(block1.gate.ty(), GgmlType::IQ2_XXS);
    assert_eq!(block1.down.ty(), GgmlType::Q2_0);
    assert_eq!(
        layout
            .experts
            .iter()
            .map(|b| b.blob_bytes() * 512)
            .sum::<u64>(),
        42_912_972_800
    );

    // §2.4/§2.6 PLE and router.
    let ple = layout.ple.expect("PLE table");
    assert_eq!(ple.dims(), &[160, 320_001_536]);
    assert_eq!(ple.ty(), GgmlType::IQ4_NL);
    assert!(shards.parts()[1]
        .tensor("per_layer_token_embd.weight")
        .is_some());
    for name in [
        "blk.1.ple_key.weight",
        "blk.1.ple_value.weight",
        "blk.1.ple_conv1d.weight",
    ] {
        assert!(shards.tensor(name).is_some(), "{name} missing");
    }
    assert_eq!(
        shards.tensor("blk.1.ple_conv1d.weight").unwrap().ty(),
        GgmlType::F16
    );
    let router = shards.tensor("blk.0.ffn_gate_inp.weight").unwrap();
    assert_eq!(router.dims(), &[2560, 512]);
    assert_eq!(router.ty(), GgmlType::BF16);

    // Zero-copy: an expert slice points inside the mapping of shard 1.
    let expert = block0.expert_slices(511).unwrap();
    assert_eq!(expert.len(), 1_868_800);
    let base = shards.parts()[0].bytes().as_ptr() as usize;
    assert!(
        (base..base + shards.parts()[0].bytes().len()).contains(&(expert.gate.as_ptr() as usize))
    );

    // §4: the pack must agree with the GGUF it was generated from.
    if Path::new(PACK_DIR).exists() {
        let pack = Pack::open(PACK_DIR).expect("open pack");
        let stats = pack.validate_against(&shards).expect("pack vs gguf");
        assert_eq!(stats.rows, 1079);
        assert_eq!(stats.native_rows, 302);
        assert_eq!(stats.dense_rows, 777);
        assert_eq!(stats.native_layers, 48);
        assert_eq!(stats.expert_bytes, 42_912_972_800);
        // Expert-major layout: pack blob == GGUF slices, byte for byte.
        let dims = (ple.ne0(), block0.gate.ne1());
        let from_pack = pack.expert_slices(1, 7, 2560, 640).unwrap();
        let from_gguf = layout.block(1).unwrap().expert_slices(7).unwrap();
        assert_eq!(from_pack[0], from_gguf.gate);
        assert_eq!(from_pack[1], from_gguf.up);
        assert_eq!(from_pack[2], from_gguf.down);
        let _ = dims;
    }

    // Loading must not touch the 75 GB of payload: mmap + header parse only.
    assert!(
        elapsed.as_secs_f64() < 2.0,
        "opening the shards took {elapsed:?} (should be a header-only mmap parse)"
    );
    eprintln!(
        "real shards: 1224 tensors, experts {} B, layout+pack validate in {elapsed:?}",
        layout.expert_bytes()
    );
}
