//! `breccia-format` — the engine's ingestion layer: a zero-copy GGUF v3 reader
//! with split-residency classification, plus read-only readers for the strata
//! native pack and the MTP draft runtime files.
//!
//! Everything is memory-mapped: no tensor byte is copied at open time, and
//! [`TensorView::as_slice`] hands out slices of the mapping. Opening the real
//! 75 GB two-shard model touches only the header pages (≈11 MB of metadata for
//! shard 1).
//!
//! # Pieces
//!
//! | API | What it does |
//! |---|---|
//! | [`Gguf::open`] | one GGUF file: header, typed metadata, tensor table, alignment, data section end |
//! | [`ShardSet`] | N shards as one logical tensor namespace (`split.*` validated) |
//! | [`ModelLayout`] | dense / experts / PLE / embedding / output classification with per-group byte totals |
//! | [`ExpertBlock`] | per-block routed-expert geometry: 512 experts × `[gate\|up\|down]` blob |
//! | [`Pack::open`] | strata pack (`index.txt` v3, `native_experts.txt`, `experts.bin`, `dense.bin`) |
//! | [`MtpRt::open`] | MTP draft runtime (`dense.txt`, `experts.bin`, `draft_vocab.bin`) |
//! | [`GgmlType`] | ggml type ids → block geometry, including the fork-only `Q2_0 = 64/18` |
//!
//! # Split residency
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use breccia_format::ShardSet;
//!
//! let shards = ShardSet::open_family("/srv/strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf")?;
//! let layout = shards.layout()?;
//! println!("dense {} B, experts {} B, ple {} B",
//!     layout.group_bytes(breccia_format::Group::Dense),
//!     layout.expert_bytes(),
//!     layout.group_bytes(breccia_format::Group::Ple));
//! let block0 = layout.block(0).unwrap();
//! let expert3 = block0.expert_slices(3)?;       // gate | up | down, zero-copy
//! println!("expert 3 of block 0: {} bytes", expert3.len());
//! # Ok(())
//! # }
//! ```
//!
//! # Error policy
//!
//! Every failure is a typed [`FormatError`]; malformed files never panic. The
//! GGUF path has no dependency on the pack/MTP readers — those are opt-in.

mod filemap;

pub mod error;
pub mod gguf;
pub mod layout;
pub mod metadata;
pub mod mtp;
pub mod pack;
pub mod quant;
pub mod shard;

#[cfg(test)]
pub(crate) mod testutil;

pub use error::{FormatError, Result};
pub use gguf::{Gguf, SplitRef, TensorView, DEFAULT_ALIGNMENT};
pub use layout::{
    classify, expert_role, split_block_name, ExpertBlock, ExpertRole, ExpertSlices, Group,
    GroupSummary, ModelLayout,
};
pub use metadata::{MetaType, MetadataArray, MetadataEntry, MetadataMap, MetadataValue};
pub use mtp::{MtpDenseKind, MtpDenseRow, MtpGeometry, MtpRt};
pub use pack::{
    Conversions, ConvertedTensor, NativeExpertTable, NativeLayer, Pack, PackHeader, PackKind,
    PackRow, PackStats, PackTensor, SourceManifest, SourceShard,
};
pub use quant::{BlockGeometry, GgmlType};
pub use shard::{expand_glob, sibling_paths, wildcard_match, ShardSet, SplitInfo};
