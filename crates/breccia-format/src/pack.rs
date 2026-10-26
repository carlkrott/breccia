//! Read-only reader for the strata **native pack** (`/srv/strata-data/packs/…`).
//!
//! Files (inventory §4):
//!
//! * `index.txt` — `# strata pack index v3` + `# align A pool P tensors N`, then
//!   one row of exactly **19** whitespace-separated fields per tensor
//!   (`name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits
//!   code_bias group_elems codebook has_offset codes_bytes scales_bytes
//!   offset_bytes scales_fp16 act_kind`). A row that does not parse into 19
//!   fields is refused, matching `strata/src/core/weights.cpp:177`.
//! * `dense.bin` — headerless arena of exactly `pool` bytes, 64-byte aligned rows.
//! * `native_experts.txt` — header + 48 rows
//!   `layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard]`.
//! * `experts.bin` — 48 layers × 512 expert-major `[gate|up|down]` blobs.
//! * `conversions.json`, `experts.bin.src.json` — provenance sidecars.
//!
//! This module is only reachable through [`Pack::open`]: the plain GGUF path has
//! no dependency on the pack.

use crate::error::{FormatError, Result};
use crate::filemap::map_optional;
use crate::gguf::TensorView;
use crate::layout::ExpertBlock;
use crate::quant::GgmlType;
use crate::shard::ShardSet;
use memmap2::Mmap;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Storage kind of an `index.txt` row.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PackKind {
    /// 0 — served straight from the GGUF given by `--native`; the row carries
    /// shape only (`src_*` and `dst_bytes` are 0).
    Native,
    /// 1 — BF16 promoted into F32 (reserved; unused in this pack).
    Bf16InF32,
    /// 2 — F32 copied verbatim into `dense.bin`.
    F32,
    /// 3 — F16 narrowed into F32 (reserved; unused in this pack).
    F16InF32,
    /// 4 — BF16 stored raw in `dense.bin`.
    Bf16,
    /// 5 — F16 stored raw in `dense.bin`.
    F16,
}

impl PackKind {
    pub fn from_code(code: u32) -> Option<Self> {
        Some(match code {
            0 => Self::Native,
            1 => Self::Bf16InF32,
            2 => Self::F32,
            3 => Self::F16InF32,
            4 => Self::Bf16,
            5 => Self::F16,
            _ => return None,
        })
    }

    pub fn code(self) -> u32 {
        match self {
            Self::Native => 0,
            Self::Bf16InF32 => 1,
            Self::F32 => 2,
            Self::F16InF32 => 3,
            Self::Bf16 => 4,
            Self::F16 => 5,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Bf16InF32 => "bf16_in_f32",
            Self::F32 => "f32",
            Self::F16InF32 => "f16_in_f32",
            Self::Bf16 => "bf16",
            Self::F16 => "f16",
        }
    }

    /// Whether the row's bytes live in `dense.bin`.
    pub fn in_dense(self) -> bool {
        !matches!(self, Self::Native)
    }

    /// GGML type these bytes represent, when the kind is a raw float copy.
    pub fn ggml_type(self) -> Option<GgmlType> {
        Some(match self {
            Self::Native => return None,
            Self::Bf16 | Self::Bf16InF32 => GgmlType::BF16,
            Self::F32 => GgmlType::F32,
            Self::F16 | Self::F16InF32 => GgmlType::F16,
        })
    }
}

impl std::fmt::Display for PackKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// `# align A pool P tensors N` header of `index.txt`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackHeader {
    pub version: u32,
    pub align: u64,
    pub pool_bytes: u64,
    pub tensors: usize,
    /// Source shard named in the index header, when present.
    pub source: Option<String>,
}

/// One 19-field `index.txt` row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PackRow {
    pub name: String,
    /// File id (always 0 = `dense.bin` in this pack).
    pub file: u32,
    pub kind: PackKind,
    pub src_off: u64,
    pub src_bytes: u64,
    pub dst_off: u64,
    pub dst_bytes: u64,
    pub ne0: u64,
    pub ne1: u64,
    pub code_bits: u32,
    pub code_bias: i32,
    pub group_elems: u32,
    pub codebook: u32,
    pub has_offset: u32,
    pub codes_bytes: u64,
    pub scales_bytes: u64,
    pub offset_bytes: u64,
    pub scales_fp16: u32,
    pub act_kind: u32,
}

/// One row of `native_experts.txt`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeLayer {
    pub layer: u32,
    /// ggml type of gate **and** up for this layer.
    pub gate_up_type: GgmlType,
    /// ggml type of down.
    pub down_type: GgmlType,
    /// Offset of this layer's first expert blob inside `experts.bin`.
    pub offset: u64,
    /// Bytes per expert blob (`gate rows | up rows | down rows`).
    pub blob_bytes: u64,
    /// Absolute offsets of the three GGUF tensors inside the source shard.
    pub gate_off: u64,
    pub up_off: u64,
    pub down_off: u64,
    /// v4 packs may name the shard a layer's experts live in.
    pub shard: Option<u32>,
}

impl NativeLayer {
    /// `[gate, up, down]` bytes for one expert, derived from the layer's types.
    pub fn role_bytes(&self, n_embd: u64, n_ff: u64) -> Result<[u64; 3]> {
        let gu = self
            .gate_up_type
            .checked_bytes(n_embd * n_ff)
            .ok_or_else(|| {
                FormatError::malformed(
                    format!("native_experts layer {}", self.layer),
                    format!(
                        "{} × {} elements are not {} blocks",
                        n_embd, n_ff, self.gate_up_type
                    ),
                )
            })?;
        let down = self.down_type.checked_bytes(n_ff * n_embd).ok_or_else(|| {
            FormatError::malformed(
                format!("native_experts layer {}", self.layer),
                format!(
                    "{} × {} elements are not {} blocks",
                    n_ff, n_embd, self.down_type
                ),
            )
        })?;
        Ok([gu, gu, down])
    }
}

/// Parsed `native_experts.txt`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeExpertTable {
    pub n_expert: u64,
    /// Total `experts.bin` bytes declared by the header.
    pub total_bytes: u64,
    pub layers: Vec<NativeLayer>,
}

impl NativeExpertTable {
    pub fn layer(&self, layer: u32) -> Option<&NativeLayer> {
        self.layers.get(layer as usize).filter(|l| l.layer == layer)
    }

    /// Sum of `blob_bytes × n_expert` over all layers.
    pub fn computed_bytes(&self) -> u64 {
        self.layers
            .iter()
            .map(|l| l.blob_bytes * self.n_expert)
            .sum()
    }
}

/// A tensor resolved through the pack.
#[derive(Clone, Copy, Debug)]
pub enum PackTensor<'a> {
    /// Bytes inside `dense.bin`.
    Dense(&'a [u8]),
    /// Bytes served straight from the GGUF shards (`kind 0`).
    Native(TensorView<'a>),
}

impl<'a> PackTensor<'a> {
    pub fn as_slice(&self) -> &'a [u8] {
        match self {
            PackTensor::Dense(bytes) => bytes,
            PackTensor::Native(t) => t.as_slice(),
        }
    }

    pub fn source(&self) -> &'static str {
        match self {
            PackTensor::Dense(_) => "dense.bin",
            PackTensor::Native(_) => "gguf",
        }
    }

    pub fn native(&self) -> Option<TensorView<'a>> {
        match self {
            PackTensor::Native(t) => Some(*t),
            PackTensor::Dense(_) => None,
        }
    }
}

/// Counters produced by [`Pack::validate_against`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PackStats {
    pub rows: usize,
    pub native_rows: usize,
    pub dense_rows: usize,
    pub native_layers: usize,
    pub expert_bytes: u64,
    pub dense_bytes: u64,
}

/// The pack directory, mapped read-only.
pub struct Pack {
    dir: PathBuf,
    header: PackHeader,
    rows: Vec<PackRow>,
    index: HashMap<String, usize>,
    dense: Option<Mmap>,
    experts: Option<Mmap>,
    native: NativeExpertTable,
}

impl Pack {
    /// Open `dir` (containing `index.txt`, `dense.bin`, `native_experts.txt`,
    /// `experts.bin`) and validate its internal consistency.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let (header, rows) = parse_index(&dir)?;
        let native = parse_native_experts(&dir)?;
        let dense = map_optional(&dir.join("dense.bin"))?;
        let experts = map_optional(&dir.join("experts.bin"))?;

        let mut index = HashMap::with_capacity(rows.len());
        for (i, row) in rows.iter().enumerate() {
            if index.insert(row.name.clone(), i).is_some() {
                return Err(FormatError::malformed(
                    dir.join("index.txt").display().to_string(),
                    format!("duplicate row for '{}'", row.name),
                ));
            }
        }

        if rows.len() != header.tensors {
            return Err(FormatError::malformed(
                dir.join("index.txt").display().to_string(),
                format!(
                    "header declares {} tensors but {} rows were parsed",
                    header.tensors,
                    rows.len()
                ),
            ));
        }

        // Arena invariants: pool size, 64-byte alignment, non-overlap.
        let mut arena_bytes = 0u64;
        for row in &rows {
            if row.kind.in_dense() {
                arena_bytes += row.dst_bytes;
                if header.align != 0 && row.dst_off % header.align != 0 {
                    return Err(FormatError::malformed(
                        dir.join("index.txt").display().to_string(),
                        format!(
                            "'{}': dst_off {} is not {}-byte aligned",
                            row.name, row.dst_off, header.align
                        ),
                    ));
                }
                if row.dst_off + row.dst_bytes > header.pool_bytes {
                    return Err(FormatError::malformed(
                        dir.join("index.txt").display().to_string(),
                        format!(
                            "'{}': dst range {}..{} exceeds pool {}",
                            row.name,
                            row.dst_off,
                            row.dst_off + row.dst_bytes,
                            header.pool_bytes
                        ),
                    ));
                }
            } else if row.dst_bytes != 0 || row.src_bytes != 0 {
                return Err(FormatError::malformed(
                    dir.join("index.txt").display().to_string(),
                    format!(
                        "'{}': native row must have zero src/dst bytes (got {} / {})",
                        row.name, row.src_bytes, row.dst_bytes
                    ),
                ));
            }
        }
        if arena_bytes != header.pool_bytes {
            return Err(FormatError::malformed(
                dir.join("index.txt").display().to_string(),
                format!(
                    "non-native dst_bytes sum to {arena_bytes} but header pool is {}",
                    header.pool_bytes
                ),
            ));
        }

        match &dense {
            Some(map) => {
                if map.len() as u64 != header.pool_bytes {
                    return Err(FormatError::malformed(
                        dir.join("dense.bin").display().to_string(),
                        format!(
                            "file is {} bytes but index pool is {}",
                            map.len(),
                            header.pool_bytes
                        ),
                    ));
                }
            }
            None => {
                if header.pool_bytes != 0 {
                    return Err(FormatError::malformed(
                        dir.join("dense.bin").display().to_string(),
                        format!("missing (index pool is {} bytes)", header.pool_bytes),
                    ));
                }
            }
        }

        // Expert table ↔ experts.bin.
        if native.n_expert == 0 {
            return Err(FormatError::malformed(
                dir.join("native_experts.txt").display().to_string(),
                "n_expert is zero",
            ));
        }
        for (i, layer) in native.layers.iter().enumerate() {
            if layer.layer as usize != i {
                return Err(FormatError::malformed(
                    dir.join("native_experts.txt").display().to_string(),
                    format!("layer rows are not sequential at row {i}"),
                ));
            }
            if layer.blob_bytes == 0 {
                return Err(FormatError::malformed(
                    dir.join("native_experts.txt").display().to_string(),
                    format!("layer {} has a zero-size expert blob", layer.layer),
                ));
            }
        }
        let computed = native.computed_bytes();
        if computed != native.total_bytes {
            return Err(FormatError::malformed(
                dir.join("native_experts.txt").display().to_string(),
                format!(
                    "blob_bytes × n_expert sums to {computed} but the header declares {}",
                    native.total_bytes
                ),
            ));
        }
        match &experts {
            Some(map) => {
                if map.len() as u64 != native.total_bytes {
                    return Err(FormatError::malformed(
                        dir.join("experts.bin").display().to_string(),
                        format!(
                            "file is {} bytes but native_experts declares {}",
                            map.len(),
                            native.total_bytes
                        ),
                    ));
                }
            }
            None => {
                return Err(FormatError::malformed(
                    dir.join("experts.bin").display().to_string(),
                    "missing",
                ))
            }
        }

        Ok(Self {
            dir,
            header,
            rows,
            index,
            dense,
            experts,
            native,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn header(&self) -> &PackHeader {
        &self.header
    }

    pub fn rows(&self) -> &[PackRow] {
        &self.rows
    }

    pub fn row(&self, name: &str) -> Option<&PackRow> {
        self.index.get(name).map(|&i| &self.rows[i])
    }

    pub fn native(&self) -> &NativeExpertTable {
        &self.native
    }

    /// `dense.bin` bytes for a non-native row.
    pub fn dense_bytes(&self, name: &str) -> Result<&[u8]> {
        let row = self.row(name).ok_or_else(|| FormatError::UnknownTensor {
            name: name.to_string(),
        })?;
        if !row.kind.in_dense() {
            return Err(FormatError::malformed(
                name,
                format!("kind {} is served from the GGUF, not dense.bin", row.kind),
            ));
        }
        let map = self.dense.as_ref().ok_or_else(|| {
            FormatError::malformed(self.dir.join("dense.bin").display().to_string(), "missing")
        })?;
        map.get(row.src_off as usize..(row.src_off + row.src_bytes) as usize)
            .ok_or_else(|| {
                FormatError::malformed(
                    name,
                    format!(
                        "row range {}..{} exceeds dense.bin ({})",
                        row.src_off,
                        row.src_off + row.src_bytes,
                        map.len()
                    ),
                )
            })
    }

    /// Resolve a pack tensor to its bytes: `dense.bin` for kinds 1..5, the GGUF
    /// shards for kind 0 (which is why `native` is required for those rows).
    pub fn tensor_view<'a>(
        &'a self,
        name: &str,
        native: Option<&'a ShardSet>,
    ) -> Result<PackTensor<'a>> {
        let row = self.row(name).ok_or_else(|| FormatError::UnknownTensor {
            name: name.to_string(),
        })?;
        if row.kind.in_dense() {
            Ok(PackTensor::Dense(self.dense_bytes(name)?))
        } else {
            let shards = native.ok_or_else(|| {
                FormatError::malformed(
                    name,
                    "native row needs the GGUF shard set (open it with ShardSet)",
                )
            })?;
            let tensor = shards
                .tensor(name)
                .ok_or_else(|| FormatError::UnknownTensor {
                    name: name.to_string(),
                })?;
            Ok(PackTensor::Native(tensor))
        }
    }

    /// Contiguous expert-major blob: 512 blobs of `[gate rows | up rows | down
    /// rows]` per layer.
    pub fn expert_blob(&self, layer: u32, expert: u32) -> Result<&[u8]> {
        let row = self.native.layer(layer).ok_or_else(|| {
            FormatError::malformed(format!("layer {layer}"), "not in native_experts.txt")
        })?;
        if u64::from(expert) >= self.native.n_expert {
            return Err(FormatError::ExpertOutOfRange {
                block: layer as usize,
                expert: expert as usize,
                n_experts: self.native.n_expert as usize,
            });
        }
        let map = self.experts.as_ref().ok_or_else(|| {
            FormatError::malformed(
                self.dir.join("experts.bin").display().to_string(),
                "missing",
            )
        })?;
        let start = row.offset + u64::from(expert) * row.blob_bytes;
        let end = start + row.blob_bytes;
        map.get(start as usize..end as usize).ok_or_else(|| {
            FormatError::malformed(
                format!("layer {layer} expert {expert}"),
                format!("blob range {start}..{end} exceeds experts.bin"),
            )
        })
    }

    /// Expert blob split into its three role slices (sizes from the layer's
    /// quant types and the model's `(n_embd, n_ff)`).
    pub fn expert_slices(
        &self,
        layer: u32,
        expert: u32,
        n_embd: u64,
        n_ff: u64,
    ) -> Result<[&[u8]; 3]> {
        let blob = self.expert_blob(layer, expert)?;
        let row = self
            .native
            .layer(layer)
            .ok_or_else(|| FormatError::malformed(format!("layer {layer}"), "missing"))?;
        let [gate, up, down] = row.role_bytes(n_embd, n_ff)?;
        if gate + up + down != row.blob_bytes {
            return Err(FormatError::malformed(
                format!("layer {layer}"),
                format!(
                    "role sizes {gate}+{up}+{down} do not sum to blob_bytes {}",
                    row.blob_bytes
                ),
            ));
        }
        let (a, b, c) = (gate as usize, up as usize, down as usize);
        Ok([&blob[..a], &blob[a..a + b], &blob[a + b..a + b + c]])
    }

    /// `conversions.json` sidecar.
    pub fn conversions(&self) -> Result<Conversions> {
        let path = self.dir.join("conversions.json");
        let bytes = std::fs::read(&path).map_err(|e| FormatError::io(&path, e))?;
        serde_json::from_slice(&bytes)
            .map_err(|e| FormatError::malformed(path.display().to_string(), e.to_string()))
    }

    /// `experts.bin.src.json` provenance sidecar.
    pub fn source_manifest(&self) -> Result<SourceManifest> {
        let path = self.dir.join("experts.bin.src.json");
        let bytes = std::fs::read(&path).map_err(|e| FormatError::io(&path, e))?;
        serde_json::from_slice(&bytes)
            .map_err(|e| FormatError::malformed(path.display().to_string(), e.to_string()))
    }

    /// Cross-check the pack against the GGUF shards it was generated from:
    /// native expert rows must match the GGUF tensors byte-for-byte in type,
    /// offset and per-expert size, and every dense row must exist in the GGUF
    /// with the same shape and a matching raw float type.
    pub fn validate_against(&self, shards: &ShardSet) -> Result<PackStats> {
        let layout = shards.layout()?;
        let mut stats = PackStats {
            rows: self.rows.len(),
            dense_bytes: self.header.pool_bytes,
            ..PackStats::default()
        };

        // Model dimensions for role-size derivation.
        let (n_embd, n_ff) = model_dims(&layout)?;

        for row in &self.rows {
            let tensor = shards.tensor(&row.name).ok_or_else(|| {
                FormatError::malformed(
                    self.dir.join("index.txt").display().to_string(),
                    format!("'{}' is not a tensor of the GGUF shard set", row.name),
                )
            })?;
            if tensor.ne0() != row.ne0 {
                return Err(FormatError::malformed(
                    row.name.clone(),
                    format!("ne0 {} vs GGUF {}", row.ne0, tensor.ne0()),
                ));
            }
            if row.ne1 != 0 && tensor.ne(1).unwrap_or(1) != row.ne1 {
                return Err(FormatError::malformed(
                    row.name.clone(),
                    format!("ne1 {} vs GGUF {}", row.ne1, tensor.ne(1).unwrap_or(1)),
                ));
            }
            if row.kind == PackKind::Native {
                stats.native_rows += 1;
            } else {
                stats.dense_rows += 1;
                if let Some(expect) = row.kind.ggml_type() {
                    if tensor.ty() != expect {
                        return Err(FormatError::malformed(
                            row.name.clone(),
                            format!("pack kind {} but GGUF type {}", row.kind, tensor.ty()),
                        ));
                    }
                }
            }
        }

        for layer in &self.native.layers {
            let block: &ExpertBlock<'_> = layout.block(layer.layer as usize).ok_or_else(|| {
                FormatError::malformed(
                    format!("native_experts layer {}", layer.layer),
                    "no routed-expert tensors in the GGUF",
                )
            })?;
            if block.gate.ty() != layer.gate_up_type || block.up.ty() != layer.gate_up_type {
                return Err(FormatError::malformed(
                    format!("layer {}", layer.layer),
                    format!(
                        "gate/up types {}/{} but native_experts says {}",
                        block.gate.ty(),
                        block.up.ty(),
                        layer.gate_up_type
                    ),
                ));
            }
            if block.down.ty() != layer.down_type {
                return Err(FormatError::malformed(
                    format!("layer {}", layer.layer),
                    format!(
                        "down type {} but native_experts says {}",
                        block.down.ty(),
                        layer.down_type
                    ),
                ));
            }
            let starts = [
                block.gate.file_range().start,
                block.up.file_range().start,
                block.down.file_range().start,
            ];
            let declared = [layer.gate_off, layer.up_off, layer.down_off];
            if starts != declared {
                return Err(FormatError::malformed(
                    format!("layer {}", layer.layer),
                    format!(
                        "GGUF tensor offsets {starts:?} do not match native_experts {declared:?}"
                    ),
                ));
            }
            if block.blob_bytes() != layer.blob_bytes {
                return Err(FormatError::malformed(
                    format!("layer {}", layer.layer),
                    format!(
                        "per-expert blob {} B from GGUF geometry, {} B declared",
                        block.blob_bytes(),
                        layer.blob_bytes
                    ),
                ));
            }
            let [gate, up, down] = layer.role_bytes(n_embd, n_ff)?;
            if gate + up + down != layer.blob_bytes {
                return Err(FormatError::malformed(
                    format!("layer {}", layer.layer),
                    format!(
                        "role sizes {gate}+{up}+{down} do not sum to blob_bytes {}",
                        layer.blob_bytes
                    ),
                ));
            }
            stats.native_layers += 1;
            stats.expert_bytes += block.total_bytes();
        }

        if stats.expert_bytes != self.native.total_bytes {
            return Err(FormatError::malformed(
                self.dir.join("native_experts.txt").display().to_string(),
                format!(
                    "GGUF expert tensors total {} B, experts.bin is {} B",
                    stats.expert_bytes, self.native.total_bytes
                ),
            ));
        }
        Ok(stats)
    }
}

impl std::fmt::Debug for Pack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pack")
            .field("dir", &self.dir)
            .field("header", &self.header)
            .field("rows", &self.rows.len())
            .field("native_layers", &self.native.layers.len())
            .field("dense", &self.dense.as_ref().map(|m| m.len()))
            .field("experts", &self.experts.as_ref().map(|m| m.len()))
            .finish()
    }
}

fn model_dims(layout: &crate::layout::ModelLayout<'_>) -> Result<(u64, u64)> {
    // n_embd from the embedding table, n_ff from any routed expert gate tensor.
    let n_embd = layout
        .embedding
        .map(|t| t.ne0())
        .or_else(|| layout.dense.first().map(|t| t.ne0()))
        .ok_or_else(|| {
            FormatError::malformed("layout", "cannot derive n_embd (no embedding tensor)")
        })?;
    let n_ff = layout
        .experts
        .first()
        .map(|b| b.gate.ne1())
        .ok_or_else(|| FormatError::malformed("layout", "no routed-expert blocks"))?;
    Ok((n_embd, n_ff))
}

fn parse_index(dir: &Path) -> Result<(PackHeader, Vec<PackRow>)> {
    let path = dir.join("index.txt");
    let text = std::fs::read_to_string(&path).map_err(|e| FormatError::io(&path, e))?;
    let mut lines = text.lines().enumerate();

    let (_, first) = lines.next().ok_or_else(|| {
        FormatError::malformed(path.display().to_string(), "empty file (no header)")
    })?;
    if !first.starts_with("# strata pack index") {
        return Err(FormatError::malformed(
            path.display().to_string(),
            format!("unexpected header {first:?}"),
        ));
    }
    let version = first
        .split_whitespace()
        .find_map(|t| t.strip_prefix('v').and_then(|v| v.parse::<u32>().ok()))
        .ok_or_else(|| {
            FormatError::malformed(path.display().to_string(), "header has no version")
        })?;
    let source = first
        .find(" from ")
        .map(|i| first[i + 6..].trim().to_string())
        .filter(|s| !s.is_empty());

    let (_, second) = lines.next().ok_or_else(|| {
        FormatError::malformed(
            path.display().to_string(),
            "missing '# align …' header line",
        )
    })?;
    let tokens: Vec<&str> = second.split_whitespace().collect();
    let get = |key: &str| -> Option<u64> {
        tokens
            .iter()
            .position(|t| *t == key)
            .and_then(|i| tokens.get(i + 1))
            .and_then(|v| v.parse().ok())
    };
    let align = get("align").ok_or_else(|| {
        FormatError::malformed(path.display().to_string(), "header has no 'align' field")
    })?;
    let pool_bytes = get("pool").ok_or_else(|| {
        FormatError::malformed(path.display().to_string(), "header has no 'pool' field")
    })?;
    let tensors = get("tensors").ok_or_else(|| {
        FormatError::malformed(path.display().to_string(), "header has no 'tensors' field")
    })?;

    let mut rows = Vec::new();
    for (lineno, line) in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let ctx = format!("{}:{}", path.display(), lineno + 1);
        let fields: Vec<&str> = trimmed.split_whitespace().collect();
        if fields.len() != 19 {
            return Err(FormatError::malformed(
                ctx,
                format!("expected 19 fields, got {}", fields.len()),
            ));
        }
        rows.push(parse_row(&ctx, &fields)?);
    }

    Ok((
        PackHeader {
            version,
            align,
            pool_bytes,
            tensors: usize::try_from(tensors).map_err(|_| {
                FormatError::malformed(
                    path.display().to_string(),
                    format!("tensor count {tensors} does not fit in usize"),
                )
            })?,
            source,
        },
        rows,
    ))
}

fn parse_row(ctx: &str, f: &[&str]) -> Result<PackRow> {
    let u64_at = |i: usize, what: &str| -> Result<u64> {
        f[i].parse::<u64>().map_err(|_| {
            FormatError::malformed(ctx, format!("field {i} ({what}) {:?} is not u64", f[i]))
        })
    };
    let u32_at = |i: usize, what: &str| -> Result<u32> {
        f[i].parse::<u32>().map_err(|_| {
            FormatError::malformed(ctx, format!("field {i} ({what}) {:?} is not u32", f[i]))
        })
    };
    let file = u32_at(1, "file")?;
    let kind_code = u32_at(2, "kind")?;
    let kind = PackKind::from_code(kind_code)
        .ok_or_else(|| FormatError::malformed(ctx, format!("unknown pack kind {kind_code}")))?;
    Ok(PackRow {
        name: f[0].to_string(),
        file,
        kind,
        src_off: u64_at(3, "src_off")?,
        src_bytes: u64_at(4, "src_bytes")?,
        dst_off: u64_at(5, "dst_off")?,
        dst_bytes: u64_at(6, "dst_bytes")?,
        ne0: u64_at(7, "ne0")?,
        ne1: u64_at(8, "ne1")?,
        code_bits: u32_at(9, "code_bits")?,
        code_bias: f[10]
            .parse::<i32>()
            .map_err(|_| FormatError::malformed(ctx, format!("field 10 {:?} is not i32", f[10])))?,
        group_elems: u32_at(11, "group_elems")?,
        codebook: u32_at(12, "codebook")?,
        has_offset: u32_at(13, "has_offset")?,
        codes_bytes: u64_at(14, "codes_bytes")?,
        scales_bytes: u64_at(15, "scales_bytes")?,
        offset_bytes: u64_at(16, "offset_bytes")?,
        scales_fp16: u32_at(17, "scales_fp16")?,
        act_kind: u32_at(18, "act_kind")?,
    })
}

fn parse_native_experts(dir: &Path) -> Result<NativeExpertTable> {
    let path = dir.join("native_experts.txt");
    let text = std::fs::read_to_string(&path).map_err(|e| FormatError::io(&path, e))?;
    let mut lines = text.lines().enumerate();

    let (_, header) = lines.next().ok_or_else(|| {
        FormatError::malformed(path.display().to_string(), "empty file (no header)")
    })?;
    if !header.starts_with("# strata native experts") {
        return Err(FormatError::malformed(
            path.display().to_string(),
            format!("unexpected header {header:?}"),
        ));
    }
    // Header form: `… (n_expert 512, total 42912972800; …)` — the key may be
    // prefixed by '(' and the value suffixed by ',' or ';'.
    let tokens: Vec<&str> = header.split_whitespace().collect();
    let header_uint = |key: &str| -> Option<u64> {
        let at = tokens
            .iter()
            .position(|t| t.trim_start_matches('(') == key)?;
        tokens
            .get(at + 1)?
            .trim_matches(|c: char| !c.is_ascii_digit())
            .parse()
            .ok()
    };
    let n_expert = header_uint("n_expert").ok_or_else(|| {
        FormatError::malformed(path.display().to_string(), "header has no n_expert")
    })?;
    let total_bytes = header_uint("total")
        .ok_or_else(|| FormatError::malformed(path.display().to_string(), "header has no total"))?;

    let mut layers = Vec::new();
    for (lineno, line) in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let ctx = format!("{}:{}", path.display(), lineno + 1);
        let f: Vec<&str> = trimmed.split_whitespace().collect();
        if f.len() != 8 && f.len() != 9 {
            return Err(FormatError::malformed(
                ctx,
                format!("expected 8 or 9 fields, got {}", f.len()),
            ));
        }
        let num = |i: usize, what: &str| -> Result<u64> {
            f[i].parse::<u64>().map_err(|_| {
                FormatError::malformed(&ctx, format!("field {i} ({what}) {:?} is not u64", f[i]))
            })
        };
        layers.push(NativeLayer {
            layer: u32::try_from(num(0, "layer")?)
                .map_err(|_| FormatError::malformed(&ctx, "layer does not fit in u32"))?,
            gate_up_type: GgmlType::from_id(
                u32::try_from(num(1, "gu_type")?)
                    .map_err(|_| FormatError::malformed(&ctx, "gu_type does not fit in u32"))?,
            ),
            down_type: GgmlType::from_id(
                u32::try_from(num(2, "d_type")?)
                    .map_err(|_| FormatError::malformed(&ctx, "d_type does not fit in u32"))?,
            ),
            offset: num(3, "offset")?,
            blob_bytes: num(4, "blob_bytes")?,
            gate_off: num(5, "gate_off")?,
            up_off: num(6, "up_off")?,
            down_off: num(7, "down_off")?,
            shard: match f.get(8) {
                Some(v) => Some(v.parse::<u32>().map_err(|_| {
                    FormatError::malformed(&ctx, format!("shard {v:?} is not u32"))
                })?),
                None => None,
            },
        });
    }

    // Offset chain: layer L+1 starts where layer L's blobs end.
    let mut cursor = 0u64;
    for layer in &layers {
        if layer.offset != cursor {
            return Err(FormatError::malformed(
                path.display().to_string(),
                format!(
                    "layer {} offset {} should be {} (contiguous expert-major layout)",
                    layer.layer, layer.offset, cursor
                ),
            ));
        }
        cursor = layer.offset + layer.blob_bytes * n_expert;
    }

    Ok(NativeExpertTable {
        n_expert,
        total_bytes,
        layers,
    })
}

/// `conversions.json` (schema 1).
#[derive(Clone, Debug, Deserialize)]
pub struct Conversions {
    pub schema: u32,
    #[serde(default)]
    pub tool: Option<String>,
    #[serde(default)]
    pub compat_bf16: bool,
    #[serde(default)]
    pub source_shards: Vec<SourceShard>,
    #[serde(default)]
    pub tensors: Vec<ConvertedTensor>,
}

/// Source shard recorded in a sidecar.
#[derive(Clone, Debug, Deserialize)]
pub struct SourceShard {
    pub name: String,
    pub size: u64,
}

/// One recorded tensor conversion (empty for this model).
#[derive(Clone, Debug, Deserialize)]
pub struct ConvertedTensor {
    pub name: String,
    #[serde(default)]
    pub shard: Option<serde_json::Value>,
    #[serde(default)]
    pub source_type: Option<String>,
    #[serde(default)]
    pub dst_type: Option<String>,
    #[serde(default)]
    pub method: Option<String>,
    #[serde(default)]
    pub exact: bool,
    #[serde(default)]
    pub max_abs_err: Option<f64>,
    #[serde(default)]
    pub source_sha256: Option<String>,
}

/// `experts.bin.src.json` (schema 1) — provenance fingerprint of `experts.bin`.
#[derive(Clone, Debug, Deserialize)]
pub struct SourceManifest {
    pub schema: u32,
    #[serde(default)]
    pub shards: Vec<SourceShard>,
    #[serde(default)]
    pub native_experts_sha256: Option<String>,
    #[serde(default)]
    pub bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::GgmlType;
    use crate::testutil::{Builder, TempDir};

    const INDEX_HEADER: &str = "# strata pack index v3 -- generated by tools/iq_pack.py (native experts) from m-00001-of-00002.gguf";

    fn write_index(dir: &TempDir, pool: u64, tensors: usize, rows: &[&str]) -> String {
        let mut text = String::new();
        text.push_str(INDEX_HEADER);
        text.push('\n');
        text.push_str(&format!("# align 64 pool {pool} tensors {tensors}\n"));
        for row in rows {
            text.push_str(row);
            text.push('\n');
        }
        dir.write("index.txt", text.as_bytes());
        text
    }

    fn write_native(dir: &TempDir, n_expert: u64, total: u64, rows: &[&str]) {
        let mut text = format!(
            "# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard] (n_expert {n_expert}, total {total}; absolute offsets in m-00001-of-00002.gguf)\n"
        );
        for row in rows {
            text.push_str(row);
            text.push('\n');
        }
        dir.write("native_experts.txt", text.as_bytes());
    }

    fn basic_pack() -> TempDir {
        let dir = TempDir::new("pack-basic");
        dir.write("dense.bin", &[0xAB; 16]);
        write_index(
            &dir,
            16,
            2,
            &[
                "dense_a.weight 0 2 0 16 0 16 4 0 8 0 32 0 0 0 0 0 0 0",
                "native_b.weight 0 0 0 0 16 0 32 0 8 0 32 0 0 0 0 0 0 0",
            ],
        );
        // No experts in this fixture: an empty (but valid) table.
        write_native(&dir, 2, 384, &["0 30 30 0 192 0 0 0"]);
        dir.write("experts.bin", &[0u8; 384]);
        dir.write(
            "conversions.json",
            br#"{"schema":1,"tool":"tools/iq_pack.py","compat_bf16":false,
                "source_shards":[{"name":"m-00001-of-00002.gguf","size":47039860096}],
                "tensors":[]}"#,
        );
        dir.write(
            "experts.bin.src.json",
            br#"{"schema":1,"shards":[{"name":"m-00001-of-00002.gguf","size":47039860096}],
                "native_experts_sha256":"ae28cf7a","bytes":384}"#,
        );
        dir
    }

    #[test]
    fn parses_index_rows_and_provenance() {
        let dir = basic_pack();
        let pack = Pack::open(dir.path()).unwrap();
        assert_eq!(pack.header().version, 3);
        assert_eq!(pack.header().align, 64);
        assert_eq!(pack.header().pool_bytes, 16);
        assert_eq!(pack.header().tensors, 2);
        assert_eq!(
            pack.header().source.as_deref(),
            Some("m-00001-of-00002.gguf")
        );
        assert_eq!(pack.rows().len(), 2);

        let dense = pack.row("dense_a.weight").unwrap();
        assert_eq!(dense.kind, PackKind::F32);
        assert_eq!(dense.ne0, 4);
        assert_eq!(dense.kind.ggml_type(), Some(GgmlType::F32));
        assert_eq!(pack.dense_bytes("dense_a.weight").unwrap(), &[0xAB; 16]);

        let native = pack.row("native_b.weight").unwrap();
        assert_eq!(native.kind, PackKind::Native);
        assert_eq!(native.src_bytes, 0);
        assert_eq!(native.dst_bytes, 0);
        assert!(matches!(
            pack.dense_bytes("native_b.weight"),
            Err(FormatError::Malformed { .. })
        ));
        assert!(matches!(
            pack.tensor_view("native_b.weight", None),
            Err(FormatError::Malformed { .. })
        ));
        assert!(matches!(
            pack.dense_bytes("missing.weight"),
            Err(FormatError::UnknownTensor { .. })
        ));

        // Dense rows resolve without any GGUF.
        match pack.tensor_view("dense_a.weight", None).unwrap() {
            PackTensor::Dense(bytes) => assert_eq!(bytes, &[0xAB; 16]),
            other => panic!("expected dense bytes, got {other:?}"),
        }
        assert_eq!(
            pack.tensor_view("dense_a.weight", None).unwrap().source(),
            "dense.bin"
        );

        let conversions = pack.conversions().unwrap();
        assert_eq!(conversions.schema, 1);
        assert!(conversions.tensors.is_empty());
        assert_eq!(conversions.source_shards[0].size, 47_039_860_096);
        let manifest = pack.source_manifest().unwrap();
        assert_eq!(manifest.bytes, 384);
        assert_eq!(manifest.native_experts_sha256.as_deref(), Some("ae28cf7a"));
    }

    #[test]
    fn native_rows_resolve_through_the_gguf() {
        let dir = basic_pack();
        let gguf = dir.join("m-00001-of-00002.gguf");
        Builder::new()
            .quant_tensor("dense_a.weight", &[4], GgmlType::F32)
            .quant_tensor("native_b.weight", &[32], GgmlType::F32)
            .write(&gguf);
        let shards = ShardSet::open(&gguf).unwrap();
        let pack = Pack::open(dir.path()).unwrap();
        match pack.tensor_view("native_b.weight", Some(&shards)).unwrap() {
            PackTensor::Native(t) => {
                assert_eq!(t.name(), "native_b.weight");
                assert_eq!(t.n_bytes(), 128);
            }
            other => panic!("expected native view, got {other:?}"),
        }
        assert_eq!(
            pack.tensor_view("native_b.weight", Some(&shards))
                .unwrap()
                .source(),
            "gguf"
        );
    }

    #[test]
    fn expert_blobs_are_contiguous_and_distinct() {
        let dir = TempDir::new("pack-experts");
        // kind 4 rows are required only for the arena; keep it minimal.
        dir.write("dense.bin", &bytes_of_len(1024));
        write_index(
            &dir,
            1024,
            1,
            &["f.weight 0 4 0 1024 0 1024 1024 0 0 0 0 0 0 0 0 0 0 0"],
        );
        write_native(&dir, 2, 384, &["0 30 30 0 192 0 0 0"]);
        let mut experts = vec![0u8; 384];
        experts[..192].fill(0x11);
        experts[192..].fill(0x22);
        dir.write("experts.bin", &experts);

        let pack = Pack::open(dir.path()).unwrap();
        assert_eq!(pack.native().n_expert, 2);
        assert_eq!(pack.native().total_bytes, 384);
        assert_eq!(pack.native().layer(0).unwrap().blob_bytes, 192);
        assert_eq!(pack.expert_blob(0, 0).unwrap(), &[0x11u8; 192][..]);
        assert_eq!(pack.expert_blob(0, 1).unwrap(), &[0x22u8; 192][..]);
        assert!(matches!(
            pack.expert_blob(0, 2),
            Err(FormatError::ExpertOutOfRange { .. })
        ));
        assert!(matches!(
            pack.expert_blob(9, 0),
            Err(FormatError::Malformed { .. })
        ));

        // BF16 roles: 4×8 elements = 64 B each, 192 B blob.
        let slices = pack.expert_slices(0, 1, 4, 8).unwrap();
        assert_eq!(slices[0], &[0x22u8; 64][..]);
        assert_eq!(slices[2], &[0x22u8; 64][..]);
        // Wrong model dims must not silently slice.
        assert!(matches!(
            pack.expert_slices(0, 0, 2560, 640),
            Err(FormatError::Malformed { .. })
        ));
    }

    #[test]
    fn malformed_rows_and_arena_failures_are_typed() {
        // 18 fields.
        let dir = TempDir::new("pack-18fields");
        write_index(
            &dir,
            0,
            1,
            &["a.weight 0 2 0 16 0 16 4 0 8 0 32 0 0 0 0 0 0"],
        );
        assert!(matches!(
            Pack::open(dir.path()),
            Err(FormatError::Malformed { .. })
        ));

        // 20 fields.
        let dir = TempDir::new("pack-20fields");
        write_index(
            &dir,
            0,
            1,
            &["a.weight 0 2 0 16 0 16 4 0 8 0 32 0 0 0 0 0 0 0 7"],
        );
        assert!(matches!(
            Pack::open(dir.path()),
            Err(FormatError::Malformed { .. })
        ));

        // Unknown kind.
        let dir = TempDir::new("pack-kind");
        write_index(
            &dir,
            0,
            1,
            &["a.weight 0 9 0 0 0 0 4 0 8 0 32 0 0 0 0 0 0 0"],
        );
        assert!(matches!(
            Pack::open(dir.path()),
            Err(FormatError::Malformed { .. })
        ));

        // pool/dense.bin mismatch.
        let dir = TempDir::new("pack-pool");
        dir.write("dense.bin", &[0u8; 8]);
        write_index(
            &dir,
            16,
            1,
            &["a.weight 0 2 0 16 0 16 4 0 8 0 32 0 0 0 0 0 0 0"],
        );
        write_native(&dir, 2, 384, &["0 30 30 0 192 0 0 0"]);
        dir.write("experts.bin", &[0u8; 384]);
        assert!(matches!(
            Pack::open(dir.path()),
            Err(FormatError::Malformed { .. })
        ));

        // Native row carrying bytes.
        let dir = TempDir::new("pack-native-bytes");
        write_index(
            &dir,
            0,
            1,
            &["a.weight 0 0 0 16 0 0 4 0 8 0 32 0 0 0 0 0 0 0"],
        );
        write_native(&dir, 2, 384, &["0 30 30 0 192 0 0 0"]);
        dir.write("experts.bin", &[0u8; 384]);
        assert!(matches!(
            Pack::open(dir.path()),
            Err(FormatError::Malformed { .. })
        ));

        // Duplicate row.
        let dir = TempDir::new("pack-dup");
        write_index(
            &dir,
            0,
            2,
            &[
                "a.weight 0 0 0 0 0 0 4 0 8 0 32 0 0 0 0 0 0 0",
                "a.weight 0 0 0 0 0 0 4 0 8 0 32 0 0 0 0 0 0 0",
            ],
        );
        write_native(&dir, 2, 384, &["0 30 30 0 192 0 0 0"]);
        dir.write("experts.bin", &[0u8; 384]);
        assert!(matches!(
            Pack::open(dir.path()),
            Err(FormatError::Malformed { .. })
        ));
    }

    #[test]
    fn expert_offset_chain_is_enforced() {
        let dir = TempDir::new("pack-chain");
        write_index(&dir, 0, 0, &[]);
        write_native(
            &dir,
            2,
            384,
            &["0 30 30 0 192 0 0 0", "1 30 30 99 192 0 0 0"],
        );
        dir.write("experts.bin", &[0u8; 384]);
        let err = Pack::open(dir.path()).unwrap_err();
        assert!(
            matches!(err, FormatError::Malformed { .. })
                && err.to_string().contains("contiguous expert-major layout"),
            "unexpected error: {err}"
        );

        let dir = TempDir::new("pack-chain-ok");
        write_index(&dir, 0, 0, &[]);
        write_native(
            &dir,
            2,
            768,
            &["0 30 30 0 192 0 0 0", "1 30 30 384 192 0 0 0"],
        );
        dir.write("experts.bin", &[0u8; 768]);
        let pack = Pack::open(dir.path()).unwrap();
        assert_eq!(pack.native().layers.len(), 2);
        assert_eq!(pack.native().computed_bytes(), 768);
    }

    #[test]
    fn v4_native_rows_may_name_a_shard() {
        let dir = TempDir::new("pack-v4");
        write_index(&dir, 0, 0, &[]);
        write_native(&dir, 2, 384, &["0 30 30 0 192 0 0 0 1"]);
        dir.write("experts.bin", &[0u8; 384]);
        let pack = Pack::open(dir.path()).unwrap();
        assert_eq!(pack.native().layer(0).unwrap().shard, Some(1));
    }

    #[test]
    fn role_bytes_follow_documented_block_geometry() {
        // Real layer 0 numbers from native_experts.txt: IQ2_XS gate/up + IQ4_NL down.
        let layer = NativeLayer {
            layer: 0,
            gate_up_type: GgmlType::IQ2_XS,
            down_type: GgmlType::IQ4_NL,
            offset: 0,
            blob_bytes: 1_868_800,
            gate_off: 0,
            up_off: 0,
            down_off: 0,
            shard: None,
        };
        assert_eq!(
            layer.role_bytes(2560, 640).unwrap(),
            [473_600, 473_600, 921_600]
        );
        assert_eq!(
            layer.role_bytes(2560, 640).unwrap().iter().sum::<u64>(),
            layer.blob_bytes
        );
    }

    fn bytes_of_len(n: usize) -> Vec<u8> {
        (0..n).map(|i| i as u8).collect()
    }
}
