//! Split-residency classification: dense / routed experts / PLE / embedding /
//! output.
//!
//! Naming scheme is the llama.cpp-style one verified in
//! `docs/specs/model-artifacts-inventory.md` §2.4:
//!
//! ```text
//! token_embd.weight                     → Embedding   (shard 1)
//! output.weight                         → Output      (shard 1)
//! per_layer_token_embd.weight           → Ple         (shard 2, [160, 320001536])
//! blk.N.ffn_{gate,up,down}_exps.weight  → Experts     (48 × 512-expert blobs)
//! everything else (attn_*, ssm_*, hc_*, indexer.*, norms, ffn_gate_inp*,
//!   ffn_*_shexp)                        → Dense
//! ```
//!
//! The GGUF stores routed experts **tensor-major**: one `[ne0, ne1, 512]`
//! tensor per role and block, so expert *e*'s bytes are a contiguous slice
//! inside each role tensor. The `[gate|up|down]` blob order only exists
//! contiguously in the strata pack's `experts.bin` — see
//! [`crate::pack::Pack::expert_blob`]. [`ExpertBlock::expert_slices`] gives the
//! three role slices in that same documented order without copying.

use crate::error::{FormatError, Result};
use crate::gguf::TensorView;
use crate::quant::GgmlType;
use crate::shard::ShardSet;
use std::collections::BTreeMap;
use std::ops::Range;

/// Residency group of a tensor.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Group {
    /// Token embedding table (`token_embd.weight`).
    Embedding,
    /// Output head (`output.weight`) — never tied to the embedding here.
    Output,
    /// Everything that stays resident: attention, SSM, hyper-connections,
    /// indexer, norms, routers, shared experts.
    Dense,
    /// Routed expert projections (`blk.N.ffn_{gate,up,down}_exps.weight`).
    Experts,
    /// PLE token table (`per_layer_token_embd.weight`).
    Ple,
}

impl Group {
    /// All groups, in report order.
    pub const ALL: [Group; 5] = [
        Group::Embedding,
        Group::Output,
        Group::Dense,
        Group::Experts,
        Group::Ple,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Group::Embedding => "embedding",
            Group::Output => "output",
            Group::Dense => "dense",
            Group::Experts => "experts",
            Group::Ple => "ple",
        }
    }
}

impl std::fmt::Display for Group {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Role of a routed-expert projection inside one expert blob.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExpertRole {
    Gate,
    Up,
    Down,
}

impl ExpertRole {
    pub const ALL: [ExpertRole; 3] = [ExpertRole::Gate, ExpertRole::Up, ExpertRole::Down];

    /// Position inside the documented `[gate | up | down]` blob layout.
    pub fn index(self) -> usize {
        match self {
            ExpertRole::Gate => 0,
            ExpertRole::Up => 1,
            ExpertRole::Down => 2,
        }
    }

    /// GGUF tensor-name suffix.
    pub fn suffix(self) -> &'static str {
        match self {
            ExpertRole::Gate => "ffn_gate_exps.weight",
            ExpertRole::Up => "ffn_up_exps.weight",
            ExpertRole::Down => "ffn_down_exps.weight",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            ExpertRole::Gate => "gate",
            ExpertRole::Up => "up",
            ExpertRole::Down => "down",
        }
    }
}

/// Classify a GGUF tensor name into its residency group.
pub fn classify(name: &str) -> Group {
    if name == "per_layer_token_embd.weight" {
        Group::Ple
    } else if name == "token_embd.weight" {
        Group::Embedding
    } else if name == "output.weight" {
        Group::Output
    } else if expert_role(name).is_some() {
        Group::Experts
    } else {
        Group::Dense
    }
}

/// Parse `blk.<N>.<suffix>` into `(N, suffix)`.
pub fn split_block_name(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let (index, suffix) = rest.split_once('.')?;
    let block: usize = index.parse().ok()?;
    Some((block, suffix))
}

/// Parse a routed-expert tensor name into `(block, role)`.
pub fn expert_role(name: &str) -> Option<(usize, ExpertRole)> {
    let (block, suffix) = split_block_name(name)?;
    ExpertRole::ALL
        .iter()
        .find(|r| r.suffix() == suffix)
        .map(|&role| (block, role))
}

/// Per-group totals: tensor count, bytes, and quant-type histogram.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GroupSummary {
    pub tensors: usize,
    pub bytes: u64,
    pub types: BTreeMap<GgmlType, usize>,
}

impl GroupSummary {
    fn add(&mut self, tensor: &TensorView<'_>) {
        self.tensors += 1;
        self.bytes += tensor.n_bytes();
        *self.types.entry(tensor.ty()).or_default() += 1;
    }

    /// `"BF16×484 F32×292 …"`, heaviest type first.
    pub fn types_summary(&self) -> String {
        let mut v: Vec<(GgmlType, usize)> = self.types.iter().map(|(t, n)| (*t, *n)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v.iter()
            .map(|(t, n)| format!("{t}×{n}"))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// One block's three routed-expert tensors, with per-expert geometry.
#[derive(Clone, Copy, Debug)]
pub struct ExpertBlock<'a> {
    pub block: usize,
    pub gate: TensorView<'a>,
    pub up: TensorView<'a>,
    pub down: TensorView<'a>,
    /// `dims[2]` — 512 for this model.
    pub n_experts: usize,
}

impl<'a> ExpertBlock<'a> {
    fn new(
        block: usize,
        gate: TensorView<'a>,
        up: TensorView<'a>,
        down: TensorView<'a>,
    ) -> Result<Self> {
        let err = |reason: String| FormatError::ExpertGeometry { block, reason };
        if gate.n_dims() != 3 || up.n_dims() != 3 || down.n_dims() != 3 {
            return Err(err(format!(
                "expected 3-dim [ne0, ne1, n_expert] tensors, got {}, {}, {}",
                gate.n_dims(),
                up.n_dims(),
                down.n_dims()
            )));
        }
        let n_experts = gate.ne(2).unwrap_or(0);
        if n_experts != up.ne(2).unwrap_or(0) || n_experts != down.ne(2).unwrap_or(0) {
            return Err(err(format!(
                "n_expert mismatch: gate {}, up {}, down {}",
                gate.ne(2).unwrap_or(0),
                up.ne(2).unwrap_or(0),
                down.ne(2).unwrap_or(0)
            )));
        }
        if n_experts == 0 {
            return Err(err("n_expert is zero".into()));
        }
        if gate.ne0() != up.ne0() || gate.ne1() != up.ne1() {
            return Err(err(format!(
                "gate {} and up {} must share a shape",
                gate.shape(),
                up.shape()
            )));
        }
        if down.ne0() != gate.ne1() || down.ne1() != gate.ne0() {
            return Err(err(format!(
                "down {} must be the transpose of gate {}",
                down.shape(),
                gate.shape()
            )));
        }
        for role in ExpertRole::ALL {
            let t = match role {
                ExpertRole::Gate => &gate,
                ExpertRole::Up => &up,
                ExpertRole::Down => &down,
            };
            let per_expert_elems = t.ne0() * t.ne1();
            if !t.n_bytes().is_multiple_of(n_experts) {
                return Err(err(format!(
                    "{} bytes {} do not split into {n_experts} experts",
                    role.name(),
                    t.n_bytes()
                )));
            }
            let geometry = t
                .ty()
                .block_geometry()
                .ok_or_else(|| unsupported_type_error(block, t))?;
            if geometry.bytes_for(per_expert_elems).is_none() {
                return Err(err(format!(
                    "{} per-expert {per_expert_elems} elements are not {}-element blocks",
                    role.name(),
                    geometry.elements
                )));
            }
        }
        Ok(Self {
            block,
            gate,
            up,
            down,
            n_experts: n_experts as usize,
        })
    }

    pub fn role(&self, role: ExpertRole) -> &TensorView<'a> {
        match role {
            ExpertRole::Gate => &self.gate,
            ExpertRole::Up => &self.up,
            ExpertRole::Down => &self.down,
        }
    }

    /// Bytes of one role for one expert.
    pub fn bytes_per_expert(&self, role: ExpertRole) -> u64 {
        self.role(role).n_bytes() / self.n_experts as u64
    }

    /// Bytes of one full expert blob (`gate | up | down`).
    pub fn blob_bytes(&self) -> u64 {
        ExpertRole::ALL
            .iter()
            .map(|r| self.bytes_per_expert(*r))
            .sum()
    }

    /// Blob-relative offsets of the three roles: `[0, gate_end, up_end, blob]`.
    pub fn blob_offsets(&self) -> [u64; 4] {
        let gate = self.bytes_per_expert(ExpertRole::Gate);
        let up = self.bytes_per_expert(ExpertRole::Up);
        let down = self.bytes_per_expert(ExpertRole::Down);
        [0, gate, gate + up, gate + up + down]
    }

    /// Absolute file ranges of expert `e`'s three role slices.
    pub fn expert_ranges(&self, expert: usize) -> Result<[Range<u64>; 3]> {
        if expert >= self.n_experts {
            return Err(FormatError::ExpertOutOfRange {
                block: self.block,
                expert,
                n_experts: self.n_experts,
            });
        }
        let mut out: [Range<u64>; 3] = Default::default();
        for (i, role) in ExpertRole::ALL.iter().enumerate() {
            let t = self.role(*role);
            let per = self.bytes_per_expert(*role);
            let start = t.file_range().start + expert as u64 * per;
            out[i] = start..start + per;
        }
        Ok(out)
    }

    /// Expert `e` as three zero-copy slices in `gate | up | down` order.
    pub fn expert_slices(&self, expert: usize) -> Result<ExpertSlices<'a>> {
        if expert >= self.n_experts {
            return Err(FormatError::ExpertOutOfRange {
                block: self.block,
                expert,
                n_experts: self.n_experts,
            });
        }
        let slice = |role: ExpertRole| {
            let per = self.bytes_per_expert(role) as usize;
            let start = expert * per;
            &self.role(role).as_slice()[start..start + per]
        };
        Ok(ExpertSlices {
            gate: slice(ExpertRole::Gate),
            up: slice(ExpertRole::Up),
            down: slice(ExpertRole::Down),
        })
    }

    /// Sum of the three role tensors' bytes.
    pub fn total_bytes(&self) -> u64 {
        self.gate.n_bytes() + self.up.n_bytes() + self.down.n_bytes()
    }

    /// Customary naming for logs: `blk.N.ffn_{gate,up,down}_exps`.
    pub fn label(&self) -> String {
        format!("blk.{}.ffn_*_exps", self.block)
    }
}

fn unsupported_type_error(block: usize, tensor: &TensorView<'_>) -> FormatError {
    FormatError::ExpertGeometry {
        block,
        reason: format!(
            "{} has unsupported ggml type {}",
            tensor.name(),
            tensor.ty()
        ),
    }
}

/// Three zero-copy expert slices, in the documented `[gate | up | down]` order.
#[derive(Clone, Copy, Debug)]
pub struct ExpertSlices<'a> {
    pub gate: &'a [u8],
    pub up: &'a [u8],
    pub down: &'a [u8],
}

impl<'a> ExpertSlices<'a> {
    pub fn iter(&self) -> impl Iterator<Item = &'a [u8]> {
        [self.gate, self.up, self.down].into_iter()
    }

    pub fn len(&self) -> usize {
        self.gate.len() + self.up.len() + self.down.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Group classification of a whole model (or shard set).
#[derive(Debug)]
pub struct ModelLayout<'a> {
    pub summaries: BTreeMap<Group, GroupSummary>,
    pub dense: Vec<TensorView<'a>>,
    pub experts: Vec<ExpertBlock<'a>>,
    pub ple: Option<TensorView<'a>>,
    pub embedding: Option<TensorView<'a>>,
    pub output: Option<TensorView<'a>>,
}

impl<'a> ModelLayout<'a> {
    pub(crate) fn from_shards(shards: &'a ShardSet) -> Result<Self> {
        let mut summaries: BTreeMap<Group, GroupSummary> = BTreeMap::new();
        let mut dense = Vec::new();
        let mut ple: Option<TensorView<'a>> = None;
        let mut embedding: Option<TensorView<'a>> = None;
        let mut output: Option<TensorView<'a>> = None;
        let mut partial: BTreeMap<usize, [Option<TensorView<'a>>; 3]> = BTreeMap::new();

        for tensor in shards.tensors() {
            let group = classify(tensor.name());
            match group {
                Group::Dense => dense.push(tensor),
                Group::Experts => {
                    let (block, role) = expert_role(tensor.name()).ok_or_else(|| {
                        FormatError::malformed(
                            tensor.name(),
                            "classified as experts but unparsable",
                        )
                    })?;
                    partial.entry(block).or_insert([None, None, None])[role.index()] = Some(tensor);
                }
                Group::Ple => {
                    if ple.replace(tensor).is_some() {
                        return Err(FormatError::malformed(
                            tensor.name(),
                            "duplicate per-layer token embedding table",
                        ));
                    }
                }
                Group::Embedding => {
                    if embedding.replace(tensor).is_some() {
                        return Err(FormatError::malformed(
                            tensor.name(),
                            "duplicate token embedding table",
                        ));
                    }
                }
                Group::Output => {
                    if output.replace(tensor).is_some() {
                        return Err(FormatError::malformed(
                            tensor.name(),
                            "duplicate output head",
                        ));
                    }
                }
            }
            summaries.entry(group).or_default().add(&tensor);
        }

        let mut experts = Vec::with_capacity(partial.len());
        for (block, slots) in partial {
            let (gate, up, down) = match (slots[0], slots[1], slots[2]) {
                (Some(gate), Some(up), Some(down)) => (gate, up, down),
                _ => {
                    let missing: Vec<&str> = ExpertRole::ALL
                        .iter()
                        .filter(|r| slots[r.index()].is_none())
                        .map(|r| r.name())
                        .collect();
                    return Err(FormatError::ExpertGeometry {
                        block,
                        reason: format!("missing {}", missing.join(", ")),
                    });
                }
            };
            experts.push(ExpertBlock::new(block, gate, up, down)?);
        }

        Ok(Self {
            summaries,
            dense,
            experts,
            ple,
            embedding,
            output,
        })
    }

    pub fn summary(&self, group: Group) -> Option<&GroupSummary> {
        self.summaries.get(&group)
    }

    pub fn group_bytes(&self, group: Group) -> u64 {
        self.summary(group).map_or(0, |s| s.bytes)
    }

    /// Total bytes across every tensor.
    pub fn total_bytes(&self) -> u64 {
        self.summaries.values().map(|s| s.bytes).sum()
    }

    /// Bytes held by routed experts (must be 42,912,972,800 for this model).
    pub fn expert_bytes(&self) -> u64 {
        self.experts.iter().map(ExpertBlock::total_bytes).sum()
    }

    pub fn block(&self, block: usize) -> Option<&ExpertBlock<'a>> {
        self.experts.iter().find(|b| b.block == block)
    }

    /// Expert blocks in ascending block order.
    pub fn blocks(&self) -> impl Iterator<Item = &ExpertBlock<'a>> {
        self.experts.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::Gguf;
    use crate::testutil::{Builder, TempDir};

    #[test]
    fn classifies_documented_names() {
        let cases = [
            ("token_embd.weight", Group::Embedding),
            ("output.weight", Group::Output),
            ("per_layer_token_embd.weight", Group::Ple),
            ("output_hc_down.weight", Group::Dense),
            ("output_hc_norm.weight", Group::Dense),
            ("blk.0.attn_qkv.weight", Group::Dense),
            ("blk.3.attn_q.weight", Group::Dense),
            ("blk.3.attn_q_norm.weight", Group::Dense),
            ("blk.0.ssm_conv1d.weight", Group::Dense),
            ("blk.0.ssm_dt.bias", Group::Dense),
            ("blk.0.hc_attn_down.weight", Group::Dense),
            ("blk.47.indexer.q_proj.weight", Group::Dense),
            ("blk.0.ffn_gate_inp.weight", Group::Dense),
            ("blk.0.ffn_gate_inp_shexp.weight", Group::Dense),
            ("blk.0.ffn_gate_shexp.weight", Group::Dense),
            ("blk.1.ple_key.weight", Group::Dense),
            ("blk.12.ffn_gate_exps.weight", Group::Experts),
            ("blk.12.ffn_up_exps.weight", Group::Experts),
            ("blk.12.ffn_down_exps.weight", Group::Experts),
        ];
        for (name, group) in cases {
            assert_eq!(classify(name), group, "{name}");
        }
        assert_eq!(
            expert_role("blk.47.ffn_down_exps.weight"),
            Some((47, ExpertRole::Down))
        );
        assert_eq!(expert_role("blk.47.attn_q.weight"), None);
        assert_eq!(
            split_block_name("blk.7.attn_q.weight"),
            Some((7, "attn_q.weight"))
        );
        assert_eq!(split_block_name("output.weight"), None);
    }

    /// Miniature model with the same shape family as the real one.
    fn mini_model(dir: &TempDir) -> crate::shard::ShardSet {
        let path = dir.join("mini.gguf");
        Builder::new()
            .quant_tensor("token_embd.weight", &[32, 8], GgmlType::IQ4_NL)
            .quant_tensor("output.weight", &[32, 8], GgmlType::IQ4_NL)
            .quant_tensor("per_layer_token_embd.weight", &[8, 4], GgmlType::BF16)
            .quant_tensor("blk.0.attn_qkv.weight", &[32, 8], GgmlType::IQ4_NL)
            .quant_tensor("blk.0.ffn_gate_inp.weight", &[32, 4], GgmlType::BF16)
            .quant_tensor("blk.0.ffn_gate_exps.weight", &[32, 64, 4], GgmlType::IQ4_NL)
            .quant_tensor("blk.0.ffn_up_exps.weight", &[32, 64, 4], GgmlType::IQ4_NL)
            .quant_tensor("blk.0.ffn_down_exps.weight", &[64, 32, 4], GgmlType::IQ4_NL)
            .quant_tensor("blk.1.ple_key.weight", &[32, 4], GgmlType::BF16)
            .quant_tensor("blk.1.ffn_gate_exps.weight", &[32, 64, 4], GgmlType::IQ4_NL)
            .quant_tensor("blk.1.ffn_up_exps.weight", &[32, 64, 4], GgmlType::IQ4_NL)
            .quant_tensor("blk.1.ffn_down_exps.weight", &[64, 32, 4], GgmlType::IQ4_NL)
            .write(&path);
        crate::shard::ShardSet::open(&path).unwrap()
    }

    #[test]
    fn groups_and_expert_geometry() {
        let dir = TempDir::new("layout-model");
        let set = mini_model(&dir);
        let layout = set.layout().unwrap();

        assert_eq!(layout.summary(Group::Embedding).unwrap().tensors, 1);
        assert_eq!(layout.summary(Group::Output).unwrap().tensors, 1);
        assert_eq!(layout.summary(Group::Ple).unwrap().tensors, 1);
        assert_eq!(layout.summary(Group::Dense).unwrap().tensors, 3);
        assert_eq!(layout.summary(Group::Experts).unwrap().tensors, 6);
        assert_eq!(layout.total_bytes(), set.total_tensor_bytes());
        // 2 blocks × 4 experts × 3456 B blob
        assert_eq!(layout.expert_bytes(), 2 * 4 * 3456);

        assert_eq!(layout.experts.len(), 2);
        let block = layout.block(0).unwrap();
        assert_eq!(block.n_experts, 4);
        assert_eq!(block.bytes_per_expert(ExpertRole::Gate), 1152);
        assert_eq!(block.bytes_per_expert(ExpertRole::Up), 1152);
        assert_eq!(block.bytes_per_expert(ExpertRole::Down), 1152);
        assert_eq!(block.blob_bytes(), 3456);
        assert_eq!(block.blob_offsets(), [0, 1152, 2304, 3456]);
        assert_eq!(block.total_bytes(), 3 * 1152 * 4);

        let slices = block.expert_slices(2).unwrap();
        assert_eq!(slices.gate.len(), 1152);
        assert_eq!(slices.up.len(), 1152);
        assert_eq!(slices.down.len(), 1152);
        assert_eq!(slices.len(), 3456);
        assert_eq!(slices.iter().count(), 3);

        // Slices are zero-copy windows of the role tensors.
        let gate_start = block.gate.file_range().start + 2 * 1152;
        assert_eq!(
            slices.gate.as_ptr() as u64 - set.primary().bytes().as_ptr() as u64,
            gate_start
        );

        let ranges = block.expert_ranges(2).unwrap();
        assert_eq!(ranges[0].start, gate_start);
        assert_eq!(ranges[1].start, block.up.file_range().start + 2 * 1152);
        assert_eq!(ranges[2].start, block.down.file_range().start + 2 * 1152);
        assert_eq!(ranges[0].end - ranges[0].start, 1152);

        assert!(matches!(
            block.expert_slices(4),
            Err(FormatError::ExpertOutOfRange { .. })
        ));
    }

    #[test]
    fn missing_expert_role_is_an_error() {
        let dir = TempDir::new("layout-missing-role");
        let path = dir.join("m.gguf");
        Builder::new()
            .quant_tensor("blk.0.ffn_gate_exps.weight", &[32, 64, 4], GgmlType::IQ4_NL)
            .quant_tensor("blk.0.ffn_up_exps.weight", &[32, 64, 4], GgmlType::IQ4_NL)
            .write(&path);
        let set = crate::shard::ShardSet::open(&path).unwrap();
        match set.layout() {
            Err(FormatError::ExpertGeometry { block, reason }) => {
                assert_eq!(block, 0);
                assert!(reason.contains("down"), "{reason}");
            }
            other => panic!("expected ExpertGeometry, got {other:?}"),
        }
    }

    #[test]
    fn transposed_down_shape_is_required() {
        let dir = TempDir::new("layout-bad-down");
        let path = dir.join("m.gguf");
        Builder::new()
            .quant_tensor("blk.0.ffn_gate_exps.weight", &[32, 64, 4], GgmlType::IQ4_NL)
            .quant_tensor("blk.0.ffn_up_exps.weight", &[32, 64, 4], GgmlType::IQ4_NL)
            .quant_tensor("blk.0.ffn_down_exps.weight", &[32, 64, 4], GgmlType::IQ4_NL)
            .write(&path);
        let set = crate::shard::ShardSet::open(&path).unwrap();
        assert!(matches!(
            set.layout(),
            Err(FormatError::ExpertGeometry { block: 0, .. })
        ));
    }

    #[test]
    fn quant_histogram_is_reported_per_group() {
        let dir = TempDir::new("layout-histogram");
        let path = dir.join("m.gguf");
        Builder::new()
            .quant_tensor("blk.0.attn_qkv.weight", &[32, 8], GgmlType::IQ4_NL)
            .quant_tensor("blk.0.attn_gate.weight", &[32, 8], GgmlType::Q6_K)
            .write(&path);
        let set = crate::shard::ShardSet::open(&path).unwrap();
        let layout = set.layout().unwrap();
        let dense = layout.summary(Group::Dense).unwrap();
        assert_eq!(dense.types.get(&GgmlType::IQ4_NL), Some(&1));
        assert_eq!(dense.types.get(&GgmlType::Q6_K), Some(&1));
        assert!(dense.types_summary().contains("IQ4_NL×1"));
        let _ = Gguf::open(dir.join("m.gguf")).unwrap();
    }
}
