//! MTP draft-layer runtime artifacts (`/srv/strata-data/mtp/rt/`).
//!
//! Produced by `tools/mtp_rt.py`, consumed by `strata/src/core/mtp.cpp`:
//!
//! * `dense.txt` — `name kind rows cols offset bytes`, kinds `q8_0|bf16|f32`,
//!   offsets absolute inside `dense.bin`, contiguous, no padding;
//! * `dense.bin` — 116,099,072 B of packed rows (RMSNorm weights already carry
//!   the Gemma `+1`);
//! * `experts.bin` — 512 × `BLOB` with the *engine* plane-split layout
//!   (`gu codes | down codes | gu scales | down scales`), `BLOB` = 1,382,400 B;
//! * `draft_vocab.bin` — bare little-endian `i32` token ids (106,299 here).

use crate::error::{FormatError, Result};
use crate::filemap::{map_optional, map_readonly};
use crate::quant::GgmlType;
use memmap2::Mmap;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Storage kind of one `dense.txt` row.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MtpDenseKind {
    Q8_0,
    Bf16,
    F32,
}

impl MtpDenseKind {
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "q8_0" => Self::Q8_0,
            "bf16" => Self::Bf16,
            "f32" => Self::F32,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Q8_0 => "q8_0",
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
        }
    }

    pub fn ggml_type(self) -> GgmlType {
        match self {
            Self::Q8_0 => GgmlType::Q8_0,
            Self::Bf16 => GgmlType::BF16,
            Self::F32 => GgmlType::F32,
        }
    }
}

impl std::fmt::Display for MtpDenseKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// One `dense.txt` row (name already stripped of `mtp.` / `mtp.layers.0.`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MtpDenseRow {
    pub name: String,
    pub kind: MtpDenseKind,
    pub rows: u64,
    pub cols: u64,
    pub offset: u64,
    pub bytes: u64,
}

impl MtpDenseRow {
    /// Elements in the row.
    pub fn elements(&self) -> u64 {
        self.rows * self.cols
    }
}

/// Draft-layer geometry needed to slice `experts.bin`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MtpGeometry {
    pub n_embd: u64,
    pub n_ff: u64,
    pub n_expert: u64,
}

impl Default for MtpGeometry {
    fn default() -> Self {
        // Qwen3.8-Flash-Next draft layer (inventory §5.2).
        Self {
            n_embd: 2560,
            n_ff: 640,
            n_expert: 512,
        }
    }
}

impl MtpGeometry {
    /// Bytes per expert blob: `3 × (n_embd · n_ff · 18 / 64)` = 1,382,400 here.
    pub fn blob_bytes(&self) -> Result<u64> {
        let per_projection = GgmlType::Q2_0
            .checked_bytes(self.n_embd * self.n_ff)
            .ok_or_else(|| {
                FormatError::malformed(
                    "mtp geometry",
                    format!(
                        "{} × {} elements are not {} blocks",
                        self.n_embd,
                        self.n_ff,
                        GgmlType::Q2_0
                    ),
                )
            })?;
        Ok(3 * per_projection)
    }
}

/// `mtp/rt/` mapped read-only.
pub struct MtpRt {
    dir: PathBuf,
    geometry: MtpGeometry,
    rows: Vec<MtpDenseRow>,
    index: HashMap<String, usize>,
    dense: Mmap,
    experts: Mmap,
    draft_vocab: Option<Mmap>,
}

impl MtpRt {
    /// Open `dir` with the default draft geometry, then validate every
    /// file-size and offset invariant.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_geometry(dir, MtpGeometry::default())
    }

    pub fn open_with_geometry(dir: impl AsRef<Path>, geometry: MtpGeometry) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let rows = parse_dense_txt(&dir)?;
        let dense = map_readonly(&dir.join("dense.bin"))?;
        let experts = map_readonly(&dir.join("experts.bin"))?;
        let draft_vocab = map_optional(&dir.join("draft_vocab.bin"))?;

        let mut index = HashMap::with_capacity(rows.len());
        for (i, row) in rows.iter().enumerate() {
            if index.insert(row.name.clone(), i).is_some() {
                return Err(FormatError::malformed(
                    dir.join("dense.txt").display().to_string(),
                    format!("duplicate row for '{}'", row.name),
                ));
            }
        }

        // Rows must tile dense.bin exactly, in order, with exact byte sizes.
        let mut cursor = 0u64;
        for row in &rows {
            if row.offset != cursor {
                return Err(FormatError::malformed(
                    dir.join("dense.txt").display().to_string(),
                    format!(
                        "'{}' offset {} should be {cursor} (rows must be contiguous)",
                        row.name, row.offset
                    ),
                ));
            }
            let expect = row
                .kind
                .ggml_type()
                .checked_bytes(row.elements())
                .ok_or_else(|| {
                    FormatError::malformed(
                        dir.join("dense.txt").display().to_string(),
                        format!(
                            "'{}': {} elements are not {}-block aligned",
                            row.name,
                            row.elements(),
                            row.kind.ggml_type()
                        ),
                    )
                })?;
            if row.bytes != expect {
                return Err(FormatError::malformed(
                    dir.join("dense.txt").display().to_string(),
                    format!(
                        "'{}': {} bytes declared, {expect} implied by {} {}×{}",
                        row.name, row.bytes, row.kind, row.rows, row.cols
                    ),
                ));
            }
            cursor = row.offset + row.bytes;
        }
        if cursor != dense.len() as u64 {
            return Err(FormatError::malformed(
                dir.join("dense.bin").display().to_string(),
                format!("rows cover {cursor} bytes but the file is {}", dense.len()),
            ));
        }

        let blob_bytes = geometry.blob_bytes()?;
        let expected_experts = blob_bytes * geometry.n_expert;
        if experts.len() as u64 != expected_experts {
            return Err(FormatError::malformed(
                dir.join("experts.bin").display().to_string(),
                format!(
                    "{} bytes = {expected_experts} implied by {} × {blob_bytes} B blobs",
                    experts.len(),
                    geometry.n_expert
                ),
            ));
        }

        // Cross-check the geometry against the embedding row when present.
        if let Some(i) = index.get("fc_embedding.weight") {
            let row = &rows[*i];
            if row.cols != geometry.n_embd {
                return Err(FormatError::malformed(
                    dir.join("dense.txt").display().to_string(),
                    format!(
                        "fc_embedding.weight is {} columns but geometry says n_embd = {}",
                        row.cols, geometry.n_embd
                    ),
                ));
            }
        }

        let draft_vocab = match draft_vocab {
            Some(map) => {
                if map.len() % 4 != 0 {
                    return Err(FormatError::malformed(
                        dir.join("draft_vocab.bin").display().to_string(),
                        format!("{} bytes is not a whole number of i32 token ids", map.len()),
                    ));
                }
                Some(map)
            }
            None => None,
        };

        Ok(Self {
            dir,
            geometry,
            rows,
            index,
            dense,
            experts,
            draft_vocab,
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn geometry(&self) -> MtpGeometry {
        self.geometry
    }

    pub fn rows(&self) -> &[MtpDenseRow] {
        &self.rows
    }

    pub fn row(&self, name: &str) -> Option<&MtpDenseRow> {
        self.index.get(name).map(|&i| &self.rows[i])
    }

    /// Zero-copy bytes of one dense row.
    pub fn dense_bytes(&self, name: &str) -> Result<&[u8]> {
        let row = self.row(name).ok_or_else(|| FormatError::UnknownTensor {
            name: name.to_string(),
        })?;
        self.dense
            .get(row.offset as usize..(row.offset + row.bytes) as usize)
            .ok_or_else(|| FormatError::malformed(name, "row range exceeds dense.bin".to_string()))
    }

    /// Bytes per expert blob (`3 × gu/down Q2_0 plane`).
    pub fn blob_bytes(&self) -> u64 {
        self.experts.len() as u64 / self.geometry.n_expert.max(1)
    }

    pub fn n_experts(&self) -> usize {
        self.geometry.n_expert as usize
    }

    /// Zero-copy draft expert blob (plane-split layout, see module docs).
    pub fn expert_blob(&self, expert: usize) -> Result<&[u8]> {
        if expert >= self.n_experts() {
            return Err(FormatError::ExpertOutOfRange {
                block: 0,
                expert,
                n_experts: self.n_experts(),
            });
        }
        let per = self.blob_bytes() as usize;
        let start = expert * per;
        self.experts
            .get(start..start + per)
            .ok_or_else(|| FormatError::malformed("mtp experts", "blob out of range"))
    }

    /// Number of allowed draft token ids (`draft_vocab.bin` length / 4).
    pub fn draft_vocab_len(&self) -> Option<usize> {
        self.draft_vocab.as_ref().map(|m| m.len() / 4)
    }

    /// `i32` draft token id at `index`.
    pub fn draft_token(&self, index: usize) -> Option<i32> {
        let map = self.draft_vocab.as_ref()?;
        let at = index.checked_mul(4)?;
        let b = map.get(at..at + 4)?;
        Some(i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Iterate the draft vocabulary (no copy of the mapping).
    pub fn draft_tokens(&self) -> impl Iterator<Item = i32> + '_ {
        (0..self.draft_vocab_len().unwrap_or(0)).filter_map(move |i| self.draft_token(i))
    }

    /// Raw `draft_vocab.bin` bytes.
    pub fn draft_vocab_bytes(&self) -> Option<&[u8]> {
        self.draft_vocab.as_deref()
    }
}

impl std::fmt::Debug for MtpRt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtpRt")
            .field("dir", &self.dir)
            .field("geometry", &self.geometry)
            .field("rows", &self.rows.len())
            .field("dense_bytes", &self.dense.len())
            .field("experts_bytes", &self.experts.len())
            .field("draft_vocab", &self.draft_vocab_len())
            .finish()
    }
}

fn parse_dense_txt(dir: &Path) -> Result<Vec<MtpDenseRow>> {
    let path = dir.join("dense.txt");
    let text = std::fs::read_to_string(&path).map_err(|e| FormatError::io(&path, e))?;
    let mut rows = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let ctx = format!("{}:{}", path.display(), lineno + 1);
        let f: Vec<&str> = trimmed.split_whitespace().collect();
        if f.len() != 6 {
            return Err(FormatError::malformed(
                &ctx,
                format!(
                    "expected 6 fields (name kind rows cols offset bytes), got {}",
                    f.len()
                ),
            ));
        }
        let num = |i: usize, what: &str| -> Result<u64> {
            f[i].parse::<u64>()
                .map_err(|_| FormatError::malformed(&ctx, format!("{what} {:?} is not u64", f[i])))
        };
        let kind = MtpDenseKind::from_name(f[1])
            .ok_or_else(|| FormatError::malformed(&ctx, format!("unknown kind {:?}", f[1])))?;
        rows.push(MtpDenseRow {
            name: f[0].to_string(),
            kind,
            rows: num(2, "rows")?,
            cols: num(3, "cols")?,
            offset: num(4, "offset")?,
            bytes: num(5, "bytes")?,
        });
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;

    // Geometry for tiny fixtures: 64 × 64 Q2_0 plane per projection → 1152 B,
    // blob = 3 × 1152 = 3456 B, 2 experts → 6912 B of `experts.bin`.
    const GEOM: MtpGeometry = MtpGeometry {
        n_embd: 64,
        n_ff: 64,
        n_expert: 2,
    };

    fn fixture(tag: &str) -> TempDir {
        let dir = TempDir::new(tag);
        dir.write(
            "dense.txt",
            b"fc_hidden.weight f32 1 4 0 16\nproj.weight q8_0 1 32 16 34\n",
        );
        let mut dense = vec![0u8; 50];
        dense[..16].fill(0xAA);
        dense[16..].fill(0xBB);
        dir.write("dense.bin", &dense);
        let blob = GEOM.blob_bytes().unwrap() as usize;
        let mut experts = vec![0u8; 2 * blob];
        experts[..blob].fill(0x11);
        experts[blob..].fill(0x22);
        dir.write("experts.bin", &experts);
        let mut vocab = Vec::new();
        for id in [0i32, 1, 248_319] {
            vocab.extend_from_slice(&id.to_le_bytes());
        }
        dir.write("draft_vocab.bin", &vocab);
        dir
    }

    #[test]
    fn parses_dense_rows_experts_and_draft_vocab() {
        let dir = fixture("mtp-basic");
        let mtp = MtpRt::open_with_geometry(dir.path(), GEOM).unwrap();
        assert_eq!(mtp.rows().len(), 2);
        assert_eq!(mtp.geometry(), GEOM);
        assert_eq!(mtp.blob_bytes(), 3456);
        assert_eq!(mtp.n_experts(), 2);

        let row = mtp.row("proj.weight").unwrap();
        assert_eq!(row.kind, MtpDenseKind::Q8_0);
        assert_eq!(row.elements(), 32);
        assert_eq!(
            mtp.dense_bytes("fc_hidden.weight").unwrap(),
            &[0xAAu8; 16][..]
        );
        assert_eq!(mtp.dense_bytes("proj.weight").unwrap(), &[0xBBu8; 34][..]);
        assert!(matches!(
            mtp.dense_bytes("nope"),
            Err(FormatError::UnknownTensor { .. })
        ));

        assert_eq!(mtp.expert_blob(0).unwrap(), &[0x11u8; 3456][..]);
        assert_eq!(mtp.expert_blob(1).unwrap(), &[0x22u8; 3456][..]);
        assert!(matches!(
            mtp.expert_blob(2),
            Err(FormatError::ExpertOutOfRange { .. })
        ));

        assert_eq!(mtp.draft_vocab_len(), Some(3));
        assert_eq!(mtp.draft_token(2), Some(248_319));
        assert_eq!(mtp.draft_token(3), None);
        assert_eq!(mtp.draft_tokens().collect::<Vec<_>>(), vec![0, 1, 248_319]);
        assert_eq!(mtp.draft_vocab_bytes().unwrap().len(), 12);
    }

    #[test]
    fn q8_0_row_size_is_validated() {
        let dir = TempDir::new("mtp-bad-size");
        dir.write("dense.txt", b"x.weight q8_0 1 32 0 32\n"); // 34 B expected
        dir.write("dense.bin", &[0u8; 32]);
        dir.write("experts.bin", &vec![0u8; 2 * 3456]);
        let err = MtpRt::open_with_geometry(dir.path(), GEOM).unwrap_err();
        assert!(err.to_string().contains("34 implied by q8_0"), "{err}");
    }

    #[test]
    fn offsets_must_be_contiguous() {
        let dir = TempDir::new("mtp-gap");
        dir.write(
            "dense.txt",
            b"a.weight f32 1 4 0 16\nb.weight f32 1 4 20 16\n",
        );
        dir.write("dense.bin", &[0u8; 64]);
        dir.write("experts.bin", &vec![0u8; 2 * 3456]);
        let err = MtpRt::open_with_geometry(dir.path(), GEOM).unwrap_err();
        assert!(err.to_string().contains("contiguous"), "{err}");
    }

    #[test]
    fn experts_and_vocab_sizes_are_validated() {
        let dir = fixture("mtp-size");
        dir.write("experts.bin", &vec![0u8; 3456 * 3]);
        assert!(MtpRt::open_with_geometry(dir.path(), GEOM).is_err());

        let dir = fixture("mtp-vocab");
        dir.write("draft_vocab.bin", &[0u8; 6]);
        assert!(MtpRt::open_with_geometry(dir.path(), GEOM).is_err());
    }

    #[test]
    fn unknown_kind_and_bad_field_count_are_errors() {
        let dir = TempDir::new("mtp-kind");
        dir.write("dense.txt", b"x.weight q4_0 1 32 0 34\n");
        dir.write("dense.bin", &[0u8; 34]);
        dir.write("experts.bin", &vec![0u8; 2 * 3456]);
        let err = MtpRt::open_with_geometry(dir.path(), GEOM).unwrap_err();
        assert!(err.to_string().contains("unknown kind"), "{err}");

        let dir = TempDir::new("mtp-fields");
        dir.write("dense.txt", b"x.weight f32 1 4 0\n");
        dir.write("dense.bin", &[0u8; 16]);
        dir.write("experts.bin", &vec![0u8; 2 * 3456]);
        let err = MtpRt::open_with_geometry(dir.path(), GEOM).unwrap_err();
        assert!(err.to_string().contains("6 fields"), "{err}");
    }

    #[test]
    fn draft_vocab_is_optional() {
        let dir = fixture("mtp-novocab");
        std::fs::remove_file(dir.join("draft_vocab.bin")).unwrap();
        let mtp = MtpRt::open_with_geometry(dir.path(), GEOM).unwrap();
        assert_eq!(mtp.draft_vocab_len(), None);
        assert_eq!(mtp.draft_tokens().count(), 0);
    }

    #[test]
    fn default_geometry_matches_the_real_draft_layer() {
        let g = MtpGeometry::default();
        assert_eq!(g.blob_bytes().unwrap(), 1_382_400);
        assert_eq!(g.n_expert, 512);
        // Inventory §5.2: 512 × 1,382,400 = 707,788,800 = rt/experts.bin size.
        assert_eq!(g.blob_bytes().unwrap() * g.n_expert, 707_788_800);
    }
}
