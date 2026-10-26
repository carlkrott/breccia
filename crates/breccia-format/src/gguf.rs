//! Single-file GGUF v3 reader (mmap, zero-copy).
//!
//! Layout (GGUF v3, `docs/specs/model-artifacts-inventory.md` §9):
//!
//! ```text
//! "GGUF" | u32 version | u64 tensor_count | u64 kv_count
//! kv_count × { u64 len + key | u32 type | value }
//! tensor_count × { u64 len + name | u32 n_dims | u64 dims[n] | u32 type | u64 offset }
//! <pad to general.alignment>          ← data_offset
//! tensor payloads at data_offset + offset
//! ```
//!
//! Only the header pages are touched at open time; tensor bytes stay in the
//! mapping and are handed out as `&[u8]`.

use crate::error::{FormatError, Result};
use crate::metadata::{decode_value, MetaType, MetadataEntry, MetadataMap, MetadataValue};
use crate::quant::GgmlType;
use memmap2::Mmap;
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::path::{Path, PathBuf};

/// GGUF alignment when `general.alignment` is absent.
pub const DEFAULT_ALIGNMENT: u64 = 32;
/// Largest dimension count accepted for a tensor record.
pub const MAX_TENSOR_DIMS: usize = 8;

/// Shard coordinates read from the `split.*` KVs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitRef {
    /// 0-based index of this shard.
    pub no: u32,
    /// Total number of shards.
    pub count: u32,
    /// `split.tensors.count`: tensors across all shards (logs 1224 here).
    pub tensors_count: Option<u64>,
}

struct KvRecord {
    key: Box<str>,
    type_id: u32,
    value_at: usize,
    value_end: usize,
}

struct TensorRecord {
    name: Box<str>,
    n_dims: u8,
    dims: [u64; MAX_TENSOR_DIMS],
    ty: GgmlType,
    /// Declared offset inside the tensor-data section.
    offset: u64,
    abs_start: u64,
    abs_end: u64,
    n_bytes: u64,
}

/// One mmap'd GGUF file (a single shard, or a whole single-file model).
pub struct Gguf {
    path: PathBuf,
    mmap: Mmap,
    version: u32,
    alignment: u64,
    data_offset: u64,
    data_end: u64,
    kv_records: Vec<KvRecord>,
    tensors: Vec<TensorRecord>,
    index: HashMap<Box<str>, usize>,
    split: Option<SplitRef>,
    total_tensor_bytes: u64,
}

impl Gguf {
    /// Map `path` and parse its header, metadata and tensor table.
    ///
    /// The file is memory-mapped read-only; no tensor byte is read here.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path).map_err(|e| FormatError::io(&path, e))?;
        let len = file
            .metadata()
            .map_err(|e| FormatError::io(&path, e))?
            .len();
        if len < 24 {
            return Err(FormatError::Truncated {
                ctx: "GGUF header",
                offset: 0,
                need: 24,
                available: len,
            });
        }
        // SAFETY: the mapping is read-only and outlives every borrow we hand
        // out (it is owned by `Gguf`). A concurrent writer could still mutate
        // the file underneath us; callers must treat model files as immutable,
        // which the phase contracts already require.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| FormatError::io(&path, e))?;
        Self::parse(path, mmap)
    }

    fn parse(path: PathBuf, mmap: Mmap) -> Result<Self> {
        let file_len = mmap.len() as u64;
        let mut cur = Cursor::new(&mmap);

        let mut magic = [0u8; 4];
        magic.copy_from_slice(cur.take(4)?);
        if &magic != b"GGUF" {
            return Err(FormatError::BadMagic { got: magic });
        }
        let version = cur.u32()?;
        if version != 3 {
            return Err(FormatError::UnsupportedVersion(version));
        }

        let tensor_count = usize_of(cur.u64()?, "tensor_count")?;
        let kv_count = usize_of(cur.u64()?, "kv_count")?;
        // Cheap plausibility bounds so a corrupt count cannot trigger a giant
        // allocation: the smallest possible KV is 12 bytes, tensor record 24.
        if kv_count as u64 * 12 > file_len {
            return Err(FormatError::malformed(
                "metadata",
                format!("kv_count {kv_count} cannot fit in {file_len} bytes"),
            ));
        }
        if tensor_count as u64 * 24 > file_len {
            return Err(FormatError::malformed(
                "tensor table",
                format!("tensor_count {tensor_count} cannot fit in {file_len} bytes"),
            ));
        }

        let mut kv_records = Vec::with_capacity(kv_count);
        for _ in 0..kv_count {
            cur.set_ctx("metadata key");
            let key = cur.string()?.to_owned();
            let type_id = cur.u32()?;
            let value_at = cur.pos();
            cur.set_ctx("metadata value");
            skip_value(&mut cur, type_id, value_at)?;
            let value_end = cur.pos();
            kv_records.push(KvRecord {
                key: key.into_boxed_str(),
                type_id,
                value_at,
                value_end,
            });
        }

        let alignment =
            match find_value(&kv_records, &mmap, "general.alignment").and_then(|v| v.as_u64()) {
                Some(a) => a,
                None => DEFAULT_ALIGNMENT,
            };
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(FormatError::BadAlignment(alignment));
        }

        cur.set_ctx("tensor info");
        let mut tensors = Vec::with_capacity(tensor_count);
        let mut index = HashMap::with_capacity(tensor_count);
        let mut total_tensor_bytes = 0u64;
        for _ in 0..tensor_count {
            let name = cur.string()?.to_owned();
            let n_dims = cur.u32()?;
            if n_dims as usize > MAX_TENSOR_DIMS {
                return Err(FormatError::TooManyDims { name, n_dims });
            }
            let mut dims = [0u64; MAX_TENSOR_DIMS];
            for dim in dims.iter_mut().take(n_dims as usize) {
                *dim = cur.u64()?;
            }
            let type_id = cur.u32()?;
            let offset = cur.u64()?;
            let ty = GgmlType::from_id(type_id);
            let shape = &dims[..n_dims as usize];

            let mut elements = 1u64;
            for &d in shape {
                elements = elements.checked_mul(d).ok_or_else(|| {
                    FormatError::malformed(
                        format!("tensor '{name}'"),
                        "element count overflows u64",
                    )
                })?;
            }
            let geometry =
                ty.block_geometry()
                    .ok_or_else(|| FormatError::UnsupportedTensorType {
                        name: name.clone(),
                        type_id,
                    })?;
            let n_bytes =
                geometry
                    .bytes_for(elements)
                    .ok_or_else(|| FormatError::ShapeNotAligned {
                        name: name.clone(),
                        ty,
                        elements,
                        block: geometry.elements,
                    })?;
            total_tensor_bytes += n_bytes;

            let record = TensorRecord {
                name: name.as_str().into(),
                n_dims: n_dims as u8,
                dims,
                ty,
                offset,
                abs_start: 0,
                abs_end: 0,
                n_bytes,
            };
            if index.insert(record.name.clone(), tensors.len()).is_some() {
                return Err(FormatError::DuplicateTensor(name));
            }
            tensors.push(record);
        }

        let header_end = cur.pos() as u64;
        let data_offset = if tensor_count == 0 {
            // GGUF: with no tensors the padding is optional.
            header_end.min(file_len)
        } else {
            align_up(header_end, alignment)
        };
        if data_offset > file_len {
            return Err(FormatError::Truncated {
                ctx: "tensor-data section",
                offset: header_end,
                need: alignment,
                available: file_len - header_end,
            });
        }

        let mut data_end = data_offset;
        for t in &mut tensors {
            let start = t.offset.checked_add(data_offset).ok_or_else(|| {
                FormatError::malformed(format!("tensor '{}'", t.name), "offset overflows u64")
            })?;
            let end = start.checked_add(t.n_bytes).ok_or_else(|| {
                FormatError::malformed(format!("tensor '{}'", t.name), "range overflows u64")
            })?;
            if end > file_len {
                return Err(FormatError::TensorOutOfBounds {
                    name: t.name.to_string(),
                    start,
                    end,
                    file_size: file_len,
                });
            }
            t.abs_start = start;
            t.abs_end = end;
            data_end = data_end.max(end);
        }

        let find_uint = |key: &str| {
            find_value(&kv_records, &mmap, key).and_then(|v| {
                v.as_u64()
                    .or_else(|| v.as_i64().and_then(|i| u64::try_from(i).ok()))
            })
        };
        let split_no = find_uint("split.no");
        let split_count = find_uint("split.count");
        let split_tensors = find_uint("split.tensors.count");
        let split = match (split_no, split_count) {
            (Some(no), Some(count)) => {
                let no = u32::try_from(no).map_err(|_| {
                    FormatError::SplitMismatch(format!("split.no {no} does not fit in u32"))
                })?;
                let count = u32::try_from(count).map_err(|_| {
                    FormatError::SplitMismatch(format!("split.count {count} does not fit in u32"))
                })?;
                Some(SplitRef {
                    no,
                    count,
                    tensors_count: split_tensors,
                })
            }
            (None, None) => None,
            _ => {
                return Err(FormatError::SplitMismatch(format!(
                    "{}: only one of split.no / split.count is present",
                    path.display()
                )))
            }
        };

        Ok(Self {
            path,
            mmap,
            version,
            alignment,
            data_offset,
            data_end,
            kv_records,
            tensors,
            index,
            split,
            total_tensor_bytes,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// GGUF container version (always 3 for files this reader accepts).
    pub fn version(&self) -> u32 {
        self.version
    }

    /// `general.alignment` in bytes (default 32).
    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    /// Byte offset of the tensor-data section (aligned header end).
    pub fn data_offset(&self) -> u64 {
        self.data_offset
    }

    /// End of the last tensor's bytes (equal to the file size in healthy files).
    pub fn data_end(&self) -> u64 {
        self.data_end
    }

    /// Bytes after the last tensor (`0` for the real shards).
    pub fn trailing_bytes(&self) -> u64 {
        (self.mmap.len() as u64).saturating_sub(self.data_end)
    }

    pub fn file_len(&self) -> u64 {
        self.mmap.len() as u64
    }

    /// Number of metadata KV pairs actually present.
    pub fn kv_count(&self) -> usize {
        self.kv_records.len()
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Sum of all tensor payload sizes.
    pub fn total_tensor_bytes(&self) -> u64 {
        self.total_tensor_bytes
    }

    /// Split coordinates, if the file declares any.
    pub fn split(&self) -> Option<SplitRef> {
        self.split
    }

    /// Decode all metadata (values borrow from the mapping).
    pub fn metadata(&self) -> Result<MetadataMap<'_>> {
        let mut entries = Vec::with_capacity(self.kv_records.len());
        for rec in &self.kv_records {
            let value = decode_value(
                rec.type_id,
                &self.mmap[rec.value_at..rec.value_end],
                rec.value_at as u64,
            )?;
            entries.push(MetadataEntry {
                key: rec.key.as_ref(),
                value,
            });
        }
        Ok(MetadataMap::new(entries))
    }

    /// Decode a single metadata value by key without materialising the map.
    pub fn metadata_value(&self, key: &str) -> Option<MetadataValue<'_>> {
        let rec = self.kv_records.iter().find(|r| &*r.key == key)?;
        decode_value(
            rec.type_id,
            &self.mmap[rec.value_at..rec.value_end],
            rec.value_at as u64,
        )
        .ok()
    }

    /// Shortcut for `general.<...>` string values.
    pub fn metadata_str(&self, key: &str) -> Option<&str> {
        self.metadata_value(key)?.as_str()
    }

    /// `general.architecture`, e.g. `qwen4exp`.
    pub fn architecture(&self) -> Option<&str> {
        self.metadata_str("general.architecture")
    }

    pub fn tensors(&self) -> impl Iterator<Item = TensorView<'_>> {
        self.tensors.iter().map(move |t| TensorView {
            name: t.name.as_ref(),
            dims: &t.dims[..t.n_dims as usize],
            ty: t.ty,
            offset: t.offset,
            file_start: t.abs_start,
            file_end: t.abs_end,
            data: &self.mmap[t.abs_start as usize..t.abs_end as usize],
        })
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|t| &*t.name)
    }

    /// Look a tensor up by its exact GGUF name.
    pub fn tensor(&self, name: &str) -> Option<TensorView<'_>> {
        let idx = *self.index.get(name)?;
        self.tensor_at(idx)
    }

    /// Look a tensor up by its position in the tensor table.
    pub fn tensor_by_index(&self, idx: usize) -> Option<TensorView<'_>> {
        self.tensor_at(idx)
    }

    fn tensor_at(&self, idx: usize) -> Option<TensorView<'_>> {
        let t = self.tensors.get(idx)?;
        Some(TensorView {
            name: t.name.as_ref(),
            dims: &t.dims[..t.n_dims as usize],
            ty: t.ty,
            offset: t.offset,
            file_start: t.abs_start,
            file_end: t.abs_end,
            data: &self.mmap[t.abs_start as usize..t.abs_end as usize],
        })
    }

    /// Raw mapping bytes (header included). Mostly useful for hashing/IO.
    pub fn bytes(&self) -> &[u8] {
        &self.mmap
    }
}

impl fmt::Debug for Gguf {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Gguf")
            .field("path", &self.path)
            .field("version", &self.version)
            .field("alignment", &self.alignment)
            .field("kv_count", &self.kv_records.len())
            .field("tensor_count", &self.tensors.len())
            .field("data_offset", &self.data_offset)
            .field("data_end", &self.data_end)
            .field("split", &self.split)
            .finish()
    }
}

/// A borrowed, read-only view of one tensor's bytes.
#[derive(Clone, Copy)]
pub struct TensorView<'a> {
    name: &'a str,
    dims: &'a [u64],
    ty: GgmlType,
    offset: u64,
    file_start: u64,
    file_end: u64,
    data: &'a [u8],
}

impl<'a> TensorView<'a> {
    pub fn name(&self) -> &'a str {
        self.name
    }

    /// GGML (contiguous-first) dimensions: `dims[0]` is the innermost axis.
    pub fn dims(&self) -> &'a [u64] {
        self.dims
    }

    pub fn n_dims(&self) -> usize {
        self.dims.len()
    }

    pub fn ne(&self, axis: usize) -> Option<u64> {
        self.dims.get(axis).copied()
    }

    pub fn ne0(&self) -> u64 {
        self.dims.first().copied().unwrap_or(0)
    }

    pub fn ne1(&self) -> u64 {
        self.dims.get(1).copied().unwrap_or(1)
    }

    pub fn ty(&self) -> GgmlType {
        self.ty
    }

    /// Element count (product of the dimensions).
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }

    /// Payload size in bytes.
    pub fn n_bytes(&self) -> u64 {
        self.data.len() as u64
    }

    /// Declared offset inside the tensor-data section.
    pub fn data_offset(&self) -> u64 {
        self.offset
    }

    /// Absolute byte range inside the mapped file.
    pub fn file_range(&self) -> std::ops::Range<u64> {
        self.file_start..self.file_end
    }

    /// The tensor bytes. Zero-copy: this is a slice of the mmap.
    pub fn as_slice(&self) -> &'a [u8] {
        self.data
    }

    /// Element count of one row (`dims[0]`), useful for addressing raw rows.
    pub fn row_elements(&self) -> u64 {
        self.ne0()
    }

    /// Shape rendered as `[ne0, ne1, ...]`.
    pub fn shape(&self) -> String {
        let inner = self
            .dims
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        format!("[{inner}]")
    }
}

impl fmt::Debug for TensorView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TensorView({} {} {} {} B)",
            self.name,
            self.shape(),
            self.ty,
            self.n_bytes()
        )
    }
}

/// Bounds-checked little-endian cursor over the mapped header.
pub(crate) struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
    ctx: &'static str,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            ctx: "GGUF header",
        }
    }

    fn set_ctx(&mut self, ctx: &'static str) {
        self.ctx = ctx;
    }

    fn pos(&self) -> usize {
        self.pos
    }

    fn take(&mut self, need: usize) -> Result<&'a [u8]> {
        let available = self.data.len().saturating_sub(self.pos);
        if need > available {
            return Err(FormatError::Truncated {
                ctx: self.ctx,
                offset: self.pos as u64,
                need: need as u64,
                available: available as u64,
            });
        }
        let out = &self.data[self.pos..self.pos + need];
        self.pos += need;
        Ok(out)
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn string(&mut self) -> Result<&'a str> {
        let start = self.pos;
        let len = usize_of(self.u64()?, "string length")?;
        let bytes = self.take(len)?;
        std::str::from_utf8(bytes).map_err(|source| FormatError::BadUtf8 {
            ctx: self.ctx,
            offset: start as u64,
            source,
        })
    }
}

fn skip_value(cur: &mut Cursor<'_>, type_id: u32, offset: usize) -> Result<()> {
    let ty = MetaType::from_id(type_id).ok_or(FormatError::UnknownValueType {
        offset: offset as u64,
        type_id,
    })?;
    match ty {
        MetaType::String => {
            let len = usize_of(cur.u64()?, "string length")?;
            cur.take(len)?;
        }
        MetaType::Array => {
            let elem_id = cur.u32()?;
            let count = cur.u64()?;
            let elem = MetaType::from_id(elem_id).ok_or(FormatError::UnknownValueType {
                offset: offset as u64,
                type_id: elem_id,
            })?;
            match elem.fixed_size() {
                Some(size) => {
                    let total = count.checked_mul(size as u64).ok_or_else(|| {
                        FormatError::malformed("metadata array", "element count overflows u64")
                    })?;
                    cur.take(usize_of(total, "array length")?)?;
                }
                None if elem == MetaType::String => {
                    for _ in 0..count {
                        let len = usize_of(cur.u64()?, "string length")?;
                        cur.take(len)?;
                    }
                }
                None => {
                    return Err(FormatError::NestedArray {
                        offset: offset as u64,
                    })
                }
            }
        }
        _ => {
            let size = ty.fixed_size().ok_or_else(|| {
                FormatError::malformed("metadata", format!("type {ty} has no fixed width"))
            })?;
            cur.take(size)?;
        }
    }
    Ok(())
}

fn find_value<'a>(records: &'a [KvRecord], mmap: &'a [u8], key: &str) -> Option<MetadataValue<'a>> {
    let rec = records.iter().find(|r| &*r.key == key)?;
    decode_value(
        rec.type_id,
        &mmap[rec.value_at..rec.value_end],
        rec.value_at as u64,
    )
    .ok()
}

pub(crate) fn align_up(value: u64, alignment: u64) -> u64 {
    let mask = alignment - 1;
    (value + mask) & !mask
}

fn usize_of(value: u64, what: &'static str) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        FormatError::malformed(
            "GGUF header",
            format!("{what} {value} does not fit in usize"),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{Builder, TempDir, Value};

    fn write(dir: &TempDir, name: &str, builder: &Builder) -> PathBuf {
        let path = dir.join(name);
        builder.write(&path);
        path
    }

    #[test]
    fn parses_header_metadata_and_tensor() {
        let dir = TempDir::new("gguf-basic");
        let path = write(
            &dir,
            "m.gguf",
            &Builder::new()
                .kv("general.architecture", Value::Str("qwen4exp".into()))
                .kv("test.u32", Value::U32(48))
                .kv("test.f32", Value::F32(1e-6))
                .kv("test.bool", Value::Bool(false))
                .kv("test.str", Value::Str("hello".into()))
                .tensor("a.weight", &[4, 2], GgmlType::F32.id(), vec![7u8; 32]),
        );
        let g = Gguf::open(&path).unwrap();
        assert_eq!(g.version(), 3);
        assert_eq!(g.tensor_count(), 1);
        assert_eq!(g.kv_count(), 5);
        assert_eq!(g.alignment(), DEFAULT_ALIGNMENT);
        assert_eq!(g.architecture(), Some("qwen4exp"));
        assert_eq!(g.trailing_bytes(), 0);
        assert_eq!(g.data_end(), g.file_len());

        let md = g.metadata().unwrap();
        assert_eq!(md.len(), 5);
        assert_eq!(md.require_str("test.str").unwrap(), "hello");
        assert_eq!(md.require_u32("test.u32").unwrap(), 48);
        assert!((md.require_f32("test.f32").unwrap() - 1e-6).abs() < 1e-12);
        assert!(!md.require_bool("test.bool").unwrap());
        assert_eq!(md.u64("missing.key"), None);
        assert!(matches!(
            md.require_u32("test.str"),
            Err(FormatError::MetadataType { .. })
        ));

        let t = g.tensor("a.weight").unwrap();
        assert_eq!(t.dims(), &[4, 2]);
        assert_eq!(t.ne0(), 4);
        assert_eq!(t.ne1(), 2);
        assert_eq!(t.n_elements(), 8);
        assert_eq!(t.ty(), GgmlType::F32);
        assert_eq!(t.n_bytes(), 32);
        assert_eq!(t.as_slice(), &[7u8; 32][..]);
        assert_eq!(t.shape(), "[4, 2]");
        assert_eq!(t.data_offset(), 0);
        assert_eq!(t.file_range().start, g.data_offset());
        assert_eq!(t.file_range().end, g.file_len());

        // Zero-copy: the slice points inside the mapping.
        let base = g.bytes().as_ptr() as usize;
        let data = t.as_slice().as_ptr() as usize;
        assert!((base..base + g.bytes().len()).contains(&data));
        assert!(g.tensor("nope").is_none());
    }

    #[test]
    fn unsorted_metadata_keys_keep_file_order_and_resolve() {
        let dir = TempDir::new("gguf-unsorted");
        let path = write(
            &dir,
            "m.gguf",
            &Builder::new()
                .kv("zzz.last", Value::U32(1))
                .kv("aaa.first", Value::U32(2))
                .kv("mmm.middle", Value::U32(3))
                .kv("zzz.last", Value::U32(4)),
        );
        let g = Gguf::open(&path).unwrap();
        let md = g.metadata().unwrap();
        let keys: Vec<&str> = md.keys().collect();
        assert_eq!(
            keys,
            vec!["zzz.last", "aaa.first", "mmm.middle", "zzz.last"]
        );
        // Last duplicate wins in the index, order is preserved.
        assert_eq!(md.require_u32("zzz.last").unwrap(), 4);
        assert_eq!(md.require_u32("aaa.first").unwrap(), 2);
    }

    #[test]
    fn every_scalar_metadata_type_round_trips() {
        let dir = TempDir::new("gguf-scalars");
        let path = write(
            &dir,
            "m.gguf",
            &Builder::new()
                .kv("u8", Value::U8(200))
                .kv("i8", Value::I8(-7))
                .kv("u16", Value::U16(65535))
                .kv("i16", Value::I16(-30000))
                .kv("u32", Value::U32(4_000_000_000))
                .kv("i32", Value::I32(-2_000_000_000))
                .kv("f32", Value::F32(-1.5))
                .kv("bool", Value::Bool(true))
                .kv("u64", Value::U64(u64::MAX - 1))
                .kv("i64", Value::I64(i64::MIN + 1))
                .kv("f64", Value::F64(std::f64::consts::PI))
                .kv("f32array", Value::ArrayF32(vec![1.0, 2.5])),
        );
        let g = Gguf::open(&path).unwrap();
        let md = g.metadata().unwrap();
        assert_eq!(md.require_u64("u8").unwrap(), 200);
        assert_eq!(md.require_i64("i8").unwrap(), -7);
        assert_eq!(md.require_u64("u16").unwrap(), 65535);
        assert_eq!(md.require_i64("i16").unwrap(), -30000);
        assert_eq!(md.require_u64("u32").unwrap(), 4_000_000_000);
        assert_eq!(md.require_i64("i32").unwrap(), -2_000_000_000);
        assert_eq!(md.require_f32("f32").unwrap(), -1.5);
        assert!(md.require_bool("bool").unwrap());
        assert_eq!(md.require_u64("u64").unwrap(), u64::MAX - 1);
        assert_eq!(md.require_i64("i64").unwrap(), i64::MIN + 1);
        assert_eq!(md.f64("f64"), Some(std::f64::consts::PI));
        assert_eq!(
            md.require_array("f32array").unwrap().to_f64_vec().unwrap(),
            vec![1.0, 2.5]
        );
        // Numeric widening / narrowing endpoints.
        assert_eq!(md.get("i8").and_then(|v| v.as_f64()), Some(-7.0));
        assert_eq!(md.get("f64").and_then(|v| v.as_u64()), None);
        assert_eq!(md.get("bool").and_then(|v| v.as_i64()), None);
    }

    #[test]
    fn u64_and_string_arrays_decode() {
        let dir = TempDir::new("gguf-arrays");
        let path = write(
            &dir,
            "m.gguf",
            &Builder::new()
                .kv(
                    "ple.layer_multipliers",
                    Value::ArrayU64(vec![23703573157769, 20109073645365, 8052911324071]),
                )
                .kv(
                    "general.tags",
                    Value::ArrayStr(vec!["gsq-rco".into(), "ista".into()]),
                )
                .kv("cmp", Value::ArrayI32(vec![0, 0, 0, 4]))
                .kv("empty", Value::ArrayU32(vec![])),
        );
        let g = Gguf::open(&path).unwrap();
        let md = g.metadata().unwrap();
        let mult = md.require_array("ple.layer_multipliers").unwrap();
        assert_eq!(mult.elem_type(), MetaType::U64);
        assert_eq!(mult.len(), 3);
        assert_eq!(
            mult.to_u64_vec().unwrap(),
            vec![23703573157769, 20109073645365, 8052911324071]
        );
        assert_eq!(mult.get(1).and_then(|v| v.as_u64()), Some(20109073645365));
        assert_eq!(mult.get(3), None);

        let tags = md.require_array("general.tags").unwrap();
        assert_eq!(tags.as_strings().unwrap(), vec!["gsq-rco", "ista"]);
        assert!(matches!(
            tags.to_u64_vec(),
            Err(FormatError::ArrayType { .. })
        ));

        assert_eq!(
            md.require_array("cmp").unwrap().to_i64_vec().unwrap(),
            vec![0, 0, 0, 4]
        );
        assert!(md.require_array("empty").unwrap().is_empty());
    }

    #[test]
    fn alignment_other_than_32_is_honoured() {
        let dir = TempDir::new("gguf-align");
        let path = write(
            &dir,
            "m.gguf",
            &Builder::new()
                .kv("general.alignment", Value::U64(64))
                .tensor("a.weight", &[8], GgmlType::BF16.id(), vec![1u8; 16])
                .tensor("b.weight", &[8], GgmlType::BF16.id(), vec![2u8; 16]),
        );
        let g = Gguf::open(&path).unwrap();
        assert_eq!(g.alignment(), 64);
        assert_eq!(g.data_offset() % 64, 0);
        let a = g.tensor("a.weight").unwrap();
        let b = g.tensor("b.weight").unwrap();
        assert_eq!(a.file_range().start, g.data_offset());
        assert_eq!(b.data_offset() % 64, 0);
        assert_eq!(a.as_slice(), &[1u8; 16][..]);
        assert_eq!(b.as_slice(), &[2u8; 16][..]);
    }

    #[test]
    fn bad_alignment_is_rejected() {
        let dir = TempDir::new("gguf-bad-align");
        let path = write(
            &dir,
            "m.gguf",
            &Builder::new().kv("general.alignment", Value::U32(24)),
        );
        assert!(matches!(
            Gguf::open(&path),
            Err(FormatError::BadAlignment(24))
        ));
    }

    #[test]
    fn truncated_data_is_a_typed_error() {
        let dir = TempDir::new("gguf-trunc");
        let mut builder = Builder::new().tensor("a.weight", &[16], GgmlType::F32.id(), vec![0; 64]);
        builder.truncate = 8;
        let path = write(&dir, "m.gguf", &builder);
        match Gguf::open(&path) {
            Err(FormatError::TensorOutOfBounds { name, .. }) => assert_eq!(name, "a.weight"),
            other => panic!("expected TensorOutOfBounds, got {other:?}"),
        }
    }

    #[test]
    fn truncated_header_and_bad_magic() {
        let dir = TempDir::new("gguf-header");
        let mut builder = Builder::new();
        builder.truncate = 20;
        let short = write(&dir, "short.gguf", &builder);
        assert!(matches!(
            Gguf::open(&short),
            Err(FormatError::Truncated { .. })
        ));

        let mut builder = Builder::new().kv("k", Value::U32(1));
        builder.magic = *b"GGUX";
        let bad = write(&dir, "bad.gguf", &builder);
        assert!(matches!(
            Gguf::open(&bad),
            Err(FormatError::BadMagic { got }) if &got == b"GGUX"
        ));

        let mut builder = Builder::new().kv("k", Value::U32(1));
        builder.version = 2;
        let v2 = write(&dir, "v2.gguf", &builder);
        assert!(matches!(
            Gguf::open(&v2),
            Err(FormatError::UnsupportedVersion(2))
        ));
    }

    #[test]
    fn unknown_type_and_unaligned_shape_are_errors() {
        let dir = TempDir::new("gguf-types");
        let unknown = write(
            &dir,
            "unknown.gguf",
            &Builder::new().tensor("x.weight", &[4], 999, vec![0; 16]),
        );
        assert!(matches!(
            Gguf::open(&unknown),
            Err(FormatError::UnsupportedTensorType { type_id: 999, .. })
        ));

        let unaligned = write(
            &dir,
            "unaligned.gguf",
            &Builder::new().tensor("y.weight", &[100], GgmlType::IQ4_NL.id(), vec![0; 56]),
        );
        assert!(matches!(
            Gguf::open(&unaligned),
            Err(FormatError::ShapeNotAligned { elements: 100, .. })
        ));

        let dims = write(
            &dir,
            "dims.gguf",
            &Builder::new().tensor(
                "z.weight",
                &[2, 2, 2, 2, 2, 2, 2, 2, 2],
                GgmlType::F32.id(),
                vec![0; 4],
            ),
        );
        assert!(matches!(
            Gguf::open(&dims),
            Err(FormatError::TooManyDims { n_dims: 9, .. })
        ));
    }

    #[test]
    fn zero_tensor_file_has_no_data_section() {
        let dir = TempDir::new("gguf-empty");
        let mut builder = Builder::new()
            .kv("general.architecture", Value::Str("clip".into()))
            .kv("empty.array", Value::ArrayStr(vec![]));
        builder.compact_tail = true;
        let path = write(&dir, "m.gguf", &builder);
        let g = Gguf::open(&path).unwrap();
        assert_eq!(g.tensor_count(), 0);
        assert_eq!(g.data_offset(), g.file_len());
        assert_eq!(g.data_end(), g.file_len());
        assert_eq!(g.total_tensor_bytes(), 0);
        assert_eq!(g.tensors().count(), 0);
    }

    #[test]
    fn zero_size_tensor_is_allowed() {
        let dir = TempDir::new("gguf-zero");
        let path = write(
            &dir,
            "m.gguf",
            &Builder::new()
                .tensor("zero.weight", &[0, 4], GgmlType::F32.id(), vec![])
                .tensor("next.weight", &[4], GgmlType::F32.id(), vec![9u8; 16]),
        );
        let g = Gguf::open(&path).unwrap();
        let zero = g.tensor("zero.weight").unwrap();
        assert_eq!(zero.n_bytes(), 0);
        assert!(zero.as_slice().is_empty());
        assert_eq!(zero.n_elements(), 0);
        let next = g.tensor("next.weight").unwrap();
        assert_eq!(next.as_slice(), &[9u8; 16][..]);
        assert_eq!(g.total_tensor_bytes(), 16);
    }

    #[test]
    fn duplicate_tensor_names_are_rejected() {
        let dir = TempDir::new("gguf-dup");
        let path = write(
            &dir,
            "m.gguf",
            &Builder::new()
                .tensor("a", &[4], GgmlType::F32.id(), vec![0; 16])
                .tensor("a", &[4], GgmlType::F32.id(), vec![0; 16]),
        );
        assert!(matches!(
            Gguf::open(&path),
            Err(FormatError::DuplicateTensor(name)) if name == "a"
        ));
    }

    #[test]
    fn split_keys_are_parsed() {
        let dir = TempDir::new("gguf-split");
        let path = write(
            &dir,
            "m.gguf",
            &Builder::new()
                .kv("split.no", Value::U16(1))
                .kv("split.count", Value::U16(2))
                .kv("split.tensors.count", Value::I32(1224)),
        );
        let g = Gguf::open(&path).unwrap();
        assert_eq!(
            g.split(),
            Some(SplitRef {
                no: 1,
                count: 2,
                tensors_count: Some(1224)
            })
        );

        let half = write(
            &dir,
            "half.gguf",
            &Builder::new().kv("split.no", Value::U16(0)),
        );
        assert!(matches!(
            Gguf::open(&half),
            Err(FormatError::SplitMismatch(_))
        ));
    }
}
