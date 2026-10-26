//! In-test builders for tiny synthetic artifacts (no fixtures on disk).
//!
//! Fixtures are written to a fresh temp directory and removed on drop; nothing
//! here is used outside `#[cfg(test)]`.

use crate::quant::GgmlType;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Self-cleaning temp directory.
pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    pub fn new(tag: &str) -> Self {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path =
            std::env::temp_dir().join(format!("breccia-format-{tag}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).expect("create temp dir");
        Self { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn join(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }

    pub fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create parent dir");
        }
        std::fs::write(&path, bytes).expect("write fixture");
        path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// A metadata value for [`Builder`].
#[derive(Clone, Debug)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    ArrayU32(Vec<u32>),
    ArrayU64(Vec<u64>),
    ArrayI32(Vec<i32>),
    ArrayF32(Vec<f32>),
    ArrayStr(Vec<String>),
}

impl Value {
    fn type_id(&self) -> u32 {
        match self {
            Value::U8(_) => 0,
            Value::I8(_) => 1,
            Value::U16(_) => 2,
            Value::I16(_) => 3,
            Value::U32(_) => 4,
            Value::I32(_) => 5,
            Value::F32(_) => 6,
            Value::Bool(_) => 7,
            Value::Str(_) => 8,
            Value::ArrayU32(_)
            | Value::ArrayU64(_)
            | Value::ArrayI32(_)
            | Value::ArrayF32(_)
            | Value::ArrayStr(_) => 9,
            Value::U64(_) => 10,
            Value::I64(_) => 11,
            Value::F64(_) => 12,
        }
    }

    fn elem_type(&self) -> u32 {
        match self {
            Value::ArrayU32(_) => 4,
            Value::ArrayU64(_) => 10,
            Value::ArrayI32(_) => 5,
            Value::ArrayF32(_) => 6,
            Value::ArrayStr(_) => 8,
            _ => 0,
        }
    }

    fn write_payload(&self, out: &mut Vec<u8>) {
        match self {
            Value::U8(v) => out.push(*v),
            Value::I8(v) => out.push(*v as u8),
            Value::U16(v) => out.extend_from_slice(&v.to_le_bytes()),
            Value::I16(v) => out.extend_from_slice(&v.to_le_bytes()),
            Value::U32(v) => out.extend_from_slice(&v.to_le_bytes()),
            Value::I32(v) => out.extend_from_slice(&v.to_le_bytes()),
            Value::F32(v) => out.extend_from_slice(&v.to_le_bytes()),
            Value::Bool(v) => out.push(u8::from(*v)),
            Value::Str(s) => write_string(out, s),
            Value::U64(v) => out.extend_from_slice(&v.to_le_bytes()),
            Value::I64(v) => out.extend_from_slice(&v.to_le_bytes()),
            Value::F64(v) => out.extend_from_slice(&v.to_le_bytes()),
            Value::ArrayU32(vs) => {
                out.extend_from_slice(&(vs.len() as u64).to_le_bytes());
                for v in vs {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            Value::ArrayU64(vs) => {
                out.extend_from_slice(&(vs.len() as u64).to_le_bytes());
                for v in vs {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            Value::ArrayI32(vs) => {
                out.extend_from_slice(&(vs.len() as u64).to_le_bytes());
                for v in vs {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            Value::ArrayF32(vs) => {
                out.extend_from_slice(&(vs.len() as u64).to_le_bytes());
                for v in vs {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            Value::ArrayStr(vs) => {
                out.extend_from_slice(&(vs.len() as u64).to_le_bytes());
                for v in vs {
                    write_string(out, v);
                }
            }
        }
    }
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// One tensor to place in the synthetic data section.
#[derive(Clone, Debug)]
pub struct TensorSpec {
    pub name: String,
    pub dims: Vec<u64>,
    pub ty: u32,
    pub bytes: Vec<u8>,
}

/// Tiny GGUF writer: deliberately minimal, only what the tests need.
pub struct Builder {
    pub version: u32,
    pub magic: [u8; 4],
    pub kvs: Vec<(String, Value)>,
    pub tensors: Vec<TensorSpec>,
    /// Extra bytes chopped off the end of the finished file.
    pub truncate: u64,
    /// Skip the trailing alignment padding (legal when there are no tensors).
    pub compact_tail: bool,
}

impl Default for Builder {
    fn default() -> Self {
        Self::new()
    }
}

impl Builder {
    pub fn new() -> Self {
        Self {
            version: 3,
            magic: *b"GGUF",
            kvs: Vec::new(),
            tensors: Vec::new(),
            truncate: 0,
            compact_tail: false,
        }
    }

    pub fn kv(mut self, key: &str, value: Value) -> Self {
        self.kvs.push((key.to_string(), value));
        self
    }

    pub fn tensor(mut self, name: &str, dims: &[u64], ty: u32, bytes: Vec<u8>) -> Self {
        self.tensors.push(TensorSpec {
            name: name.to_string(),
            dims: dims.to_vec(),
            ty,
            bytes,
        });
        self
    }

    /// Tensor whose payload is `elements` zero bytes of `ty`'s geometry.
    pub fn quant_tensor(mut self, name: &str, dims: &[u64], ty: GgmlType) -> Self {
        let elements: u64 = dims.iter().product();
        let bytes = vec![0u8; ty.checked_bytes(elements).expect("aligned fixture") as usize];
        self.tensors.push(TensorSpec {
            name: name.to_string(),
            dims: dims.to_vec(),
            ty: ty.id(),
            bytes,
        });
        self
    }

    pub fn build(&self) -> Vec<u8> {
        let alignment = self
            .kvs
            .iter()
            .find(|(k, _)| k == "general.alignment")
            .and_then(|(_, v)| match v {
                Value::U32(a) => Some(u64::from(*a)),
                Value::U64(a) => Some(*a),
                _ => None,
            })
            .unwrap_or(32);

        let mut out = Vec::new();
        out.extend_from_slice(&self.magic);
        out.extend_from_slice(&self.version.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&(self.kvs.len() as u64).to_le_bytes());
        for (key, value) in &self.kvs {
            write_string(&mut out, key);
            out.extend_from_slice(&value.type_id().to_le_bytes());
            if value.type_id() == 9 {
                out.extend_from_slice(&value.elem_type().to_le_bytes());
            }
            value.write_payload(&mut out);
        }

        // Tensor table (offsets computed on the way).
        let mut offsets = Vec::with_capacity(self.tensors.len());
        let mut data_cursor = 0u64;
        for t in &self.tensors {
            write_string(&mut out, &t.name);
            out.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
            for d in &t.dims {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&t.ty.to_le_bytes());
            out.extend_from_slice(&data_cursor.to_le_bytes());
            offsets.push(data_cursor);
            data_cursor = align_up(data_cursor + t.bytes.len() as u64, alignment);
        }

        if self.compact_tail && self.tensors.is_empty() {
            return out;
        }
        let header = out.len() as u64;
        let data_offset = align_up(header, alignment);
        out.resize(data_offset as usize, 0);
        for (t, offset) in self.tensors.iter().zip(offsets) {
            let at = data_offset + offset;
            out.resize(at as usize, 0);
            out.extend_from_slice(&t.bytes);
            let padded = align_up(out.len() as u64, alignment);
            out.resize(padded as usize, 0);
        }
        if self.truncate > 0 {
            let keep = out.len().saturating_sub(self.truncate as usize);
            out.truncate(keep);
        }
        out
    }

    pub fn write(&self, path: &Path) {
        std::fs::write(path, self.build()).expect("write synthetic gguf");
    }
}

fn align_up(v: u64, a: u64) -> u64 {
    (v + a - 1) & !(a - 1)
}
