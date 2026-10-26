//! GGUF metadata: typed, zero-copy values.
//!
//! Values borrow directly from the mmap'd file. Arrays keep the raw byte span
//! and decode lazily, so the 247,587-entry tokenizer merge list costs nothing
//! unless a caller asks for it.

use crate::error::{FormatError, Result};
use std::collections::HashMap;
use std::fmt;

/// GGUF metadata value type ids (`docs/specs/model-artifacts-inventory.md` §9).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MetaType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    Array,
    U64,
    I64,
    F64,
}

impl MetaType {
    pub fn from_id(id: u32) -> Option<Self> {
        Some(match id {
            0 => Self::U8,
            1 => Self::I8,
            2 => Self::U16,
            3 => Self::I16,
            4 => Self::U32,
            5 => Self::I32,
            6 => Self::F32,
            7 => Self::Bool,
            8 => Self::String,
            9 => Self::Array,
            10 => Self::U64,
            11 => Self::I64,
            12 => Self::F64,
            _ => return None,
        })
    }

    pub fn id(self) -> u32 {
        match self {
            Self::U8 => 0,
            Self::I8 => 1,
            Self::U16 => 2,
            Self::I16 => 3,
            Self::U32 => 4,
            Self::I32 => 5,
            Self::F32 => 6,
            Self::Bool => 7,
            Self::String => 8,
            Self::Array => 9,
            Self::U64 => 10,
            Self::I64 => 11,
            Self::F64 => 12,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::I8 => "i8",
            Self::U16 => "u16",
            Self::I16 => "i16",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::F32 => "f32",
            Self::Bool => "bool",
            Self::String => "string",
            Self::Array => "array",
            Self::U64 => "u64",
            Self::I64 => "i64",
            Self::F64 => "f64",
        }
    }

    /// Byte width of a fixed-size element, `None` for variable-size types.
    pub fn fixed_size(self) -> Option<usize> {
        Some(match self {
            Self::U8 | Self::I8 | Self::Bool => 1,
            Self::U16 | Self::I16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
            Self::String | Self::Array => return None,
        })
    }
}

impl fmt::Display for MetaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A metadata value (borrowed from the mapped file).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum MetadataValue<'a> {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(&'a str),
    Array(MetadataArray<'a>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl<'a> MetadataValue<'a> {
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::U8(_) => "u8",
            Self::I8(_) => "i8",
            Self::U16(_) => "u16",
            Self::I16(_) => "i16",
            Self::U32(_) => "u32",
            Self::I32(_) => "i32",
            Self::F32(_) => "f32",
            Self::Bool(_) => "bool",
            Self::Str(_) => "string",
            Self::Array(_) => "array",
            Self::U64(_) => "u64",
            Self::I64(_) => "i64",
            Self::F64(_) => "f64",
        }
    }

    pub fn as_str(&self) -> Option<&'a str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&MetadataArray<'a>> {
        match self {
            Self::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Any unsigned integer value, widened.
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            Self::U8(v) => u64::from(v),
            Self::U16(v) => u64::from(v),
            Self::U32(v) => u64::from(v),
            Self::U64(v) => v,
            _ => return None,
        })
    }

    /// Any signed integer value, widened (unsigned values must fit).
    pub fn as_i64(&self) -> Option<i64> {
        Some(match *self {
            Self::I8(v) => i64::from(v),
            Self::I16(v) => i64::from(v),
            Self::I32(v) => i64::from(v),
            Self::I64(v) => v,
            Self::U8(v) => i64::from(v),
            Self::U16(v) => i64::from(v),
            Self::U32(v) => i64::from(v),
            Self::U64(v) => i64::try_from(v).ok()?,
            _ => return None,
        })
    }

    /// Float value; integers are accepted and widened.
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            Self::F32(v) => Some(f64::from(v)),
            Self::F64(v) => Some(v),
            other => other.as_i64().map(|v| v as f64),
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match *self {
            Self::Bool(v) => Some(v),
            _ => None,
        }
    }
}

/// A GGUF array: `{u32 elem_type, u64 count, data}` decoded lazily.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MetadataArray<'a> {
    elem_type: MetaType,
    count: usize,
    data: &'a [u8],
}

impl<'a> MetadataArray<'a> {
    pub(crate) fn new(elem_type: MetaType, count: usize, data: &'a [u8]) -> Self {
        Self {
            elem_type,
            count,
            data,
        }
    }

    pub fn elem_type(&self) -> MetaType {
        self.elem_type
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn iter(&self) -> MetadataArrayIter<'a> {
        MetadataArrayIter {
            elem_type: self.elem_type,
            remaining: self.count,
            data: self.data,
        }
    }

    pub fn get(&self, index: usize) -> Option<MetadataValue<'a>> {
        if index >= self.count {
            return None;
        }
        self.iter().nth(index)
    }

    pub fn as_strings(&self) -> Result<Vec<&'a str>> {
        if self.elem_type != MetaType::String {
            return Err(self.type_error("string"));
        }
        self.iter()
            .map(|v| v.as_str().ok_or_else(|| self.type_error("string")))
            .collect()
    }

    pub fn to_u64_vec(&self) -> Result<Vec<u64>> {
        self.iter()
            .map(|v| {
                v.as_u64()
                    .ok_or_else(|| self.type_error("unsigned integer"))
            })
            .collect()
    }

    pub fn to_i64_vec(&self) -> Result<Vec<i64>> {
        self.iter()
            .map(|v| v.as_i64().ok_or_else(|| self.type_error("integer")))
            .collect()
    }

    pub fn to_f64_vec(&self) -> Result<Vec<f64>> {
        self.iter()
            .map(|v| v.as_f64().ok_or_else(|| self.type_error("number")))
            .collect()
    }

    pub fn to_bool_vec(&self) -> Result<Vec<bool>> {
        if self.elem_type != MetaType::Bool {
            return Err(self.type_error("bool"));
        }
        self.iter()
            .map(|v| v.as_bool().ok_or_else(|| self.type_error("bool")))
            .collect()
    }

    fn type_error(&self, expected: &'static str) -> FormatError {
        FormatError::ArrayType {
            expected,
            got: self.elem_type.name(),
        }
    }
}

impl<'a> IntoIterator for &MetadataArray<'a> {
    type Item = MetadataValue<'a>;
    type IntoIter = MetadataArrayIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over array elements. Elements were bounds-checked when the file was
/// opened, so iteration always yields exactly [`MetadataArray::len`] values.
pub struct MetadataArrayIter<'a> {
    elem_type: MetaType,
    remaining: usize,
    data: &'a [u8],
}

impl<'a> Iterator for MetadataArrayIter<'a> {
    type Item = MetadataValue<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let (value, used) = decode_inline(self.elem_type, self.data)?;
        self.remaining -= 1;
        self.data = &self.data[used..];
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<'a> ExactSizeIterator for MetadataArrayIter<'a> {}

/// One `key → value` pair.
#[derive(Clone, Copy, Debug)]
pub struct MetadataEntry<'a> {
    pub key: &'a str,
    pub value: MetadataValue<'a>,
}

/// All KVs of one GGUF, in file order, with O(1) lookup by key.
#[derive(Debug)]
pub struct MetadataMap<'a> {
    entries: Vec<MetadataEntry<'a>>,
    index: HashMap<&'a str, usize>,
}

impl<'a> MetadataMap<'a> {
    pub(crate) fn new(entries: Vec<MetadataEntry<'a>>) -> Self {
        let index = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.key, i))
            .collect();
        Self { entries, index }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &MetadataEntry<'a>> {
        self.entries.iter()
    }

    pub fn keys(&self) -> impl Iterator<Item = &'a str> + '_ {
        self.entries.iter().map(|e| e.key)
    }

    pub fn get(&self, key: &str) -> Option<&MetadataValue<'a>> {
        self.index.get(key).map(|&i| &self.entries[i].value)
    }

    pub fn require(&self, key: &str) -> Result<&MetadataValue<'a>> {
        self.get(key)
            .ok_or_else(|| FormatError::MissingMetadata { key: key.into() })
    }

    pub fn str(&self, key: &str) -> Option<&'a str> {
        self.get(key).and_then(|v| v.as_str())
    }

    pub fn u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(|v| v.as_u64())
    }

    pub fn i64(&self, key: &str) -> Option<i64> {
        self.get(key).and_then(|v| v.as_i64())
    }

    pub fn f64(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(|v| v.as_f64())
    }

    pub fn bool(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(|v| v.as_bool())
    }

    pub fn array(&self, key: &str) -> Option<&MetadataArray<'a>> {
        self.get(key).and_then(|v| v.as_array())
    }

    pub fn require_str(&self, key: &str) -> Result<&'a str> {
        let v = self.require(key)?;
        v.as_str().ok_or_else(|| self.type_err(key, "string", v))
    }

    pub fn require_u32(&self, key: &str) -> Result<u32> {
        let v = self.require(key)?;
        v.as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| self.type_err(key, "u32", v))
    }

    pub fn require_u64(&self, key: &str) -> Result<u64> {
        let v = self.require(key)?;
        v.as_u64().ok_or_else(|| self.type_err(key, "u64", v))
    }

    pub fn require_i64(&self, key: &str) -> Result<i64> {
        let v = self.require(key)?;
        v.as_i64().ok_or_else(|| self.type_err(key, "i64", v))
    }

    pub fn require_f32(&self, key: &str) -> Result<f32> {
        let v = self.require(key)?;
        v.as_f64()
            .map(|n| n as f32)
            .ok_or_else(|| self.type_err(key, "f32", v))
    }

    pub fn require_bool(&self, key: &str) -> Result<bool> {
        let v = self.require(key)?;
        v.as_bool().ok_or_else(|| self.type_err(key, "bool", v))
    }

    pub fn require_array(&self, key: &str) -> Result<&MetadataArray<'a>> {
        let v = self.require(key)?;
        v.as_array().ok_or_else(|| self.type_err(key, "array", v))
    }

    fn type_err(&self, key: &str, expected: &'static str, got: &MetadataValue<'a>) -> FormatError {
        FormatError::MetadataType {
            key: key.into(),
            expected,
            got: got.type_name(),
        }
    }
}

/// Decode one value payload (bytes after the type id).
pub(crate) fn decode_value(type_id: u32, payload: &[u8], offset: u64) -> Result<MetadataValue<'_>> {
    let ty = MetaType::from_id(type_id).ok_or(FormatError::UnknownValueType { offset, type_id })?;
    if ty == MetaType::Array {
        let elem_id = read_u32(payload, 0).ok_or(FormatError::Truncated {
            ctx: "array header",
            offset,
            need: 4,
            available: payload.len() as u64,
        })?;
        let count = read_u64(payload, 4).ok_or(FormatError::Truncated {
            ctx: "array header",
            offset,
            need: 8,
            available: payload.len() as u64,
        })?;
        let elem_type = MetaType::from_id(elem_id).ok_or(FormatError::UnknownValueType {
            offset,
            type_id: elem_id,
        })?;
        if elem_type == MetaType::Array {
            return Err(FormatError::NestedArray { offset });
        }
        let count = usize::try_from(count).map_err(|_| FormatError::NestedArray { offset })?;
        Ok(MetadataValue::Array(MetadataArray::new(
            elem_type,
            count,
            &payload[12..],
        )))
    } else {
        let (value, _) = decode_inline(ty, payload).ok_or(FormatError::Truncated {
            ctx: "metadata value",
            offset,
            need: ty.fixed_size().unwrap_or(8) as u64,
            available: payload.len() as u64,
        })?;
        Ok(value)
    }
}

/// Decode one inline value (no arrays) and report how many bytes it used.
pub(crate) fn decode_inline<'a>(
    ty: MetaType,
    data: &'a [u8],
) -> Option<(MetadataValue<'a>, usize)> {
    Some(match ty {
        MetaType::String => {
            let len = usize::try_from(read_u64(data, 0)?).ok()?;
            let bytes = data.get(8..8 + len)?;
            (
                MetadataValue::Str(std::str::from_utf8(bytes).ok()?),
                8 + len,
            )
        }
        _ => {
            let size = ty.fixed_size()?;
            let bytes = data.get(..size)?;
            (decode_fixed(ty, bytes)?, size)
        }
    })
}

fn decode_fixed(ty: MetaType, b: &[u8]) -> Option<MetadataValue<'static>> {
    Some(match ty {
        MetaType::U8 => MetadataValue::U8(*b.first()?),
        MetaType::I8 => MetadataValue::I8(*b.first()? as i8),
        MetaType::Bool => MetadataValue::Bool(*b.first()? != 0),
        MetaType::U16 => MetadataValue::U16(u16::from_le_bytes(b.get(..2)?.try_into().ok()?)),
        MetaType::I16 => MetadataValue::I16(i16::from_le_bytes(b.get(..2)?.try_into().ok()?)),
        MetaType::U32 => MetadataValue::U32(read_u32(b, 0)?),
        MetaType::I32 => MetadataValue::I32(i32::from_le_bytes(b.get(..4)?.try_into().ok()?)),
        MetaType::F32 => MetadataValue::F32(f32::from_le_bytes(b.get(..4)?.try_into().ok()?)),
        MetaType::U64 => MetadataValue::U64(read_u64(b, 0)?),
        MetaType::I64 => MetadataValue::I64(i64::from_le_bytes(b.get(..8)?.try_into().ok()?)),
        MetaType::F64 => MetadataValue::F64(f64::from_le_bytes(b.get(..8)?.try_into().ok()?)),
        MetaType::String | MetaType::Array => return None,
    })
}

pub(crate) fn read_u32(data: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(data.get(at..at + 4)?.try_into().ok()?))
}

pub(crate) fn read_u64(data: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(data.get(at..at + 8)?.try_into().ok()?))
}
