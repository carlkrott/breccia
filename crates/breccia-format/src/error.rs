//! Typed errors. Every parser path returns one of these — no panics on
//! malformed input, no `unwrap` on file data.

use crate::quant::GgmlType;
use std::path::PathBuf;
use thiserror::Error;

/// Result alias for every fallible operation in this crate.
pub type Result<T> = std::result::Result<T, FormatError>;

/// Anything that can go wrong while reading a model artifact.
#[derive(Debug, Error)]
pub enum FormatError {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("not a GGUF file: magic {got:?} (expected \"GGUF\")")]
    BadMagic { got: [u8; 4] },

    #[error("unsupported GGUF version {0} (this reader implements v3)")]
    UnsupportedVersion(u32),

    #[error("truncated {ctx} at byte {offset}: need {need} more bytes, {available} available")]
    Truncated {
        ctx: &'static str,
        offset: u64,
        need: u64,
        available: u64,
    },

    #[error("invalid UTF-8 in {ctx} at byte {offset}")]
    BadUtf8 {
        ctx: &'static str,
        offset: u64,
        #[source]
        source: std::str::Utf8Error,
    },

    #[error("unknown metadata value type {type_id} at byte {offset}")]
    UnknownValueType { offset: u64, type_id: u32 },

    #[error("array at byte {offset} has array elements (nested arrays are not supported)")]
    NestedArray { offset: u64 },

    #[error("invalid tensor-data alignment {0}: must be a non-zero power of two")]
    BadAlignment(u64),

    #[error("tensor '{name}': {n_dims} dimensions exceed the supported maximum of 8")]
    TooManyDims { name: String, n_dims: u32 },

    #[error("tensor '{name}': unsupported ggml type id {type_id}")]
    UnsupportedTensorType { name: String, type_id: u32 },

    #[error(
        "tensor '{name}': {elements} elements are not a multiple of the {block}-element block size"
    )]
    ShapeNotAligned {
        name: String,
        ty: GgmlType,
        elements: u64,
        block: u64,
    },

    #[error("tensor '{name}' spans bytes {start:#x}..{end:#x}, past the end of the {file_size:#x}-byte file")]
    TensorOutOfBounds {
        name: String,
        start: u64,
        end: u64,
        file_size: u64,
    },

    #[error("duplicate tensor name '{0}'")]
    DuplicateTensor(String),

    #[error("shard split inconsistency: {0}")]
    SplitMismatch(String),

    #[error("missing metadata key '{key}'")]
    MissingMetadata { key: String },

    #[error("metadata '{key}' has type {got}, expected {expected}")]
    MetadataType {
        key: String,
        expected: &'static str,
        got: &'static str,
    },

    #[error("array has element type {got}, expected {expected}")]
    ArrayType {
        expected: &'static str,
        got: &'static str,
    },

    #[error("'{name}' is not a tensor of this artifact set")]
    UnknownTensor { name: String },

    #[error("block {block}: expert {expert} out of range (0..{n_experts})")]
    ExpertOutOfRange {
        block: usize,
        expert: usize,
        n_experts: usize,
    },

    #[error("block {block}: inconsistent routed-expert geometry: {reason}")]
    ExpertGeometry { block: usize, reason: String },

    #[error("{context}: {reason}")]
    Malformed { context: String, reason: String },
}

impl FormatError {
    pub(crate) fn malformed(context: impl Into<String>, reason: impl Into<String>) -> Self {
        FormatError::Malformed {
            context: context.into(),
            reason: reason.into(),
        }
    }

    pub(crate) fn split(reason: impl Into<String>) -> Self {
        FormatError::SplitMismatch(reason.into())
    }

    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        FormatError::Io {
            path: path.into(),
            source,
        }
    }
}
