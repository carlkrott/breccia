//! Shared read-only mmap helpers.

use crate::error::{FormatError, Result};
use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

/// Map `path` read-only.
pub(crate) fn map_readonly(path: &Path) -> Result<Mmap> {
    let file = File::open(path).map_err(|e| FormatError::io(path, e))?;
    // SAFETY: the mapping is read-only and owned by the returned `Mmap`; model
    // artifacts are treated as immutable inputs (read-only mounts / files).
    unsafe { Mmap::map(&file) }.map_err(|e| FormatError::io(path, e))
}

/// Map `path` if it exists.
pub(crate) fn map_optional(path: &Path) -> Result<Option<Mmap>> {
    if path.exists() {
        map_readonly(path).map(Some)
    } else {
        Ok(None)
    }
}
