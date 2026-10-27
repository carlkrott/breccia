//! Anonymously mapped, `mlock`'d expert arena for the host lane (stage D2).
//!
//! The host lane streams routed experts off the mmap per token. This arena
//! holds the same blobs in one anonymous `MAP_PRIVATE` + `MAP_LOCKED`
//! allocation — never file-backed, so `amdgpu_ttm.c` accepts the vma as
//! USERPTR_ANONLY for a later `VK_EXT_external_memory_host` import (D3) —
//! filled once at load via `pread`/`read_exact` from GGUF `file_range()`
//! offsets.
//!
//! The mlock ulimit is small on this box; on `mlock` failure this returns a
//! clean, actionable error naming LimitMEMLOCK (D5) so the caller can degrade
//! to the streaming path.
//!
//! shortcut: no eviction/replacement policy — the arena is a fixed load-time
//! snapshot sized by the caller; upgrade when D3 lands and placement decides
//! what is resident.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::sync::{Arc, Mutex};

/// One expert blob placed in the arena: `[start, end)` bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaBlob {
    pub start: u64,
    pub end: u64,
}

impl ArenaBlob {
    pub fn len(&self) -> u64 {
        self.end - self.start
    }
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// Source for one blob: (shard file index, byte offset in file, byte length).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlobSource {
    pub file: usize,
    pub offset: u64,
    pub len: u64,
}

/// The arena: one page-rounded, page-aligned, `MAP_LOCKED` allocation plus the
/// blob ranges carved out of it. Fixed after `fill`; read-only afterwards.
pub struct HostArena {
    ptr: *mut u8,
    len: usize,
    layout: std::alloc::Layout,
    blobs: Vec<ArenaBlob>,
    pin_bytes: u64,
}

// The arena is built once per model load, then read-only from the forward pass.
unsafe impl Send for HostArena {}
unsafe impl Sync for HostArena {}

impl HostArena {
    /// Allocate `total` bytes anonymous + `MAP_LOCKED`.
    pub fn new(total: u64) -> Result<Self, String> {
        let page = 4096usize;
        let len = total.div_ceil(page as u64) as usize * page;
        if len == 0 {
            return Err("cannot build an empty host arena".into());
        }
        let layout = std::alloc::Layout::from_size_align(len, page).map_err(|e| e.to_string())?;
        // SAFETY: len >= page, both fit in usize; zeroed until `fill` runs.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(format!("host arena alloc of {len} bytes failed"));
        }
        let this = Self {
            ptr,
            len,
            layout,
            blobs: Vec::new(),
            pin_bytes: len as u64,
        };
        // mlock failure is a degrade, not an abort: name the fix (D5).
        let rc = unsafe { libc::mlock(this.ptr as *const libc::c_void, len) };
        if rc != 0 {
            let err = std::io::Error::last_os_error();
            drop(this); // dealloc, no munlock (it never locked)
            return Err(format!(
                "mlock of {len} bytes failed: {err}; raise RLIMIT_MEMLOCK \
                 (LimitMEMLOCK=infinity in the service unit, plan set D5) and retry, \
                 or keep the streaming path"
            ));
        }
        Ok(this)
    }

    /// Page-rounded bytes pinned with mlock.
    pub fn pin_bytes(&self) -> u64 {
        self.pin_bytes
    }
    /// Blob ranges, in source order.
    pub fn blobs(&self) -> &[ArenaBlob] {
        &self.blobs
    }

    /// Carve blobs from `src` (in order), filling the arena with one
    /// contiguous layout: blob i lands at `[cursor, cursor + src[i].len)`.
    /// `files` must be opened shard handles indexed by `BlobSource::file`.
    pub fn fill(&mut self, src: &[BlobSource], files: &[Arc<Mutex<File>>]) -> Result<(), String> {
        let mut layout = Vec::with_capacity(src.len());
        let mut cursor = 0u64;
        for s in src {
            if s.file >= files.len() {
                return Err(format!(
                    "blob source names shard {}, only {} opened",
                    s.file,
                    files.len()
                ));
            }
            layout.push(ArenaBlob {
                start: cursor,
                end: cursor + s.len,
            });
            cursor += s.len;
        }
        if cursor > self.len as u64 {
            return Err(format!(
                "blobs total {cursor} bytes, arena holds {}",
                self.len
            ));
        }
        let arena = self.as_mut_slice();
        let mut dst = 0usize;
        for (s, blob) in src.iter().zip(layout.iter()) {
            let mut f = files[s.file].lock().unwrap();
            f.seek(SeekFrom::Start(s.offset))
                .map_err(|e| e.to_string())?;
            let buf = &mut arena[dst..dst + s.len as usize];
            f.read_exact(buf).map_err(|e| e.to_string())?;
            dst += s.len as usize;
            debug_assert_eq!(blob.end - blob.start, s.len);
        }
        self.blobs = layout;
        Ok(())
    }

    /// Read-only view of blob `i`, checked.
    pub fn blob(&self, i: usize) -> Result<&[u8], String> {
        let b = self.blobs.get(i).ok_or_else(|| format!("no blob {i}"))?;
        let (s, e) = (b.start as usize, b.end as usize);
        if e > self.len {
            return Err(format!("blob {i} [{s}, {e}) outside arena of {}", self.len));
        }
        Ok(&self.bytes()[s..e])
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: ptr..ptr+len is a live allocation for the struct's lifetime.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: unique borrow of the live allocation for &mut self.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for HostArena {
    fn drop(&mut self) {
        unsafe { libc::munlock(self.ptr as *const libc::c_void, self.len) };
        // SAFETY: same ptr/layout this allocation was made with.
        unsafe { std::alloc::dealloc(self.ptr, self.layout) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    fn temp_shard(dir: &PathBuf, blobs: &[usize]) -> PathBuf {
        let path = dir.join("shard0.bin");
        let mut f = File::create(&path).unwrap();
        for (i, len) in blobs.iter().enumerate() {
            // blob i is exactly `len` bytes of the pattern byte (i+1)
            f.write_all(&vec![(i + 1) as u8; *len]).unwrap();
        }
        path
    }

    #[test]
    fn roundtrip_fill_and_blob_view() {
        let dir = std::env::temp_dir().join("breccia-host-arena-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = temp_shard(&dir, &[4096, 4096, 4096]);
        let files = vec![Arc::new(Mutex::new(File::open(&path).unwrap()))];
        let src = vec![
            BlobSource {
                file: 0,
                offset: 0,
                len: 4096,
            },
            BlobSource {
                file: 0,
                offset: 4096,
                len: 4096,
            },
            BlobSource {
                file: 0,
                offset: 8192,
                len: 4096,
            },
        ];
        let mut a = HostArena::new(3 * 4096).expect("mlock under limit");
        assert_eq!(a.pin_bytes(), 3 * 4096, "page-rounded, no overpin");
        a.fill(&src, &files).expect("pread fill");
        assert_eq!(a.blobs().len(), 3);
        assert_eq!(
            a.blobs()[2],
            ArenaBlob {
                start: 8192,
                end: 12288
            }
        );
        for (i, blob) in a.blobs().iter().enumerate() {
            let view = a.blob(i).expect("blob view");
            assert_eq!(view.len(), 4096);
            let want = (i + 1) as u8;
            assert!(
                view.iter().all(|b| *b == want),
                "blob {i} carries its own bytes"
            );
            assert_eq!(blob.len(), 4096);
        }
    }

    #[test]
    fn oversize_and_overfill_are_clean_errors() {
        // 512 MiB is beyond any plausible RLIMIT_MEMLOCK here: mlock must fail
        // cleanly, naming LimitMEMLOCK (D5), not abort.
        match HostArena::new(512 * 1024 * 1024) {
            Err(e) => {
                assert!(e.contains("mlock"), "must name the failing call: {e}");
                assert!(e.contains("LimitMEMLOCK"), "must point at D5: {e}");
            }
            Ok(_) => panic!("RLIMIT_MEMLOCK is high on this box; adjust the test"),
        }
        let dir = std::env::temp_dir().join("breccia-host-arena-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = temp_shard(&dir.clone(), &[4096]);
        let files = vec![Arc::new(Mutex::new(File::open(&path).unwrap()))];
        let mut a = HostArena::new(4096).expect("small arena");
        assert!(
            a.fill(
                &[BlobSource {
                    file: 0,
                    offset: 0,
                    len: 8192
                }],
                &files
            )
            .unwrap_err()
            .contains("holds"),
            "overfill must be a checked error"
        );
    }
}
