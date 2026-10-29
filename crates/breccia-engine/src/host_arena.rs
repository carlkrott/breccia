//! Anonymously mapped, `mlock`'d expert arena for the host lane (stage D2).
//!
//! The host lane streams routed experts off the mmap per token. This arena
//! holds the same blobs in one anonymous allocation (`alloc_zeroed` plus
//! `mlock`) — never file-backed, so `amdgpu_ttm.c` accepts the vma as
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

const PAGE: u64 = 4096;
/// Bytes a GEMV may read past a weight's end (`gemv::alloc` pads by the same).
const TAIL: u64 = breccia_kernels::gemv::TAIL_PAD;

/// Arena bytes one blob of `len` occupies: its bytes plus the GEMV tail slack,
/// rounded to the page so the next blob starts page-aligned.
fn slot(len: u64) -> u64 {
    (len + TAIL).next_multiple_of(PAGE)
}

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

/// The arena: one page-rounded, page-aligned, `mlock`'d allocation plus the
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
        let page = PAGE as usize;
        let len = total.div_ceil(PAGE) as usize * page;
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

    /// Arena bytes `fill` needs for `src`: every blob starts on a page and keeps
    /// zeroed GEMV tail slack, so each one can be imported on its own as a weight
    /// buffer (D3: pointer and size 4096-aligned).
    pub fn padded_total(src: &[BlobSource]) -> u64 {
        src.iter().map(|s| slot(s.len)).sum()
    }

    /// Carve blobs from `src` (in order): blob i lands at the next page
    /// boundary, `[cursor, cursor + src[i].len)`, followed by zeroed tail slack
    /// up to the next page.
    /// `files` must be opened shard handles indexed by `BlobSource::file`.
    pub fn fill(&mut self, src: &[BlobSource], files: &[Arc<Mutex<File>>]) -> Result<(), String> {
        let mut layout = Vec::with_capacity(src.len());
        let mut cursor = 0u64;
        for s in src {
            cursor = cursor.next_multiple_of(PAGE);
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
            cursor += slot(s.len);
        }
        if cursor > self.len as u64 {
            return Err(format!(
                "blobs total {cursor} bytes, arena holds {}",
                self.len
            ));
        }
        let arena = self.as_mut_slice();
        for (s, blob) in src.iter().zip(layout.iter()) {
            let mut f = files[s.file].lock().unwrap();
            f.seek(SeekFrom::Start(s.offset))
                .map_err(|e| e.to_string())?;
            let buf = &mut arena[blob.start as usize..blob.end as usize];
            f.read_exact(buf).map_err(|e| e.to_string())?;
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

    /// Page-aligned pointer and page-rounded length (blob plus tail slack) of
    /// blob `i`, for a `VK_EXT_external_memory_host` import. The pad bytes are zero.
    pub fn blob_import_range(&self, i: usize) -> Result<(*mut u8, u64), String> {
        let b = self.blobs.get(i).ok_or_else(|| format!("no blob {i}"))?;
        let len = slot(b.len());
        if b.start % PAGE != 0 || b.start + len > self.len as u64 {
            return Err(format!("blob {i} [{}, +{len}) is not an importable range", b.start));
        }
        // SAFETY: start + len <= self.len, checked above.
        Ok((unsafe { self.ptr.add(b.start as usize) }, len))
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
        let mut a = HostArena::new(HostArena::padded_total(&src)).expect("mlock under limit");
        assert_eq!(a.pin_bytes(), 6 * 4096, "a page-sized blob plus its tail slack is two pages");
        a.fill(&src, &files).expect("pread fill");
        assert_eq!(a.blobs().len(), 3);
        assert_eq!(
            a.blobs()[2],
            ArenaBlob {
                start: 16384,
                end: 20480
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

    #[test]
    fn odd_blobs_start_on_pages_and_import_ranges_are_aligned() {
        let dir = std::env::temp_dir().join("breccia-host-arena-pad-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = temp_shard(&dir, &[3000, 5000]);
        let files = vec![Arc::new(Mutex::new(File::open(&path).unwrap()))];
        let src = vec![
            BlobSource { file: 0, offset: 0, len: 3000 },
            BlobSource { file: 0, offset: 3000, len: 5000 },
        ];
        assert_eq!(HostArena::padded_total(&src), 4096 + 8192);
        let mut a = HostArena::new(HostArena::padded_total(&src)).expect("mlock under limit");
        a.fill(&src, &files).expect("pread fill");
        assert_eq!(a.blobs()[1], ArenaBlob { start: 4096, end: 9096 });
        assert!(a.blob(1).unwrap().iter().all(|b| *b == 2));
        let (p0, l0) = a.blob_import_range(0).unwrap();
        let (p1, l1) = a.blob_import_range(1).unwrap();
        assert_eq!((p0 as usize % 4096, l0, p1 as usize - p0 as usize, l1), (0, 4096, 4096, 8192));
    }
}
