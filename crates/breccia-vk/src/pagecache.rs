//! Linux page-cache hints for the shard mappings (R29 doc 29 card 1).
//!
//! Page-cache trimming is a whole-pages-only tweak: it never changes a decoded
//! token, only which file pages the kernel keeps. Callers use it around an
//! upload, so it lives next to the transfer layer the upload runs in.

use libc::{c_int, posix_fadvise, POSIX_FADV_DONTNEED, POSIX_FADV_WILLNEED};
use std::fs::File;
use std::io;
use std::os::unix::io::AsRawFd;

fn advise(file: &File, offset: u64, len: u64, advice: c_int) -> io::Result<()> {
    let fd = file.as_raw_fd();
    let start = offset & !4095;
    let end = (offset.saturating_add(len))
        .checked_add(4095)
        .map(|v| v & !4095);
    let (start, len) = match (
        i64::try_from(start),
        end.and_then(|e| i64::try_from(e).ok()),
    ) {
        (Ok(s), Some(e)) if e >= s => (s, e - s),
        _ => return Ok(()), // out-of-range end: nothing to advise on
    };
    // posix_fadvise returns 0 on success and an errno code (not -1) on failure.
    let r = unsafe { posix_fadvise(fd, start, len, advice) };
    if r != 0 {
        return Err(io::Error::from_raw_os_error(r as c_int));
    }
    Ok(())
}

/// DONTNEED the pages covering `[offset, offset+len)` of `file`. Run *after*
/// the bytes are uploaded/consumed. Offsets are file-relative and page-floor'd,
/// so the range can only over-include boundary bytes — harmless for a hint.
///
/// shortcut: fadvise does not drop pages still held by a mmap; add
/// MADV_DONTNEED on the shard mappings when a qualification run shows them
/// resident anyway.
pub fn drop_range(file: &File, offset: u64, len: u64) -> io::Result<()> {
    advise(file, offset, len, POSIX_FADV_DONTNEED)
}

/// WILLNEED the pages covering `[offset, offset+len)` — readahead hint for the
/// streamed/host-lane expert ranges ahead of the next pass.
pub fn willneed_range(file: &File, offset: u64, len: u64) -> io::Result<()> {
    advise(file, offset, len, POSIX_FADV_WILLNEED)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(create: bool) -> File {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "breccia-pagecache-{}-{}",
            std::process::id(),
            if create { "real" } else { "ovf" }
        ));
        if create {
            std::fs::write(&p, vec![0u8; 8192]).unwrap();
        } else {
            let f = File::create(&p).unwrap();
            std::fs::remove_file(&p).ok();
            return f;
        }
        File::open(&p).unwrap()
    }

    #[test]
    fn valid_range_on_real_file_is_ok() {
        let f = scratch(true);
        assert!(drop_range(&f, 0, 8192).is_ok());
        assert!(willneed_range(&f, 0, 8192).is_ok());
        std::fs::remove_file(
            std::env::temp_dir().join(format!("breccia-pagecache-{}-real", std::process::id())),
        )
        .ok();
    }

    #[test]
    fn huge_range_is_noop_not_panic() {
        let f = scratch(false);
        assert!(drop_range(&f, u64::MAX - 100, 1 << 40).is_ok());
        assert!(willneed_range(&f, u64::MAX - 100, 1 << 40).is_ok());
    }
}
