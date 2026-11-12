//! `BRECCIA_PREFIX_DIR=<dir>` (opt-in): a disk tier for serve's text prefix states.
//! RAM keeps the hot prefixes (`BRECCIA_TEXT_PREFIX_GIB`); a prefix pruned from RAM
//! is written here, and a request whose prompt starts with a stored key resumes from
//! the file instead of re-filling (a 64k state is ~0.85 GB: ~1-2 s to read vs ~2 min
//! to re-fill). Each entry is `<hash>.bpx` (the state) + `<hash>.ids` (its tokens),
//! both written to a temp name and renamed, ids last, so an `.ids` file means a
//! complete entry. The directory is re-scanned on lookup, so two serves on one
//! directory share it: an RX serve writing with `BRECCIA_PREFIX_DIR_EAGER=1` hands
//! its fills to an MI50 serve (option 7). LRU by file mtime, capped by
//! `BRECCIA_PREFIX_DIR_GIB` (default 32).
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub struct PrefixDisk {
    dir: PathBuf,
    cap: u64,
    /// key tokens, state file
    index: Vec<(Vec<u32>, PathBuf)>,
}

fn name(key: &[u32]) -> String {
    let bytes: Vec<u8> = key.iter().flat_map(|t| t.to_le_bytes()).collect();
    blake3::hash(&bytes).to_hex()[..32].to_string()
}

/// Drop a file's pages from the page cache: the expert stream lives there.
/// shortcut: buffered I/O then DONTNEED still churns ~1 GB of cache per transfer; O_DIRECT if a fill shows it.
pub fn drop_cache(p: &Path) {
    use std::os::fd::AsRawFd;
    if let Ok(f) = std::fs::File::open(p) {
        // SAFETY: a valid open fd; advice only.
        unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    }
}

impl PrefixDisk {
    pub fn new(dir: PathBuf, cap: u64) -> Result<Self> {
        std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        let mut d = Self { dir, cap, index: Vec::new() };
        d.scan();
        Ok(d)
    }

    /// Forget entries whose state is gone; index `.ids` files not seen yet (another serve's).
    fn scan(&mut self) {
        self.index.retain(|e| e.1.exists());
        let Ok(rd) = std::fs::read_dir(&self.dir) else { return };
        for p in rd.flatten().map(|e| e.path()) {
            if p.extension().is_none_or(|x| x != "ids") {
                continue;
            }
            let st = p.with_extension("bpx");
            if self.index.iter().any(|e| e.1 == st) || !st.exists() {
                continue;
            }
            if let Ok(raw) = std::fs::read(&p) {
                let key = raw.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
                self.index.push((key, st));
            }
        }
    }

    /// The longest stored key that `prompt` starts with, at most `cut` and longer than
    /// `beat` tokens; its mtime is bumped (LRU).
    pub fn longest(&mut self, prompt: &[u32], cut: usize, beat: usize) -> Option<(usize, PathBuf)> {
        self.scan();
        let e = self.index.iter().filter(|e| e.0.len() <= cut && e.0.len() > beat && prompt.starts_with(&e.0)).max_by_key(|e| e.0.len())?;
        touch(&e.1);
        Some((e.0.len(), e.1.clone()))
    }

    /// Store `key`'s state (written by `write`) unless it is already here; then prune.
    pub fn put(&mut self, key: &[u32], write: impl FnOnce(&Path) -> Result<()>) -> Result<bool> {
        let n = name(key);
        let st = self.dir.join(format!("{n}.bpx"));
        if st.exists() {
            touch(&st);
            return Ok(false);
        }
        let pid = std::process::id();
        let tmp = self.dir.join(format!("{n}.bpx.{pid}.tmp"));
        write(&tmp)?;
        drop_cache(&tmp);
        std::fs::rename(&tmp, &st)?;
        let ids = self.dir.join(format!("{n}.ids"));
        let tmp = self.dir.join(format!("{n}.ids.{pid}.tmp"));
        std::fs::write(&tmp, key.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>())?;
        std::fs::rename(&tmp, &ids)?;
        self.index.push((key.to_vec(), st));
        self.prune();
        Ok(true)
    }

    /// Newest first until the cap; the rest are removed (ids first, so a reader never sees half an entry).
    fn prune(&mut self) {
        let meta = |p: &Path| std::fs::metadata(p).ok().map(|m| (m.len(), m.modified().ok()));
        let mut v: Vec<_> = self.index.iter().filter_map(|e| meta(&e.1).map(|(len, t)| (t, len, e.1.clone()))).collect();
        v.sort_by(|a, b| b.0.cmp(&a.0));
        let mut total = 0;
        for (_, len, p) in v {
            total += len;
            if total > self.cap {
                let _ = std::fs::remove_file(p.with_extension("ids"));
                let _ = std::fs::remove_file(&p);
            }
        }
        self.index.retain(|e| e.1.exists());
    }
}

fn touch(p: &Path) {
    if let Ok(f) = std::fs::File::options().write(true).open(p) {
        let _ = f.set_modified(std::time::SystemTime::now());
    }
}

#[cfg(test)]
mod tests {
    use super::PrefixDisk;
    #[test]
    fn put_longest_share_prune() {
        let dir = std::env::temp_dir().join(format!("bpx-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let w = |n: usize| move |p: &std::path::Path| Ok(std::fs::write(p, vec![0u8; n])?);
        let mut a = PrefixDisk::new(dir.clone(), 250).unwrap();
        assert!(a.put(&[1, 2, 3], w(100)).unwrap());
        assert!(!a.put(&[1, 2, 3], w(100)).unwrap(), "same key is stored once");
        // a second serve on the same directory sees the first one's entry
        let mut b = PrefixDisk::new(dir.clone(), 250).unwrap();
        assert!(b.put(&[1, 2, 3, 4, 5], w(100)).unwrap());
        assert_eq!(a.longest(&[1, 2, 3, 4, 5, 6], 5, 0).unwrap().0, 5, "a picks up b's longer key");
        assert_eq!(a.longest(&[1, 2, 3, 4, 5, 6], 4, 0).unwrap().0, 3, "keys over the cut are skipped");
        assert!(a.longest(&[1, 2, 3, 9], 3, 3).is_none(), "only keys longer than the RAM hit");
        assert!(a.longest(&[7, 2, 3], 3, 0).is_none());
        std::thread::sleep(std::time::Duration::from_millis(20));
        a.longest(&[1, 2, 3, 9], 3, 0); // [1,2,3] is now the most recent
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert!(a.put(&[8, 8], w(100)).unwrap()); // 300 > 250: the LRU one ([1..5]) goes
        assert!(a.longest(&[1, 2, 3, 4, 5], 5, 0).unwrap().0 == 3);
        assert!(b.longest(&[1, 2, 3, 4, 5], 5, 3).is_none(), "b forgets the pruned entry");
        assert!(std::fs::read_dir(&dir).unwrap().flatten().all(|e| !e.path().to_string_lossy().ends_with(".tmp")));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
