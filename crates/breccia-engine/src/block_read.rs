//! `BRECCIA_PF_DIRECT=<bufs>` (opt-in): sequential O_DIRECT reads of whole host-lane
//! expert blocks into a small ring of host buffers, one block ahead of prefill.
//!
//! 11-11: with the 8x200k pinned pool the page cache cannot hold the ~33 GB of
//! streamed experts, and prefill's 1.4 MB expert copies fault them in at 4-16 KiB
//! with a queue depth of ~8 (~300 MiB/s). A block's three expert tensors are one
//! contiguous ~725 MB file range, which O_DIRECT reads at 6.4 GB/s from one thread,
//! and bypassing the page cache leaves it to the PLE table and decode. Pages already
//! in the page cache (mincore) are copied from it with buffered reads instead, so a
//! warm block costs a memcpy, not a re-read (8x8k serve: ~46 % resident, all-O_DIRECT
//! burst 95 -> 154 s).
//!
//! The consumer is the prefill thread: [`BlockReader::get`] for block `b` marks `b`
//! current, frees every buffer not holding `b` or the next blocks, and queues those.
//! A buffer is only rewritten after a later `get` freed it, so the slice handed out
//! stays valid until the consumer's next `get` (one consumer thread).
//! Knobs: `BRECCIA_PF_DIRECT_RESIDENT` (default 0.6: leave a block this resident
//! to the mmap; 0.98 cost +13 % on the 8x8k gate, 0.6 is neutral), `BRECCIA_PF_DIRECT_CPUS`
//! (`a-b` or `a,b,..`: pin the reader thread).

use std::collections::VecDeque;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::sync::{Arc, Condvar, Mutex};

const ALIGN: u64 = 4096;
const PIECE: usize = 16 << 20;

/// One block's expert span: file `[lo, hi)` (4 KiB aligned) of shard `shard`,
/// mapped at address `map_lo` in this process.
#[derive(Clone, Copy)]
pub struct Span {
    pub shard: usize,
    pub lo: u64,
    pub hi: u64,
    pub map_lo: usize,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum State {
    Free,
    Queued,
    Loading,
    Ready,
    /// Resident in the page cache, or the read failed: the consumer uses the mmap.
    Skip,
}

struct Slot {
    block: usize,
    state: State,
    ptr: *mut u8,
    cap: usize,
}

#[derive(Default)]
struct Stats {
    read_blocks: u64,
    skipped: u64,
    failed: u64,
    bytes: u64,
    cached: u64,
    read_s: f64,
    wait_s: f64,
}

struct St {
    slots: Vec<Slot>,
    queue: VecDeque<usize>,
    stop: bool,
    current: Option<usize>,
    stats: Stats,
}

// SAFETY: the slot pointers are private anonymous mappings owned by the reader;
// access is ordered by the state machine under the mutex (see the module doc).
unsafe impl Send for St {}

struct Shared {
    st: Mutex<St>,
    cv: Condvar,
}

pub struct BlockReader {
    shared: Arc<Shared>,
    spans: Vec<Option<Span>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

/// Buffers in the ring (`BRECCIA_PF_DIRECT`), or `None` when off.
pub fn direct_bufs() -> Option<usize> {
    std::env::var("BRECCIA_PF_DIRECT").ok()?.parse::<usize>().ok().filter(|&n| n >= 2)
}

impl BlockReader {
    /// `spans[b]` is block `b`'s span (None: not streamed from the mmap); `paths[s]`
    /// is shard `s`.
    pub fn new(spans: Vec<Option<Span>>, paths: &[std::path::PathBuf], bufs: usize) -> anyhow::Result<Self> {
        let cap = spans.iter().flatten().map(|s| (s.hi - s.lo) as usize).max().unwrap_or(0);
        anyhow::ensure!(cap > 0, "no streamed expert blocks");
        let files = paths
            .iter()
            .map(|p| Ok((std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(p)?, std::fs::File::open(p)?)))
            .collect::<std::io::Result<Vec<_>>>()?;
        let mut slots = Vec::new();
        for _ in 0..bufs {
            // SAFETY: a fresh private anonymous mapping, page aligned as O_DIRECT needs.
            let p = unsafe {
                libc::mmap(std::ptr::null_mut(), cap, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0)
            };
            anyhow::ensure!(p != libc::MAP_FAILED, "mmap {cap} bytes: {}", std::io::Error::last_os_error());
            slots.push(Slot { block: usize::MAX, state: State::Free, ptr: p as *mut u8, cap });
        }
        let shared = Arc::new(Shared {
            st: Mutex::new(St { slots, queue: VecDeque::new(), stop: false, current: None, stats: Stats::default() }),
            cv: Condvar::new(),
        });
        let resident = std::env::var("BRECCIA_PF_DIRECT_RESIDENT").ok().and_then(|v| v.parse().ok()).unwrap_or(0.6f64);
        let cpus = std::env::var("BRECCIA_PF_DIRECT_CPUS").ok().map(|v| parse_cpus(&v));
        let (sh, sp) = (shared.clone(), spans.clone());
        let thread = std::thread::Builder::new()
            .name("pf-direct".into())
            .spawn(move || reader(sh, sp, files, resident, cpus))?;
        eprintln!(
            "breccia: pf-direct: {} streamed blocks, {bufs} x {} MiB O_DIRECT buffers, skip at {:.0} % resident",
            spans.iter().flatten().count(),
            cap >> 20,
            resident * 100.0
        );
        Ok(BlockReader { shared, spans, thread: Some(thread) })
    }

    /// The buffer holding block `b`'s span as `(map_lo, bytes)`, waiting for its read;
    /// None when `b` is not streamed or uses the mmap (resident, failed, no free buffer).
    pub fn get(&self, b: usize) -> Option<(usize, &[u8])> {
        self.spans.get(b).copied().flatten()?;
        let mut st = self.shared.st.lock().unwrap();
        if st.current != Some(b) {
            st.current = Some(b);
            let n = st.slots.len();
            let want = self.ahead(b, n);
            for s in st.slots.iter_mut() {
                if !want.contains(&s.block) && s.state != State::Loading {
                    s.state = State::Free;
                    s.block = usize::MAX;
                }
            }
            for &w in &want {
                if st.slots.iter().any(|s| s.block == w) {
                    continue;
                }
                if let Some(i) = st.slots.iter().position(|s| s.state == State::Free) {
                    st.slots[i].block = w;
                    st.slots[i].state = State::Queued;
                    st.queue.push_back(i);
                }
            }
            self.shared.cv.notify_all();
        }
        let t0 = std::time::Instant::now();
        let i = st.slots.iter().position(|s| s.block == b)?;
        while matches!(st.slots[i].state, State::Queued | State::Loading) {
            st = self.shared.cv.wait(st).unwrap();
        }
        st.stats.wait_s += t0.elapsed().as_secs_f64();
        let s = &st.slots[i];
        (s.state == State::Ready && s.block == b).then(|| {
            let sp = self.spans[b].unwrap();
            // SAFETY: Ready means the reader wrote [0, hi - lo) and will not touch the
            // slot again until a later get() frees it (module doc).
            (sp.map_lo, unsafe { std::slice::from_raw_parts(s.ptr, (sp.hi - sp.lo) as usize) })
        })
    }

    /// `b` and the next `n - 1` streamed blocks after it, wrapping (the next chunk).
    fn ahead(&self, b: usize, n: usize) -> Vec<usize> {
        let l = self.spans.len();
        (0..l).map(|i| (b + i) % l).filter(|&x| self.spans[x].is_some()).take(n).collect()
    }
}

impl Drop for BlockReader {
    fn drop(&mut self) {
        self.shared.st.lock().unwrap().stop = true;
        self.shared.cv.notify_all();
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        let st = self.shared.st.lock().unwrap();
        let s = &st.stats;
        eprintln!(
            "breccia: pf-direct: read {} blocks {:.1} GiB in {:.1} s ({:.2} GB/s), skipped {} resident, {} failed, prefill waited {:.1} s",
            s.read_blocks,
            s.bytes as f64 / (1u64 << 30) as f64,
            s.read_s,
            if s.read_s > 0.0 { s.bytes as f64 / s.read_s / 1e9 } else { 0.0 },
            s.skipped,
            s.failed,
            s.wait_s
        );
        for sl in &st.slots {
            // SAFETY: the reader thread has exited; nothing else holds the mapping.
            unsafe { libc::munmap(sl.ptr as *mut libc::c_void, sl.cap) };
        }
    }
}

fn parse_cpus(v: &str) -> Vec<usize> {
    v.split(',')
        .flat_map(|p| match p.split_once('-') {
            Some((a, b)) => a.trim().parse().unwrap_or(0)..=b.trim().parse().unwrap_or(0),
            None => {
                let a = p.trim().parse().unwrap_or(usize::MAX);
                a..=a
            }
        })
        .filter(|&c| c < 1024)
        .collect()
}

/// Per 4 KiB page of the span: true when it is in the page cache (all false when
/// mincore fails).
fn residency(sp: &Span) -> Vec<bool> {
    let len = (sp.hi - sp.lo) as usize;
    let mut v = vec![0u8; len.div_ceil(4096)];
    // SAFETY: map_lo is page aligned (lo is) and [map_lo, map_lo + len) lies in the shard's mapping.
    let r = unsafe { libc::mincore(sp.map_lo as *mut libc::c_void, len, v.as_mut_ptr()) };
    v.iter().map(|&x| r == 0 && x & 1 == 1).collect()
}

/// Page runs `(start, end, cold)`: cold runs are read with O_DIRECT, warm ones are
/// copied out of the page cache. A warm gap under `gap` pages joins the cold read
/// around it, so reads stay large.
fn runs(res: &[bool], gap: usize) -> Vec<(usize, usize, bool)> {
    let mut raw: Vec<(usize, usize, bool)> = Vec::new();
    for (i, &r) in res.iter().enumerate() {
        match raw.last_mut() {
            Some(l) if l.2 == !r => l.1 = i + 1,
            _ => raw.push((i, i + 1, !r)),
        }
    }
    let n = raw.len();
    for k in 0..n {
        let (a, e, cold) = raw[k];
        if !cold && e - a < gap && k > 0 && k + 1 < n {
            raw[k].2 = true;
        }
    }
    let mut out: Vec<(usize, usize, bool)> = Vec::new();
    for r in raw {
        match out.last_mut() {
            Some(l) if l.2 == r.2 => l.1 = r.1,
            _ => out.push(r),
        }
    }
    out
}

/// Fill `buf` from `f` at file offset `at` (EOF inside the rounded-up tail is past
/// every tensor, so zeros do).
fn read_full(f: &std::fs::File, buf: &mut [u8], at: u64) -> std::io::Result<()> {
    for (k, piece) in buf.chunks_mut(PIECE).enumerate() {
        let at = at + (k * PIECE) as u64;
        let mut done = 0;
        while done < piece.len() {
            match f.read_at(&mut piece[done..], at + done as u64) {
                Ok(0) => {
                    piece[done..].fill(0);
                    break;
                }
                Ok(n) => done += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
    }
    Ok(())
}

fn reader(sh: Arc<Shared>, spans: Vec<Option<Span>>, files: Vec<(std::fs::File, std::fs::File)>, resident: f64, cpus: Option<Vec<usize>>) {
    if let Some(cpus) = cpus.filter(|c| !c.is_empty()) {
        // SAFETY: a zeroed cpu_set_t with CPU_SET on in-range indices, for this thread.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            for c in &cpus {
                libc::CPU_SET(*c, &mut set);
            }
            if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
                eprintln!("breccia: pf-direct: sched_setaffinity {cpus:?}: {}", std::io::Error::last_os_error());
            }
        }
    }
    let nspans = spans.iter().flatten().count().max(1) as u64;
    loop {
        let (i, b, ptr) = {
            let mut st = sh.st.lock().unwrap();
            loop {
                if st.stop {
                    return;
                }
                if let Some(i) = st.queue.pop_front() {
                    if st.slots[i].state == State::Queued {
                        st.slots[i].state = State::Loading;
                        break (i, st.slots[i].block, st.slots[i].ptr);
                    }
                    continue;
                }
                st = sh.cv.wait(st).unwrap();
            }
        };
        let sp = spans[b].unwrap();
        let len = (sp.hi - sp.lo) as usize;
        let t0 = std::time::Instant::now();
        let res = residency(&sp);
        let warm = res.iter().filter(|&&r| r).count() as f64 >= resident * res.len() as f64;
        let mut cached = 0u64;
        let outcome = if warm {
            State::Skip
        } else {
            // SAFETY: the slot is Loading, which only this thread writes until it sets
            // Ready/Skip; cap >= len.
            let buf = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
            let (direct, buffered) = &files[sp.shard];
            let r = runs(&res, 256).into_iter().try_for_each(|(a, e, cold)| {
                let (o, end) = (a * 4096, (e * 4096).min(len));
                if !cold {
                    cached += (end - o) as u64;
                }
                read_full(if cold { direct } else { buffered }, &mut buf[o..end], sp.lo + o as u64)
            });
            match r {
                Ok(()) => State::Ready,
                Err(e) => {
                    eprintln!("breccia: pf-direct: blk.{b} read: {e}");
                    State::Skip
                }
            }
        };
        let mut st = sh.st.lock().unwrap();
        match outcome {
            State::Ready => {
                st.stats.read_blocks += 1;
                st.stats.bytes += len as u64 - cached;
                st.stats.cached += cached;
                st.stats.read_s += t0.elapsed().as_secs_f64();
                // One line per pass: serve is stopped by a signal, so Drop never reports.
                if st.stats.read_blocks % nspans == 0 {
                    let s = &st.stats;
                    eprintln!(
                        "breccia: pf-direct: {} blocks: {:.1} GiB O_DIRECT + {:.1} GiB page cache at {:.2} GB/s, {} skipped resident, prefill waited {:.1} s",
                        s.read_blocks,
                        s.bytes as f64 / (1u64 << 30) as f64,
                        s.cached as f64 / (1u64 << 30) as f64,
                        (s.bytes + s.cached) as f64 / s.read_s / 1e9,
                        s.skipped,
                        s.wait_s
                    );
                }
            }
            _ if warm => st.stats.skipped += 1,
            _ => st.stats.failed += 1,
        }
        // A get() may have moved on and freed this slot's claim meanwhile: keep it Free.
        if st.slots[i].block == b {
            st.slots[i].state = outcome;
        } else {
            st.slots[i].state = State::Free;
        }
        sh.cv.notify_all();
    }
}

/// Aligned span `[lo, hi)` around `ranges` (all in one shard).
pub fn span_of(ranges: &[std::ops::Range<u64>], file_len: u64) -> (u64, u64) {
    let lo = ranges.iter().map(|r| r.start).min().unwrap_or(0) & !(ALIGN - 1);
    let hi = ranges.iter().map(|r| r.end).max().unwrap_or(0).div_ceil(ALIGN) * ALIGN;
    (lo, hi.min(file_len.div_ceil(ALIGN).saturating_mul(ALIGN)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_rounds_out_and_cpus_parse() {
        assert_eq!(span_of(&[9422044672..9657974272, 9904709632..10147192832], u64::MAX), (9422041088, 10147192832));
        assert_eq!(span_of(&[1..4097], 5000), (0, 8192));
        assert_eq!(parse_cpus("2-4,7"), vec![2, 3, 4, 7]);
        // warm 2 | cold 1 | warm 1 (gap, joins) | cold 2 | warm 3
        let res = [true, true, false, true, false, false, true, true, true];
        assert_eq!(runs(&res, 2), vec![(0, 2, false), (2, 6, true), (6, 9, false)]);
        assert_eq!(runs(&res, 1), vec![(0, 2, false), (2, 3, true), (3, 4, false), (4, 6, true), (6, 9, false)]);
    }

    /// The ring hands out the file's bytes for each block and keeps going round.
    #[test]
    fn reads_blocks_in_order_and_wraps() {
        let dir = std::env::temp_dir().join(format!("pf-direct-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("f.bin");
        let data: Vec<u8> = (0..5 * 8192u32).map(|i| (i.wrapping_mul(2654435761) >> 7) as u8).collect();
        std::fs::write(&path, &data).unwrap();
        let spans = (0..5).map(|b| if b == 2 { None } else { Some(Span { shard: 0, lo: b * 8192, hi: (b + 1) * 8192, map_lo: 0 }) }).collect();
        std::env::set_var("BRECCIA_PF_DIRECT_RESIDENT", "2"); // never skip (mincore on map_lo 0 fails anyway)
        let Ok(r) = BlockReader::new(spans, &[path.clone()], 2) else {
            eprintln!("O_DIRECT unsupported here, skipped");
            return;
        };
        for round in 0..2 {
            for b in [0usize, 1, 3, 4] {
                let (_, got) = r.get(b).unwrap_or_else(|| panic!("round {round} blk {b}"));
                assert_eq!(got, &data[b * 8192..(b + 1) * 8192]);
            }
            assert!(r.get(2).is_none());
        }
        drop(r);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
