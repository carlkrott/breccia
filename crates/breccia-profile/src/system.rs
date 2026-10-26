//! System (host) probe: CPU, NUMA, RAM, effective mlock limit, zram, io_uring,
//! storage free space + a quick read benchmark of the Tier-3 candidate.
//!
//! Everything here is read-only. `/proc`, `/sys` and `statvfs`/`getrlimit` only —
//! no external processes except one `pgrep` for GPU-contention evidence.

use crate::schema::{HostInfo, IoUringInfo, ReadBench, StorageInfo, TierSizing, ZramInfo};
use std::ffi::CString;
use std::os::raw::c_void;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

/// Filesystems probed for free space (Tier-3 candidates + root).
pub const STORAGE_PATHS: [&str; 3] = ["/mnt/agent-storage-a", "/mnt/agent-storage-b", "/"];
/// Tier-3 candidate: gets the read benchmark (docs/phase-0/04).
pub const TIER3_PATH: &str = "/mnt/agent-storage-a";
/// 64 MiB of sequential read is enough for an MB/s estimate and stays polite.
const READ_BENCH_BYTES: u64 = 64 << 20;

// ---------------------------------------------------------------------------
// Small /proc + /sys readers
// ---------------------------------------------------------------------------

fn read_trimmed(path: &str) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// `"MemTotal:       61234567 kB"` → bytes.
pub fn parse_meminfo_kb(text: &str, key: &str) -> Option<u64> {
    let line = text.lines().find(|l| l.starts_with(key))?;
    let mut it = line.split_whitespace();
    it.next()?; // key:
    let kb: u64 = it.next()?.parse().ok()?;
    Some(kb * 1024)
}

pub fn meminfo() -> (u64, u64) {
    let text = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let total = parse_meminfo_kb(&text, "MemTotal:").unwrap_or(0);
    let avail = parse_meminfo_kb(&text, "MemAvailable:").unwrap_or(0);
    (total, avail)
}

/// `(model_name, physical_cores, logical_threads)` parsed from `/proc/cpuinfo`.
pub fn parse_cpuinfo(text: &str) -> (String, u32, u32) {
    let mut model = String::new();
    let mut cores = std::collections::HashSet::new();
    let mut threads = 0u32;
    let mut phys = String::new();
    let mut core = String::new();
    for line in text.lines() {
        if let Some((k, v)) = line.split_once(':') {
            let k = k.trim();
            let v = v.trim();
            match k {
                "model name" if model.is_empty() => model = v.to_string(),
                "processor" => threads += 1,
                "physical id" => phys = v.to_string(),
                "core id" => core = v.to_string(),
                _ => {}
            }
        }
        // A new "processor" block boundary: record the previous core identity.
        if line.starts_with("processor") && !phys.is_empty() && !core.is_empty() {
            cores.insert(format!("{phys}:{core}"));
            phys.clear();
            core.clear();
        }
    }
    if !phys.is_empty() && !core.is_empty() {
        cores.insert(format!("{phys}:{core}"));
    }
    let physical = if cores.is_empty() {
        threads
    } else {
        cores.len() as u32
    };
    (model, physical, threads.max(physical))
}

pub fn cpu_info() -> (String, u32, u32) {
    let text = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    parse_cpuinfo(&text)
}

/// Count `/sys/devices/system/node/nodeN` entries (0 → treat as 1 NUMA node).
pub fn numa_nodes() -> u32 {
    let mut n = 0u32;
    if let Ok(dir) = std::fs::read_dir("/sys/devices/system/node") {
        for e in dir.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with("node") && name[4..].chars().all(|c| c.is_ascii_digit()) {
                n += 1;
            }
        }
    }
    n.max(1)
}

/// Parse `/proc/swaps`: `Filename Type Size Used Priority`.
/// Returns `(device, size_bytes, used_bytes)` for the first zram device.
pub fn parse_swaps_zram(text: &str) -> Option<(String, u64, u64)> {
    for line in text.lines().skip(1) {
        let mut it = line.split_whitespace();
        let name = it.next()?;
        if !name.contains("zram") {
            continue;
        }
        let _typ = it.next();
        let size_kib: u64 = it.next()?.parse().ok()?;
        let used_kib: u64 = it.next()?.parse().unwrap_or(0);
        return Some((name.to_string(), size_kib * 1024, used_kib * 1024));
    }
    None
}

pub fn zram_info() -> Option<ZramInfo> {
    let text = std::fs::read_to_string("/proc/swaps").ok()?;
    let (device, disksize, used) = parse_swaps_zram(&text)?;
    // /proc/swaps' Size column is the configured disksize; cross-check sysfs.
    let sysfs = read_trimmed(&format!(
        "/sys/block/{}/disksize",
        Path::new(&device).file_name()?.to_string_lossy()
    ))
    .and_then(|s| s.parse::<u64>().ok());
    Some(ZramInfo {
        device,
        disksize_bytes: sysfs.unwrap_or(disksize),
        used_bytes: used,
    })
}

/// `/proc/sys/fs/io_uring_disabled`: absent = unrestricted kernel, 0 = allowed,
/// 1 = unprivileged disabled, 2 = fully disabled.
pub fn io_uring_from_sysctl(value: Option<&str>) -> IoUringInfo {
    match value {
        None => IoUringInfo {
            available: true,
            detail: "sysctl absent (no restriction compiled in)".to_string(),
        },
        Some("0") => IoUringInfo {
            available: true,
            detail: "/proc/sys/fs/io_uring_disabled=0".to_string(),
        },
        Some(v) => IoUringInfo {
            available: false,
            detail: format!("/proc/sys/fs/io_uring_disabled={v} (io_uring blocked)"),
        },
    }
}

pub fn io_uring_status() -> IoUringInfo {
    io_uring_from_sysctl(read_trimmed("/proc/sys/fs/io_uring_disabled").as_deref())
}

/// `RLIMIT_MEMLOCK` soft/hard in bytes (`u64::MAX` = unlimited).
pub fn mlock_limits() -> (u64, u64) {
    unsafe {
        let mut r: libc::rlimit = std::mem::zeroed();
        if libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut r) != 0 {
            return (0, 0);
        }
        (r.rlim_cur as u64, r.rlim_max as u64)
    }
}

/// `statvfs` → `(total_bytes, free_bytes_for_unprivileged_writes)`.
pub fn statvfs(path: &str) -> Option<(u64, u64)> {
    let c = CString::new(path).ok()?;
    unsafe {
        let mut st: libc::statvfs = std::mem::zeroed();
        if libc::statvfs(c.as_ptr(), &mut st) != 0 {
            return None;
        }
        let frsize = if st.f_frsize != 0 {
            st.f_frsize as u64
        } else {
            st.f_bsize as u64
        };
        Some((st.f_blocks as u64 * frsize, st.f_bavail as u64 * frsize))
    }
}

/// Connector *entries* present (`cardN-NAME`) — fallback when `status` is unreadable.
pub fn any_connector(entries: &[String]) -> bool {
    entries
        .iter()
        .any(|e| e.starts_with("card") && e.contains('-'))
}

/// Any connector reporting `connected` (writeback/disconnected do not count).
pub fn any_connected(statuses: &[String]) -> bool {
    statuses.iter().any(|s| s.trim() == "connected")
}

/// True when the PCI device has a display *connector in the `connected` state*
/// (i.e. it actually drives a display and needs a VRAM reserve). Falls back to
/// mere connector presence when no status file is readable.
pub fn has_display_outputs(pci_address: &str) -> bool {
    if pci_address.is_empty() {
        return false;
    }
    let mut names = Vec::new();
    let mut statuses = Vec::new();
    // Layout: <pci>/drm/cardN/cardN-CONNECTOR/status
    if let Ok(dir) = std::fs::read_dir(format!("/sys/bus/pci/devices/{pci_address}/drm")) {
        for card in dir
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("card"))
        {
            let Ok(connectors) = std::fs::read_dir(card.path()) else {
                continue;
            };
            for c in connectors.flatten() {
                let name = c.file_name().to_string_lossy().into_owned();
                if name.starts_with("card") && name.contains('-') {
                    if let Some(s) = read_trimmed(&format!("{}/status", c.path().display())) {
                        statuses.push(s);
                    }
                    names.push(name);
                }
            }
        }
    }
    if statuses.is_empty() {
        any_connector(&names)
    } else {
        any_connected(&statuses)
    }
}

/// Processes that looked GPU-ish at probe time (measurement-quality evidence,
/// same pattern as the Phase-0.2 harness).
pub fn competing_gpu_processes() -> Vec<String> {
    let out = std::process::Command::new("pgrep")
        .args(["-af", "llama-server|llama-bench|strata-iq|comfyui"])
        .output();
    match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty() && !l.starts_with("pgrep"))
            .collect(),
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Storage read benchmark (Tier-3 candidate)
// ---------------------------------------------------------------------------

/// Bounded scan for the largest regular file under `dir` (depth ≤ 3, ≤ 4096 entries)
/// so the probe never walks a huge tree. Read-only.
pub fn find_benchmark_file(dir: &Path, min_bytes: u64) -> Option<(PathBuf, u64)> {
    let mut best: Option<(PathBuf, u64)> = None;
    let mut stack = vec![(dir.to_path_buf(), 0u32)];
    let mut visited = 0usize;
    while let Some((d, depth)) = stack.pop() {
        if depth > 3 || visited > 4096 {
            continue;
        }
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.flatten() {
            visited += 1;
            if visited > 4096 {
                break;
            }
            let path = e.path();
            let Ok(md) = e.metadata() else { continue };
            if md.is_dir() {
                stack.push((path, depth + 1));
            } else if md.is_file() {
                let len = md.len();
                if len >= min_bytes && best.as_ref().map(|(_, l)| len > *l).unwrap_or(true) {
                    best = Some((path, len));
                }
            }
        }
    }
    best
}

fn alloc_aligned(size: usize) -> Option<*mut u8> {
    let mut p: *mut c_void = std::ptr::null_mut();
    let rc = unsafe { libc::posix_memalign(&mut p, 4096, size) };
    if rc != 0 || p.is_null() {
        None
    } else {
        Some(p as *mut u8)
    }
}

/// Sequential read through `O_DIRECT` (bypasses the page cache). Returns
/// `(bytes, seconds)`; `Err` carries the errno text for the buffered fallback.
fn read_direct(path: &Path, want: u64) -> Result<(u64, f64), String> {
    let cpath = CString::new(path.as_os_str().as_bytes()).map_err(|e| format!("bad path: {e}"))?;
    let fd = unsafe { libc::open(cpath.as_ptr(), libc::O_RDONLY | libc::O_DIRECT) };
    if fd < 0 {
        return Err(format!(
            "open(O_DIRECT): {}",
            std::io::Error::last_os_error()
        ));
    }
    const CHUNK: usize = 1 << 20;
    let buf = match alloc_aligned(CHUNK) {
        Some(b) => b,
        None => {
            unsafe { libc::close(fd) };
            return Err("posix_memalign failed".into());
        }
    };
    let mut total = 0u64;
    let mut err = None;
    let t0 = std::time::Instant::now();
    while total < want {
        let n = unsafe { libc::read(fd, buf as *mut c_void, CHUNK) };
        if n < 0 {
            err = Some(format!("read: {}", std::io::Error::last_os_error()));
            break;
        }
        if n == 0 {
            break; // EOF
        }
        total += n as u64;
    }
    let secs = t0.elapsed().as_secs_f64();
    unsafe {
        libc::free(buf as *mut c_void);
        libc::close(fd);
    }
    match err {
        Some(e) if total == 0 => Err(e),
        Some(_) | None => Ok((total, secs)),
    }
}

/// Buffered fallback — labelled as such in the report (contaminates page cache).
fn read_buffered(path: &Path, want: u64) -> std::io::Result<(u64, f64)> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut buf = vec![0u8; 1 << 20];
    let mut total = 0u64;
    let t0 = std::time::Instant::now();
    while total < want {
        let want_n = buf.len().min((want - total) as usize);
        let n = f.read(&mut buf[..want_n])?;
        if n == 0 {
            break;
        }
        total += n as u64;
    }
    Ok((total, t0.elapsed().as_secs_f64()))
}

/// Quick sequential read of up to 64 MiB of the Tier-3 candidate: O_DIRECT when
/// the filesystem accepts it, buffered otherwise (explicitly labelled).
pub fn read_benchmark(dir: &Path) -> Option<ReadBench> {
    let (file, size) = find_benchmark_file(dir, 1 << 20)?;
    // Whole 1 MiB blocks only: O_DIRECT needs offset/length alignment.
    let want = (size.min(READ_BENCH_BYTES) / (1 << 20)).max(1) * (1 << 20);
    let file_name = file.display().to_string();

    match read_direct(&file, want) {
        Ok((bytes, seconds)) => Some(ReadBench {
            method: "O_DIRECT".into(),
            file: file_name,
            bytes,
            seconds,
            mbps: mbps(bytes, seconds),
            error: None,
        }),
        Err(direct_err) => match read_buffered(&file, want) {
            Ok((bytes, seconds)) => Some(ReadBench {
                method: "buffered".into(),
                file: file_name,
                bytes,
                seconds,
                mbps: mbps(bytes, seconds),
                error: Some(format!(
                    "O_DIRECT unavailable ({direct_err}); buffered fallback"
                )),
            }),
            Err(e) => Some(ReadBench {
                method: "none".into(),
                file: file_name,
                bytes: 0,
                seconds: 0.0,
                mbps: 0.0,
                error: Some(format!("read failed: {e}; O_DIRECT error: {direct_err}")),
            }),
        },
    }
}

fn mbps(bytes: u64, seconds: f64) -> f64 {
    if seconds <= 0.0 {
        0.0
    } else {
        bytes as f64 / 1_000_000.0 / seconds
    }
}

pub fn storage_info(path: &str, bench: bool) -> StorageInfo {
    let exists = Path::new(path).is_dir();
    let (total, free) = if exists {
        statvfs(path).unwrap_or((0, 0))
    } else {
        (0, 0)
    };
    let read_bench = if bench && exists && path == TIER3_PATH {
        read_benchmark(Path::new(path))
    } else {
        None
    };
    StorageInfo {
        path: path.to_string(),
        exists,
        total_bytes: total,
        free_bytes: free,
        read_bench,
    }
}

// ---------------------------------------------------------------------------
// Orchestration
// ---------------------------------------------------------------------------

/// Probe the host. `bench_storage` runs the Tier-3 read benchmark (skipped on the
/// fingerprint fast path — free space is still reported).
pub fn probe_host(bench_storage: bool) -> HostInfo {
    let (cpu_model, cpu_cores, cpu_threads) = cpu_info();
    let (mem_total, mem_available) = meminfo();
    let (mlock_limit, mlock_max) = mlock_limits();
    let mut tiers = TierSizing::default();
    let storage: Vec<StorageInfo> = STORAGE_PATHS
        .iter()
        .map(|p| storage_info(p, bench_storage))
        .collect();
    if let Some(t3) = storage.iter().find(|s| s.path == tiers.tier3_path) {
        tiers.tier3_free_bytes = t3.free_bytes;
    }
    HostInfo {
        hostname: read_trimmed("/proc/sys/kernel/hostname").unwrap_or_else(|| "unknown".into()),
        cpu_model,
        cpu_cores,
        cpu_threads,
        numa_nodes: numa_nodes(),
        mem_total_bytes: mem_total,
        mem_available_bytes: mem_available,
        mlock_limit_bytes: mlock_limit,
        mlock_max_bytes: mlock_max,
        zram: zram_info(),
        io_uring: io_uring_status(),
        storage,
        tiers,
        competing_gpu_processes: competing_gpu_processes(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meminfo_parsing() {
        let text = "MemTotal:       62736435 kB\nMemFree: 100 kB\nMemAvailable:  44000000 kB\n";
        assert_eq!(parse_meminfo_kb(text, "MemTotal:"), Some(62_736_435 * 1024));
        assert_eq!(
            parse_meminfo_kb(text, "MemAvailable:"),
            Some(44_000_000 * 1024)
        );
        assert_eq!(parse_meminfo_kb(text, "SwapTotal:"), None);
    }

    #[test]
    fn cpuinfo_parsing() {
        let text = "\
processor\t: 0\nphysical id\t: 0\ncore id\t: 0\nmodel name\t: AMD Ryzen 9 7950X 16-Core Processor\n\
processor\t: 1\nphysical id\t: 0\ncore id\t: 1\nmodel name\t: AMD Ryzen 9 7950X 16-Core Processor\n\
processor\t: 2\nphysical id\t: 0\ncore id\t: 0\nmodel name\t: AMD Ryzen 9 7950X 16-Core Processor\n";
        let (model, cores, threads) = parse_cpuinfo(text);
        assert_eq!(model, "AMD Ryzen 9 7950X 16-Core Processor");
        assert_eq!(threads, 3);
        assert_eq!(cores, 2, "same physical/core id must count once");
    }

    #[test]
    fn swaps_parsing_picks_zram() {
        let text = "\
Filename\t\t\t\tType\t\tSize\t\tUsed\t\tPriority\n\
/dev/sda2                               partition       8388604         0               -2\n\
/dev/zram0                              partition       61268988        22153844        100\n";
        let (dev, size, used) = parse_swaps_zram(text).expect("zram present");
        assert_eq!(dev, "/dev/zram0");
        assert_eq!(size, 61_268_988 * 1024);
        assert_eq!(used, 22_153_844 * 1024);
        assert!(parse_swaps_zram("Filename Type Size Used Priority\n").is_none());
    }

    #[test]
    fn io_uring_status_from_sysctl() {
        assert!(io_uring_from_sysctl(None).available);
        assert!(io_uring_from_sysctl(Some("0")).available);
        assert!(!io_uring_from_sysctl(Some("1")).available);
        assert!(!io_uring_from_sysctl(Some("2")).available);
    }

    #[test]
    fn connector_detection() {
        let yes = vec!["card1".into(), "card1-DP-1".into(), "renderD129".into()];
        let no = vec!["card1".into(), "renderD129".into()];
        assert!(any_connector(&yes));
        assert!(!any_connector(&no));
        assert!(!any_connector(&[]));

        let connected = vec!["connected".into(), "disconnected".into()];
        assert!(any_connected(&connected));
        assert!(!any_connected(&["disconnected".into(), "unknown".into()]));
        assert!(!any_connected(&[]));
    }

    /// On this machine the RX 6950 XT drives displays (reserve applies) and the
    /// Radeon VII does not (render-only) — docs/phase-0/04.
    #[test]
    fn display_reserve_targets_only_the_driving_card() {
        assert!(
            has_display_outputs("0000:03:00.0"),
            "RX 6950 XT should show connected outputs"
        );
        assert!(
            !has_display_outputs("0000:0e:00.0"),
            "Radeon VII is render-only here"
        );
        assert!(!has_display_outputs(""));
    }

    #[test]
    fn live_host_probe_reads_real_values() {
        let h = probe_host(false);
        assert!(!h.cpu_model.is_empty(), "cpu model must be read");
        assert!(h.cpu_threads >= h.cpu_cores && h.cpu_cores > 0);
        assert!(h.mem_total_bytes > 0);
        assert_eq!(h.numa_nodes, 1, "this host is single-NUMA (phase-0/04)");
        assert!(
            h.mlock_limit_bytes > 0 && h.mlock_limit_bytes <= (1 << 40),
            "effective mlock limit must be a real number, got {}",
            h.mlock_limit_bytes
        );
        assert!(h.zram.is_some(), "zram0 is configured on this host");
        assert!(h.io_uring.available, "io_uring is usable here (phase-0/04)");
        assert_eq!(h.storage.len(), 3);
        assert!(
            h.storage.iter().all(|s| s.exists),
            "all three paths exist here"
        );
        assert!(h.storage.iter().all(|s| s.free_bytes > 0));
        assert_eq!(h.tiers.tier3_path, "/mnt/agent-storage-a");
        assert!(h.tiers.tier3_free_bytes > 0);
    }

    #[test]
    fn statvfs_matches_df_on_root() {
        let (total, free) = statvfs("/").expect("statvfs(/)");
        assert!(total > 100_000_000_000, "root fs is ~778 GiB, got {total}");
        assert!(free > 1_000_000_000 && free < total);
    }
}
