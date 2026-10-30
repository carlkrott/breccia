//! `hw_profile.json` schema (schema_version 1) — serde types only, no probing.
//!
//! Layout:
//! ```json
//! { schema_version, fingerprint, timestamp, quick, devices[], host, notes[] }
//! ```
//! `devices[]` carries identity + Vulkan discovery detail + measured
//! bandwidth/compute/PCIe/overlap + the suggested role; `host` carries
//! CPU/RAM/mlock/zram/io_uring/storage; `notes` carries environment warnings.

use crate::backend::{BackendKind, DeviceMeasurements};
use crate::role::DeviceRole;
use serde::{Deserialize, Serialize};

/// Bump when the JSON shape changes incompatibly (readers must check it).
pub const SCHEMA_VERSION: u32 = 1;

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct HwProfile {
    pub schema_version: u32,
    /// BLAKE3 over stable hardware identity (see [`crate::fingerprint`]).
    pub fingerprint: String,
    /// ISO-8601 UTC rendering of `timestamp_unix_s` (human convenience).
    pub timestamp: String,
    pub timestamp_unix_s: u64,
    /// True when produced with short benchmark windows (`--quick`).
    pub quick: bool,
    pub devices: Vec<DeviceEntry>,
    pub host: HostInfo,
    pub notes: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct DeviceEntry {
    pub identity: DeviceIdentity,
    /// Vulkan-only discovery detail (heaps, queue families, subgroup 32/64 ranges,
    /// extensions/features). `None` for a future non-Vulkan backend entry.
    pub vulkan: Option<breccia_vk::Properties>,
    /// Measured numbers; `backend` inside says which backend produced them.
    pub measurements: DeviceMeasurements,
    pub role: DeviceRole,
    pub role_reason: String,
    pub role_notes: Vec<String>,
    /// `breccia profile --prefill`: which merged-prefill flags this card runs
    /// correctly *and* faster. Optional and additive (older readers ignore it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill: Option<PrefillVerdict>,
}

/// One card's merged-prefill discovery result. Stale (and so not trusted) when
/// the driver or the compiled shaders differ from the ones it was measured with.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct PrefillVerdict {
    pub driver_version: String,
    /// `breccia_kernels::SPIRV_HASH` of the build that measured it.
    pub spirv_hash: String,
    pub timestamp: String,
    /// The model the check ran (the flags are model-shape specific).
    pub model: String,
    /// Prompt length and chunk of the check (one full chunk + a ragged one).
    pub tokens: usize,
    pub chunk: usize,
    /// ms per prompt token, every flag off / every flag on.
    pub off_ms: f64,
    pub on_ms: f64,
    pub flags: Vec<PrefillFlag>,
    /// The prefill chunk this card runs fastest at (parity-checked against 512);
    /// `0` = not measured. Bench and agents use it when `--prefill-chunk` is absent.
    #[serde(default)]
    pub best_chunk: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct PrefillFlag {
    /// `BRECCIA_PF_*` name.
    pub name: String,
    /// The engine's default for this card: correct and not slower.
    pub on: bool,
    /// KL(per-token reference || all-on-but-this) and top-1 agreement, ms/token
    /// with this flag left out.
    pub kl: f64,
    pub top1_same: bool,
    pub ms_without: f64,
    pub reason: String,
}

/// Backend-neutral device identity (the seam a HIP backend fills from
/// `hipDeviceProp_t` + sysfs later — no HIP implementation in Phase 1).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct DeviceIdentity {
    pub backend: BackendKind,
    pub index: u32,
    pub name: String,
    /// `0x1002` style hex, as reported by the backend.
    pub vendor_id: String,
    pub device_id: String,
    /// Stable per-physical-device UUID (Vulkan `deviceUUID`), `""` if unknown.
    pub uuid: String,
    /// DISCRETE_GPU / INTEGRATED_GPU / … (backend vocabulary).
    pub device_type: String,
    pub api_version: String,
    pub driver_name: String,
    pub driver_version: String,
    pub driver_info: String,
    /// `0000:03:00.0`, `""` if unknown.
    pub pci_address: String,
    /// Stable topology string `ADDR up=BRIDGE max=SPEED xWIDTH` (`?` parts allowed).
    pub pci_topology: String,
    /// True when the card exposes display connectors (drives the VRAM reserve).
    pub display_outputs: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct HostInfo {
    pub hostname: String,
    pub cpu_model: String,
    /// Physical cores.
    pub cpu_cores: u32,
    /// Logical threads.
    pub cpu_threads: u32,
    pub numa_nodes: u32,
    pub mem_total_bytes: u64,
    pub mem_available_bytes: u64,
    /// Effective (`RLIMIT_MEMLOCK` soft) limit — 8 MiB by default on this host,
    /// so pinned Tier-2 pools need `LimitMEMLOCK=infinity` (docs/phase-0/04).
    pub mlock_limit_bytes: u64,
    pub mlock_max_bytes: u64,
    pub zram: Option<ZramInfo>,
    pub io_uring: IoUringInfo,
    pub storage: Vec<StorageInfo>,
    pub tiers: TierSizing,
    /// Processes holding GPU memory at probe time (measurement-quality evidence).
    pub competing_gpu_processes: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ZramInfo {
    pub device: String,
    pub disksize_bytes: u64,
    pub used_bytes: u64,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct IoUringInfo {
    pub available: bool,
    /// e.g. `"/proc/sys/fs/io_uring_disabled=0"` or `"sysctl absent"`.
    pub detail: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct StorageInfo {
    pub path: String,
    pub exists: bool,
    pub total_bytes: u64,
    pub free_bytes: u64,
    /// Quick sequential read of the Tier-3 candidate (O_DIRECT when accepted,
    /// otherwise buffered and labelled as such).
    pub read_bench: Option<ReadBench>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ReadBench {
    /// `"O_DIRECT"` or `"buffered"` (fallback, pollutes page cache).
    pub method: String,
    pub file: String,
    pub bytes: u64,
    pub seconds: f64,
    pub mbps: f64,
    pub error: Option<String>,
}

/// Tier sizing defaults from `docs/phase-0/04` (refreshed with live free space).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct TierSizing {
    pub tier2_pinned_start_bytes: u64,
    pub tier2_pinned_max_bytes: u64,
    /// Eviction review trigger: gpu-manager stops tenants below this MemAvailable.
    pub tier2_watermark_mem_available_bytes: u64,
    pub tier3_path: String,
    pub tier3_cap_bytes: u64,
    pub tier3_free_bytes: u64,
}

impl Default for TierSizing {
    fn default() -> Self {
        const GIB: u64 = 1 << 30;
        Self {
            tier2_pinned_start_bytes: 16 * GIB,
            tier2_pinned_max_bytes: 24 * GIB,
            tier2_watermark_mem_available_bytes: 6 * GIB,
            tier3_path: "/mnt/agent-storage-a".to_string(),
            tier3_cap_bytes: 100 * GIB,
            tier3_free_bytes: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Time helpers (std-only: unix seconds + a hand-rolled ISO-8601 UTC rendering)
// ---------------------------------------------------------------------------

pub fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Days-from-civil inverse (Howard Hinnant's algorithm) for `y-m-d`.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `1761464400` → `"2026-10-26T01:00:00Z"`-style string.
pub fn iso8601(unix_s: u64) -> String {
    let days = (unix_s / 86_400) as i64;
    let rem = unix_s % 86_400;
    let (y, m, d) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BackendKind, DeviceMeasurements};

    fn sample_profile() -> HwProfile {
        HwProfile {
            schema_version: SCHEMA_VERSION,
            fingerprint: "deadbeef".into(),
            timestamp: iso8601(1_761_464_400),
            timestamp_unix_s: 1_761_464_400,
            quick: false,
            devices: vec![DeviceEntry {
                identity: DeviceIdentity {
                    backend: BackendKind::Vulkan,
                    index: 1,
                    name: "AMD Radeon RX 6950 XT".into(),
                    vendor_id: "0x1002".into(),
                    device_id: "0x73a5".into(),
                    uuid: "11111111-2222-3333-4444-555555555555".into(),
                    device_type: "DISCRETE_GPU".into(),
                    api_version: "1.3.0".into(),
                    driver_name: "RADV".into(),
                    driver_version: "26.2.4".into(),
                    driver_info: "Mesa 26.2.4".into(),
                    pci_address: "0000:03:00.0".into(),
                    pci_topology: "0000:03:00.0 up=0000:00:01.1 max=16.0 GT/s x16".into(),
                    display_outputs: false,
                },
                vulkan: None,
                measurements: DeviceMeasurements {
                    backend: BackendKind::Vulkan,
                    ..Default::default()
                },
                role: crate::role::DeviceRole::StreamingCompute,
                role_reason: "test".into(),
                role_notes: vec!["note".into()],
                prefill: None,
            }],
            host: HostInfo {
                hostname: "host".into(),
                cpu_model: "Ryzen 9 7950X".into(),
                cpu_cores: 16,
                cpu_threads: 32,
                numa_nodes: 1,
                mem_total_bytes: 62_700_000_000,
                mem_available_bytes: 46_000_000_000,
                mlock_limit_bytes: 8 << 20,
                mlock_max_bytes: 8 << 20,
                zram: None,
                io_uring: IoUringInfo {
                    available: true,
                    detail: "sysctl absent".into(),
                },
                storage: vec![StorageInfo {
                    path: "/".into(),
                    exists: true,
                    total_bytes: 800,
                    free_bytes: 121,
                    read_bench: Some(ReadBench {
                        method: "O_DIRECT".into(),
                        file: "x.bin".into(),
                        bytes: 64 << 20,
                        seconds: 0.2,
                        mbps: 349.0,
                        error: None,
                    }),
                }],
                tiers: TierSizing::default(),
                competing_gpu_processes: vec![],
            },
            notes: vec!["n1".into()],
        }
    }

    #[test]
    fn json_round_trip_is_lossless() {
        let p = sample_profile();
        let s = serde_json::to_string_pretty(&p).expect("serialize");
        let back: HwProfile = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(p, back);
        assert_eq!(back.schema_version, SCHEMA_VERSION);
        assert_eq!(back.devices[0].role, DeviceRole::StreamingCompute);
        assert_eq!(
            back.host.storage[0].read_bench.as_ref().unwrap().mbps,
            349.0
        );
    }

    #[test]
    fn iso8601_matches_known_epochs() {
        assert_eq!(iso8601(0), "1970-01-01T00:00:00Z");
        // 2026-10-26T00:00:00Z
        assert_eq!(iso8601(1_792_972_800), "2026-10-26T00:00:00Z");
        // leap-day sanity: 2024-02-29T12:34:56Z
        assert_eq!(iso8601(1_709_210_096), "2024-02-29T12:34:56Z");
    }

    #[test]
    fn tier_defaults_match_phase04() {
        const GIB: u64 = 1 << 30;
        let t = TierSizing::default();
        assert_eq!(t.tier2_pinned_start_bytes, 16 * GIB);
        assert_eq!(t.tier2_pinned_max_bytes, 24 * GIB);
        assert_eq!(t.tier3_cap_bytes, 100 * GIB);
        assert_eq!(t.tier3_path, "/mnt/agent-storage-a");
    }
}
