//! `breccia-profile` — hardware discovery + calibration → `hw_profile.json`.
//!
//! Orchestration pieces (the CLI composes them):
//!
//! * [`current_fingerprint`] — cheap identity pass (no benchmarks) used to decide
//!   whether the cached profile is still valid;
//! * [`calibrate`] — full run: Vulkan micro-benchmarks + host probe + role
//!   assignment + fingerprint;
//! * [`load_profile`] / [`save_profile`] — `hw_profile.json` I/O;
//! * [`summary_text`] / [`plan_text`] — human output (`profile` and
//!   `--dry-run-plan` respectively).
//!
//! Backend seam: [`backend::BackendProbe`] (Vulkan implemented; HIP is a design
//! slot only — see `backend.rs`).

pub mod backend;
pub mod fingerprint;
pub mod plan;
pub mod role;
pub mod schema;
pub mod system;

use anyhow::{Context, Result};
use backend::{BackendDevice, BackendProbe, BackendRun, VulkanBackend};
use breccia_vk::ProbeOptions;
use fingerprint::FingerprintInput;
use role::{steady_state_pcie, suggest_role, RoleInput};
use schema::{DeviceEntry, DeviceIdentity, HostInfo, HwProfile};
use std::path::{Path, PathBuf};

/// `$XDG_CONFIG_HOME/breccia/hw_profile.json`, else `~/.config/breccia/hw_profile.json`.
pub fn default_profile_path() -> PathBuf {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
        .or_else(|| {
            std::env::var_os("HOME")
                .map(|h| PathBuf::from(h).join(".config"))
                .filter(|p| !p.as_os_str().is_empty())
        })
        .unwrap_or_else(|| PathBuf::from(".config"));
    base.join("breccia").join("hw_profile.json")
}

pub fn load_profile(path: &Path) -> Result<Option<HwProfile>> {
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading profile {}", path.display()))?;
    let p: HwProfile = serde_json::from_str(&text)
        .with_context(|| format!("parsing profile {}", path.display()))?;
    Ok(Some(p))
}

pub fn save_profile(profile: &HwProfile, path: &Path) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let text = serde_json::to_string_pretty(profile).context("serializing hw_profile")?;
    std::fs::write(path, text + "\n").with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// A cached profile is usable when the schema still matches, the hardware
/// fingerprint is unchanged, and its benchmark window is at least as long as the
/// one requested (a full profile answers a `--quick` request, never the reverse).
pub fn cache_usable(cached: &HwProfile, current_fingerprint: &str, quick_requested: bool) -> bool {
    cached.schema_version == schema::SCHEMA_VERSION
        && cached.fingerprint == current_fingerprint
        && (!cached.quick || quick_requested)
}

/// Identity-only pass: cheap enough for every invocation (a few sysfs/proc reads
/// plus one Vulkan enumeration, no benchmarks).
#[derive(Debug, Clone)]
pub struct CurrentFingerprint {
    pub fingerprint: String,
    pub identities: Vec<DeviceIdentity>,
    pub host: HostInfo,
    pub warnings: Vec<String>,
}

/// Warning text for the `MESA_VK_DEVICE_SELECT` trap (pure — unit-tested;
/// the actual variable is cleared inside `breccia-vk` before instance creation).
pub fn device_select_warning(value: Option<&str>) -> Option<String> {
    value.map(|v| {
        format!(
            "MESA_VK_DEVICE_SELECT={v} was set in this environment; breccia cleared it before \
             enumeration (it hides both discrete GPUs — docs/PROJECT-SETUP.md)"
        )
    })
}

pub fn current_fingerprint() -> Result<CurrentFingerprint> {
    // Capture before discovery clears it (breccia-vk removes the variable).
    let trap = std::env::var("MESA_VK_DEVICE_SELECT").ok();
    let host = system::probe_host(false);
    let identities = VulkanBackend.discover().context("Vulkan discovery")?;
    let warnings = device_select_warning(trap.as_deref()).into_iter().collect();
    let fingerprint =
        fingerprint::fingerprint(&FingerprintInput::from_identities(&identities, &host));
    Ok(CurrentFingerprint {
        fingerprint,
        identities,
        host,
        warnings,
    })
}

/// Role inputs from measured values only (PCI IDs never reach the heuristic).
fn role_input(d: &BackendDevice) -> RoleInput {
    let m = &d.measurements;
    let (h2d, d2h) = m
        .pcie
        .as_ref()
        .and_then(|p| steady_state_pcie(&p.sizes))
        .map(|(h, d)| (Some(h), Some(d)))
        .unwrap_or((None, None));
    RoleInput {
        device_type: d.identity.device_type.clone(),
        dram_copy_gbps: m.vram_bandwidth.as_ref().map(|b| b.copy_gbps),
        cache_copy_gbps: m.vram_bandwidth_cache.as_ref().map(|b| b.copy_gbps),
        pcie_h2d_gbps: h2d,
        pcie_d2h_gbps: d2h,
        overlap_efficiency_pct: m.overlap.as_ref().map(|o| o.efficiency_percent),
        fp32_gflops: m.compute.as_ref().map(|c| c.fp32_gflops),
    }
}

/// Assemble the profile from a backend run + host probe (pure — no I/O).
pub fn build_profile(
    run: BackendRun,
    host: HostInfo,
    quick: bool,
    mut notes: Vec<String>,
) -> HwProfile {
    let identities: Vec<DeviceIdentity> = run.devices.iter().map(|d| d.identity.clone()).collect();
    let fp = fingerprint::fingerprint(&FingerprintInput::from_identities(&identities, &host));

    let devices: Vec<DeviceEntry> = run
        .devices
        .into_iter()
        .map(|d| {
            let input = role_input(&d);
            let decision = suggest_role(&input);
            DeviceEntry {
                identity: d.identity,
                vulkan: d.vulkan,
                measurements: d.measurements,
                role: decision.role,
                role_reason: decision.reason,
                role_notes: decision.notes,
                prefill: None,
            }
        })
        .collect();

    let ts = schema::now_unix();
    let mut all_notes = run.warnings;
    all_notes.append(&mut notes);
    if quick {
        all_notes.push(
            "quick mode: short benchmark windows — treat absolute numbers as indicative; \
             re-run without --quick for gate evidence"
                .to_string(),
        );
    }
    if !host.competing_gpu_processes.is_empty() {
        all_notes.push(format!(
            "{} GPU-related process(es) alive at probe time: {}",
            host.competing_gpu_processes.len(),
            host.competing_gpu_processes.join(" | ")
        ));
    }

    HwProfile {
        schema_version: schema::SCHEMA_VERSION,
        fingerprint: fp,
        timestamp: schema::iso8601(ts),
        timestamp_unix_s: ts,
        quick,
        devices,
        host,
        notes: all_notes,
    }
}

/// Full calibration: host probe (with the Tier-3 read benchmark) + Vulkan
/// benchmarks + roles + fingerprint.
pub fn calibrate(quick: bool) -> Result<HwProfile> {
    let trap = std::env::var("MESA_VK_DEVICE_SELECT").ok();
    let host = system::probe_host(true);
    let run = VulkanBackend
        .probe(&ProbeOptions {
            quick,
            ..Default::default()
        })
        .context("Vulkan probe")?;
    let mut notes = Vec::new();
    if let Some(w) = device_select_warning(trap.as_deref()) {
        notes.push(w);
    }
    Ok(build_profile(run, host, quick, notes))
}

/// Human summary of a profile (used after both the cached and the fresh path).
pub fn summary_text(profile: &HwProfile) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "hw profile: {} run, {} — schema v{}\n",
        if profile.quick { "quick" } else { "full" },
        profile.timestamp,
        profile.schema_version
    ));
    out.push_str(&format!("fingerprint: {}\n", profile.fingerprint));

    let h = &profile.host;
    out.push_str(&format!(
        "host: {} ({}C/{}T), NUMA {}, RAM {} ({:.1} GB available), mlock {}, io_uring {}, zram {}\n",
        h.cpu_model,
        h.cpu_cores,
        h.cpu_threads,
        h.numa_nodes,
        fmt_bytes(h.mem_total_bytes),
        h.mem_available_bytes as f64 / 1e9,
        if h.mlock_limit_bytes == u64::MAX {
            "unlimited".to_string()
        } else {
            fmt_bytes(h.mlock_limit_bytes)
        },
        if h.io_uring.available { "available" } else { "unavailable" },
        h.zram
            .as_ref()
            .map(|z| format!("{} {}", z.device, fmt_bytes(z.disksize_bytes)))
            .unwrap_or_else(|| "none".into()),
    ));
    out.push_str(&format!(
        "storage: {}\n",
        h.storage
            .iter()
            .map(|s| {
                let bench = s
                    .read_bench
                    .as_ref()
                    .map(|r| format!(", {} {:.0} MB/s", r.method, r.mbps))
                    .unwrap_or_default();
                format!("{} {} free{}", s.path, fmt_bytes(s.free_bytes), bench)
            })
            .collect::<Vec<_>>()
            .join(" | ")
    ));

    out.push_str("devices:\n");
    for e in &profile.devices {
        let m = &e.measurements;
        let dram = m
            .vram_bandwidth
            .as_ref()
            .map(|b| format!("{:.0} GB/s DRAM", b.copy_gbps))
            .unwrap_or_else(|| "not benchmarked".into());
        let cache = m
            .vram_bandwidth_cache
            .as_ref()
            .map(|b| format!(", cache window {:.0} GB/s", b.copy_gbps));
        let pcie = m
            .pcie
            .as_ref()
            .and_then(|p| steady_state_pcie(&p.sizes))
            .map(|(h2d, d2h)| format!(", PCIe {h2d:.1}/{d2h:.1} GB/s"));
        let overlap = m
            .overlap
            .as_ref()
            .map(|o| format!(", overlap {:.0}%", o.efficiency_percent));
        let fp32 = m
            .compute
            .as_ref()
            .map(|c| format!(", FP32 {:.0} GFLOPS", c.fp32_gflops));
        out.push_str(&format!(
            "  [{}] {} ({})\n      role: {} — {}\n      {}{}{}{}{}\n",
            e.identity.index,
            e.identity.name,
            e.identity.device_type,
            e.role,
            e.role_reason,
            dram,
            cache.unwrap_or_default(),
            pcie.unwrap_or_default(),
            overlap.unwrap_or_default(),
            fp32.unwrap_or_default(),
        ));
        if let Some(err) = &e.measurements.error {
            out.push_str(&format!("      ERROR: {err}\n"));
        }
    }

    if !profile.notes.is_empty() {
        out.push_str("notes:\n");
        for n in &profile.notes {
            out.push_str(&format!("  - {n}\n"));
        }
    }
    out
}

pub fn fmt_bytes(b: u64) -> String {
    if b == u64::MAX {
        return "unlimited".to_string();
    }
    const GIB: f64 = (1u64 << 30) as f64;
    const MIB: f64 = (1u64 << 20) as f64;
    let f = b as f64;
    if f >= GIB {
        format!("{:.2} GiB", f / GIB)
    } else if f >= MIB {
        format!("{:.1} MiB", f / MIB)
    } else {
        format!("{b} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::DeviceMeasurements;

    fn fake_run(identities: Vec<DeviceIdentity>) -> BackendRun {
        BackendRun {
            devices: identities
                .into_iter()
                .map(|identity| BackendDevice {
                    identity,
                    vulkan: None,
                    measurements: DeviceMeasurements::default(),
                })
                .collect(),
            warnings: vec!["trap warning".into()],
        }
    }

    /// The fast path (fingerprint-only) and the calibrated path must agree —
    /// otherwise every run would re-benchmark.
    #[test]
    fn fingerprint_matches_between_fast_path_and_build() {
        let current = current_fingerprint().expect("fingerprint pass");
        assert_eq!(current.fingerprint.len(), 64);
        assert!(
            current.identities.len() >= 3,
            "{} devices",
            current.identities.len()
        );

        let profile = build_profile(
            fake_run(current.identities.clone()),
            current.host.clone(),
            false,
            vec![],
        );
        assert_eq!(
            profile.fingerprint, current.fingerprint,
            "fingerprint must not depend on whether benchmarks ran"
        );
        assert_eq!(profile.schema_version, schema::SCHEMA_VERSION);
        // iGPU is display-only, discrete cards are unmeasured (no benchmarks ran).
        let igpu = profile
            .devices
            .iter()
            .find(|d| d.identity.device_type == "INTEGRATED_GPU")
            .expect("iGPU enumerated");
        assert_eq!(igpu.role, role::DeviceRole::DisplayOnly);
        assert!(profile
            .devices
            .iter()
            .filter(|d| d.identity.device_type == "DISCRETE_GPU")
            .all(|d| d.role == role::DeviceRole::Unmeasured));
    }

    #[test]
    fn cache_rules_prevent_quick_from_shadowing_full() {
        let current = current_fingerprint().expect("fingerprint pass");
        let mut cached = build_profile(
            fake_run(current.identities.clone()),
            current.host.clone(),
            false,
            vec![],
        );
        // full profile, same fingerprint → usable for both requests
        assert!(cache_usable(&cached, &current.fingerprint, false));
        assert!(cache_usable(&cached, &current.fingerprint, true));
        // quick profile → only satisfies a quick request
        cached.quick = true;
        assert!(cache_usable(&cached, &current.fingerprint, true));
        assert!(!cache_usable(&cached, &current.fingerprint, false));
        // fingerprint change → never usable
        let mut other = cached.clone();
        other.fingerprint = "0".repeat(64);
        assert!(!cache_usable(&other, &current.fingerprint, true));
        // schema bump → never usable
        let mut old = cached.clone();
        old.schema_version = 0;
        assert!(!cache_usable(&old, &current.fingerprint, true));
    }

    /// The MESA trap is reported when the variable was present (the live env on
    /// this machine sets it; other tests in this process may have cleared it first,
    /// so the deterministic check is the pure function).
    #[test]
    fn device_select_trap_is_reported() {
        let w = device_select_warning(Some("1002:164e!")).expect("warning text");
        assert!(w.contains("MESA_VK_DEVICE_SELECT=1002:164e!"), "{w}");
        assert!(w.contains("hides both discrete GPUs"), "{w}");
        assert!(device_select_warning(None).is_none());
    }

    #[test]
    fn default_profile_path_respects_xdg() {
        // XDG_CONFIG_HOME wins when set (set/unset here is process-local and
        // tests run in one process, so only assert the shape).
        let p = default_profile_path();
        assert!(p.ends_with("breccia/hw_profile.json"), "{}", p.display());
    }

    #[test]
    fn fmt_bytes_scales() {
        assert_eq!(fmt_bytes(8 << 20), "8.0 MiB");
        assert_eq!(fmt_bytes(62_736_435_200), "58.43 GiB");
        assert_eq!(fmt_bytes(u64::MAX), "unlimited");
        assert_eq!(fmt_bytes(512), "512 B");
    }

    #[test]
    fn save_and_load_round_trip() {
        let current = current_fingerprint().expect("fingerprint pass");
        let profile = build_profile(fake_run(current.identities), current.host, true, vec![]);
        let dir = std::env::temp_dir().join(format!("breccia-profile-test-{}", std::process::id()));
        let path = dir.join("hw_profile.json");
        save_profile(&profile, &path).expect("save");
        let back = load_profile(&path).expect("load").expect("exists");
        assert_eq!(profile, back);
        assert!(back.quick);
        std::fs::remove_dir_all(&dir).ok();
    }
}
