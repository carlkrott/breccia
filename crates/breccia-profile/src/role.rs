//! Device-role heuristic (Gate A, `docs/phase-0/gate-A-report.md` §2).
//!
//! The role is derived **from measured numbers, never from PCI IDs** — the PCI IDs
//! below appear only as comments/tests documenting the hardware this was tuned on.
//!
//! Gate-A facts the thresholds encode:
//!
//! | class | DRAM copy | PCIe H2D | overlap | role |
//! |---|---|---|---|---|
//! | RX 6950 XT (`1002:73a5`, gfx1030) | 462 GB/s (+3.9× cache lever) | ~28 GB/s | ~100 % | [`DeviceRole::StreamingCompute`] |
//! | Radeon VII / MI50 (`1002:66af`, gfx906) | 826 GB/s HBM | ~2.9 GB/s (gen3×4) | ~0 % | [`DeviceRole::StaticResident`] |
//! | Raphael iGPU (`1002:164e`) | — (not benchmarked) | — | — | [`DeviceRole::DisplayOnly`] |
//!
//! Binding decisions this encodes: the MI50 is excluded from PCIe streaming
//! (transfers serialize with compute), the RX is the streaming/expert-paging home,
//! and expert-parallel placement is the default (no per-token cross-device traffic).

use serde::{Deserialize, Serialize};

/// Minimum steady-state host link for a streaming role (GB/s, H2D).
/// RX-class ≈ 28, MI50-class ≈ 2.9 → 10 sits an order of magnitude from both.
pub const STREAM_MIN_PCIE_GBPS: f64 = 10.0;
/// Minimum compute/transfer overlap for a streaming role (%).
/// RX ≈ 100, MI50 ≈ 0-2 → 50 is "DMA hidden under compute".
pub const STREAM_MIN_OVERLAP_PCT: f64 = 50.0;
/// Device bandwidth at/above this is HBM-class (GB/s copy).
pub const HBM_MIN_COPY_GBPS: f64 = 600.0;
/// Cache-resident window ≥ this × the DRAM window ⇒ large on-die cache lever.
pub const CACHE_LEVER_MIN_RATIO: f64 = 1.5;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DeviceRole {
    /// Fast host link **and** DMA that hides under compute → streaming / expert
    /// paging home, prefill, drafter (RX-class).
    StreamingCompute,
    /// High device bandwidth but a host link that is slow and/or serial →
    /// static resident weights + bandwidth-bound decode, no streaming lane
    /// (MI50-class).
    StaticResident,
    /// Integrated / display device: not part of the compute plan.
    DisplayOnly,
    /// Benchmarks missing or failed — decide nothing.
    Unmeasured,
}

impl std::fmt::Display for DeviceRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            DeviceRole::StreamingCompute => "streaming-compute",
            DeviceRole::StaticResident => "static-resident",
            DeviceRole::DisplayOnly => "display-only",
            DeviceRole::Unmeasured => "unmeasured",
        };
        f.write_str(s)
    }
}

/// Everything [`suggest_role`] reads. All fields measured (or `None` if skipped).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RoleInput {
    /// Vulkan `device_type` vocabulary: `DISCRETE_GPU`, `INTEGRATED_GPU`, …
    pub device_type: String,
    /// DRAM-resident bandwidth window (copy = read+write per pass), GB/s.
    pub dram_copy_gbps: Option<f64>,
    /// Small, cache-resident bandwidth window, GB/s (Infinity-Cache lever).
    pub cache_copy_gbps: Option<f64>,
    /// Steady-state PCIe host↔device, GB/s (max over the measured sizes ≥ 64 MiB).
    pub pcie_h2d_gbps: Option<f64>,
    pub pcie_d2h_gbps: Option<f64>,
    /// Compute/transfer overlap efficiency, % (100 = DMA fully hidden).
    pub overlap_efficiency_pct: Option<f64>,
    pub fp32_gflops: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RoleDecision {
    pub role: DeviceRole,
    pub reason: String,
    pub notes: Vec<String>,
}

fn r1(v: Option<f64>) -> String {
    v.map(|v| format!("{v:.1}")).unwrap_or_else(|| "?".into())
}

/// The Gate-A role heuristic. Pure, documented, unit-tested on phase-0 numbers.
pub fn suggest_role(input: &RoleInput) -> RoleDecision {
    // 1. Non-discrete devices are not compute targets.
    if input.device_type != "DISCRETE_GPU" {
        let reason = if input.device_type == "INTEGRATED_GPU" {
            "integrated GPU: display/composition device, excluded from the compute plan".to_string()
        } else {
            format!(
                "device type {} is not a discrete compute target",
                input.device_type
            )
        };
        return RoleDecision {
            role: DeviceRole::DisplayOnly,
            reason,
            notes: vec![],
        };
    }

    // 2. Refuse to guess from partial evidence.
    let (h2d, d2h, overlap) = match (input.pcie_h2d_gbps, input.overlap_efficiency_pct) {
        (Some(h), Some(o)) => (h, input.pcie_d2h_gbps.unwrap_or(f64::NAN), o),
        _ => {
            let mut missing = Vec::new();
            if input.pcie_h2d_gbps.is_none() {
                missing.push("pcie");
            }
            if input.overlap_efficiency_pct.is_none() {
                missing.push("overlap");
            }
            return RoleDecision {
                role: DeviceRole::Unmeasured,
                reason: format!("missing measurements: {}", missing.join(", ")),
                notes: vec![],
            };
        }
    };

    let mut notes = Vec::new();
    if let (Some(dram), Some(cache)) = (input.dram_copy_gbps, input.cache_copy_gbps) {
        if dram > 0.0 && cache / dram >= CACHE_LEVER_MIN_RATIO {
            notes.push(format!(
                "on-die cache lever: cache-resident window {cache:.0} GB/s = {:.1}× the DRAM \
                 window ({dram:.0} GB/s) — keep hot expert/KV working sets inside it",
                cache / dram
            ));
        }
    }
    if let Some(fp32) = input.fp32_gflops {
        notes.push(format!("FP32 sustained FMA: {fp32:.0} GFLOPS"));
    }

    // 3. The two measured conditions that make streaming pay.
    if h2d >= STREAM_MIN_PCIE_GBPS && overlap >= STREAM_MIN_OVERLAP_PCT {
        notes.push(format!(
            "streaming lane: {h2d:.1} GB/s H2D with {overlap:.0}% overlap \
             (DMA hides under compute)"
        ));
        return RoleDecision {
            role: DeviceRole::StreamingCompute,
            reason: format!(
                "PCIe H2D {h2d:.1} GB/s ≥ {STREAM_MIN_PCIE_GBPS} and overlap \
                 {overlap:.0}% ≥ {STREAM_MIN_OVERLAP_PCT} → DMA can be hidden under compute: \
                 streaming/expert-paging home, prefill and drafter"
            ),
            notes,
        };
    }

    // 4. Everything else keeps weights resident.
    let dram = input.dram_copy_gbps;
    let bandwidth_part = match dram {
        Some(d) if d >= HBM_MIN_COPY_GBPS => {
            format!("HBM-class device bandwidth ({d:.0} GB/s copy) but ")
        }
        Some(d) => format!("device bandwidth {d:.0} GB/s and "),
        None => String::new(),
    };
    let serial = overlap < STREAM_MIN_OVERLAP_PCT;
    notes.push(format!(
        "host link {h2d:.1} GB/s H2D / {} GB/s D2H with {overlap:.0}% overlap ({})",
        r1(if d2h.is_nan() { None } else { Some(d2h) }),
        if serial {
            "transfers serialize with compute"
        } else {
            "link below streaming threshold"
        }
    ));
    RoleDecision {
        role: DeviceRole::StaticResident,
        reason: format!(
            "{bandwidth_part}host link {h2d:.1} GB/s (threshold {STREAM_MIN_PCIE_GBPS}) and/or \
             overlap {overlap:.0}% (threshold {STREAM_MIN_OVERLAP_PCT}) — no PCIe streaming to \
             this device; keep weights static-resident and use it for bandwidth-bound decode"
        ),
        notes,
    }
}

/// Steady-state PCIe numbers: max over the measured sizes ≥ 64 MiB (transient dips
/// in a single size are ignored; see `docs/phase-0/02` §3.3), falling back to the
/// max over all sizes when only smaller windows were measured (`--quick`).
pub fn steady_state_pcie(sizes: &[breccia_vk::PcieSize]) -> Option<(f64, f64)> {
    if sizes.is_empty() {
        return None;
    }
    let pool: Vec<&breccia_vk::PcieSize> = if sizes.iter().any(|s| s.size_mb >= 64) {
        sizes.iter().filter(|s| s.size_mb >= 64).collect()
    } else {
        sizes.iter().collect()
    };
    // Per-direction steady state: a transient dip in one direction of one size
    // must not drag the other direction down.
    let h2d = pool.iter().map(|s| s.h2d_gbps).fold(f64::MIN, f64::max);
    let d2h = pool.iter().map(|s| s.d2h_gbps).fold(f64::MIN, f64::max);
    Some((h2d, d2h))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// docs/phase-0/02 §3.1/3.3/3.4 for the RX 6950 XT (1002:73a5).
    fn rx_input() -> RoleInput {
        RoleInput {
            device_type: "DISCRETE_GPU".into(),
            dram_copy_gbps: Some(462.2),
            cache_copy_gbps: Some(1811.2),
            pcie_h2d_gbps: Some(28.7),
            pcie_d2h_gbps: Some(28.0),
            overlap_efficiency_pct: Some(102.3),
            fp32_gflops: Some(26_704.0),
        }
    }

    /// docs/phase-0/02 for the Radeon VII / MI50 (1002:66af).
    fn mi50_input() -> RoleInput {
        RoleInput {
            device_type: "DISCRETE_GPU".into(),
            dram_copy_gbps: Some(826.1),
            cache_copy_gbps: Some(792.5),
            pcie_h2d_gbps: Some(2.9),
            pcie_d2h_gbps: Some(2.9),
            overlap_efficiency_pct: Some(0.8),
            fp32_gflops: Some(6_784.0),
        }
    }

    #[test]
    fn rx_class_is_streaming_compute() {
        let d = suggest_role(&rx_input());
        assert_eq!(d.role, DeviceRole::StreamingCompute, "{}", d.reason);
        // Infinity-Cache lever must be called out.
        assert!(
            d.notes.iter().any(|n| n.contains("cache lever")),
            "{:?}",
            d.notes
        );
    }

    #[test]
    fn mi50_class_is_static_resident() {
        let d = suggest_role(&mi50_input());
        assert_eq!(d.role, DeviceRole::StaticResident, "{}", d.reason);
        assert!(d.reason.contains("HBM-class"), "{}", d.reason);
        assert!(
            d.notes.iter().any(|n| n.contains("serialize")),
            "{:?}",
            d.notes
        );
    }

    #[test]
    fn good_link_without_overlap_is_not_streaming() {
        // Link is fast but DMA cannot hide (hypothetical degenerate driver).
        let mut i = rx_input();
        i.overlap_efficiency_pct = Some(5.0);
        assert_eq!(suggest_role(&i).role, DeviceRole::StaticResident);
        // …and the converse: overlap without link bandwidth.
        let mut i = mi50_input();
        i.overlap_efficiency_pct = Some(95.0);
        assert_eq!(suggest_role(&i).role, DeviceRole::StaticResident);
    }

    #[test]
    fn integrated_gpu_is_display_only() {
        let d = suggest_role(&RoleInput {
            device_type: "INTEGRATED_GPU".into(),
            ..Default::default()
        });
        assert_eq!(d.role, DeviceRole::DisplayOnly);
    }

    #[test]
    fn missing_measurements_never_guess() {
        let d = suggest_role(&RoleInput {
            device_type: "DISCRETE_GPU".into(),
            ..Default::default()
        });
        assert_eq!(d.role, DeviceRole::Unmeasured);
        assert!(
            d.reason.contains("pcie") && d.reason.contains("overlap"),
            "{}",
            d.reason
        );
        // PCIe alone is not enough either.
        let d = suggest_role(&RoleInput {
            device_type: "DISCRETE_GPU".into(),
            pcie_h2d_gbps: Some(28.0),
            ..Default::default()
        });
        assert_eq!(d.role, DeviceRole::Unmeasured);
    }

    /// Documented threshold table (phase-0 numbers): intentionally literal so the
    /// measured separation is visible in one place.
    #[allow(clippy::assertions_on_constants)]
    #[test]
    fn thresholds_separate_the_two_cards() {
        assert!(28.7 >= STREAM_MIN_PCIE_GBPS && 2.9 < STREAM_MIN_PCIE_GBPS);
        assert!(102.3 >= STREAM_MIN_OVERLAP_PCT && 0.8 < STREAM_MIN_OVERLAP_PCT);
        assert!(826.1 >= HBM_MIN_COPY_GBPS && 462.2 < HBM_MIN_COPY_GBPS);
        assert!(1811.2 / 462.2 >= CACHE_LEVER_MIN_RATIO);
        assert!(792.5 / 826.1 < CACHE_LEVER_MIN_RATIO);
    }

    #[test]
    fn steady_state_prefers_large_sizes_and_ignores_dips() {
        use breccia_vk::PcieSize;
        let sizes = vec![
            PcieSize {
                size_mb: 16,
                h2d_gbps: 26.2,
                d2h_gbps: 27.3,
                h2d_seconds_per_copy: 0.0,
                d2h_seconds_per_copy: 0.0,
            },
            PcieSize {
                size_mb: 64,
                h2d_gbps: 28.66,
                d2h_gbps: 21.97,
                h2d_seconds_per_copy: 0.0,
                d2h_seconds_per_copy: 0.0,
            },
            PcieSize {
                size_mb: 256,
                h2d_gbps: 23.30,
                d2h_gbps: 28.0,
                h2d_seconds_per_copy: 0.0,
                d2h_seconds_per_copy: 0.0,
            },
            PcieSize {
                size_mb: 1024,
                h2d_gbps: 27.34,
                d2h_gbps: 28.06,
                h2d_seconds_per_copy: 0.0,
                d2h_seconds_per_copy: 0.0,
            },
        ];
        let (h2d, d2h) = steady_state_pcie(&sizes).unwrap();
        assert!((h2d - 28.66).abs() < 1e-9, "{h2d}");
        assert!((d2h - 28.06).abs() < 1e-9, "{d2h}");
        assert!(steady_state_pcie(&[]).is_none());
        // quick mode: only 16/64 MiB sizes exist → fall back to all sizes.
        let quick = &sizes[..2];
        let (h2d, _) = steady_state_pcie(quick).unwrap();
        assert!((h2d - 28.66).abs() < 1e-9);
    }
}
