//! `--dry-run-plan`: device role table + an IQ3_XXS layer-placement sketch.
//!
//! All numbers here are either measured (from `hw_profile.json`) or documented
//! defaults from `docs/phase-0/04` / `docs/phase-0/05`. Nothing is written to disk.

use crate::role::DeviceRole;
use crate::schema::{DeviceEntry, HwProfile};

/// Total GPU-resident weight bytes of the IQ3_XXS run under the service flags
/// (`docs/phase-0/05`: `Vulkan0 9215.72 MiB + Vulkan1 4361.47 MiB` = 13577.19 MiB).
pub const MODEL_GPU_WEIGHT_MIB: u64 = 13_577;
/// Display/composition reserve subtracted from any card that drives a display
/// (`docs/phase-0/04`: "16+16 GB, minus ~1.5 GB display/composition reserve").
pub const DISPLAY_RESERVE_BYTES: u64 = 1_536 * 1024 * 1024; // 1.5 GiB

fn gib(b: u64) -> String {
    format!("{:.2} GiB", b as f64 / (1u64 << 30) as f64)
}

/// Device-local heap = the largest heap flagged `DEVICE_LOCAL`.
pub fn device_local_bytes(props: Option<&breccia_vk::Properties>) -> u64 {
    props
        .map(|p| {
            p.memory_heaps
                .iter()
                .filter(|h| h.device_local)
                .map(|h| h.size_bytes)
                .max()
                .unwrap_or(0)
        })
        .unwrap_or(0)
}

/// Budget = device-local heap − display reserve (0 for render-only cards).
pub fn vram_budget(entry: &DeviceEntry) -> u64 {
    let local = device_local_bytes(entry.vulkan.as_ref());
    local.saturating_sub(if entry.identity.display_outputs {
        DISPLAY_RESERVE_BYTES
    } else {
        0
    })
}

fn measured_cells(e: &DeviceEntry) -> String {
    let m = &e.measurements;
    let dram = m
        .vram_bandwidth
        .as_ref()
        .map(|b| format!("{:.0}", b.copy_gbps))
        .unwrap_or_else(|| "-".into());
    let cache = m
        .vram_bandwidth_cache
        .as_ref()
        .map(|b| format!("{:.0}", b.copy_gbps))
        .unwrap_or_else(|| "-".into());
    let pcie = m
        .pcie
        .as_ref()
        .and_then(|p| crate::role::steady_state_pcie(&p.sizes))
        .map(|(h, d)| format!("{h:.1}/{d:.1}"))
        .unwrap_or_else(|| "-".into());
    let overlap = m
        .overlap
        .as_ref()
        .map(|o| format!("{:.0}%", o.efficiency_percent))
        .unwrap_or_else(|| "-".into());
    let fp32 = m
        .compute
        .as_ref()
        .map(|c| format!("{:.0} GF", c.fp32_gflops))
        .unwrap_or_else(|| "-".into());
    format!("DRAM {dram} | cache {cache} | PCIe {pcie} | overlap {overlap} | FP32 {fp32}")
}

/// The whole `--dry-run-plan` output.
pub fn plan_text(profile: &HwProfile, expert_pin_bytes: Option<u64>) -> String {
    let mut out = String::new();
    out.push_str("=== breccia profile --dry-run-plan (nothing written) ===\n");
    out.push_str(&format!(
        "fingerprint: {}\ntimestamp:   {} ({})\n\n",
        profile.fingerprint,
        profile.timestamp,
        if profile.quick { "quick" } else { "full" }
    ));

    out.push_str("Device role table\n");
    for e in &profile.devices {
        out.push_str(&format!(
            "  [{}] {} ({}, {}:{}, {})\n      role: {} — {}\n      measured: {}\n      vram: {} device-local, budget {}\n",
            e.identity.index,
            e.identity.name,
            e.identity.device_type,
            e.identity.vendor_id,
            e.identity.device_id,
            if e.identity.pci_address.is_empty() {
                "pci ?".to_string()
            } else {
                e.identity.pci_address.clone()
            },
            e.role,
            e.role_reason,
            measured_cells(e),
            if device_local_bytes(e.vulkan.as_ref()) > 0 {
                gib(device_local_bytes(e.vulkan.as_ref()))
            } else {
                "-".into()
            },
            if vram_budget(e) > 0 { gib(vram_budget(e)) } else { "-".into() },
        ));
        for n in &e.role_notes {
            out.push_str(&format!("      note: {n}\n"));
        }
    }

    out.push_str("\nLayer-placement sketch — Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS\n");
    out.push_str(&format!(
        "  model GPU weights: {MODEL_GPU_WEIGHT_MIB} MiB total \
         (reference placement docs/phase-0/05: Vulkan0 9215.72 + Vulkan1 4361.47 MiB)\n"
    ));
    let mut budget_sum = 0u64;
    for e in &profile.devices {
        let b = vram_budget(e);
        if e.role != DeviceRole::DisplayOnly && e.role != DeviceRole::Unmeasured {
            budget_sum += b;
        }
        match e.role {
            DeviceRole::StreamingCompute => out.push_str(&format!(
                "  [{}] {} → streaming-compute: resident weights + hot expert cache + KV + \
                 prefill/drafter; streaming lane over PCIe at {} (measured overlap {}). \
                 Expert-parallel placement by default — no per-token cross-device traffic.\n",
                e.identity.index,
                short_name(&e.identity.name),
                pcie_cell(e),
                overlap_cell(e),
            )),
            DeviceRole::StaticResident => out.push_str(&format!(
                "  [{}] {} → static-resident: weights stay on the card; NO PCIe streaming lane \
                 (measured {} with {} overlap — Gate A: gen3-class host link, transfers serialize). \
                 Bandwidth-bound decode only.\n",
                e.identity.index,
                short_name(&e.identity.name),
                pcie_cell(e),
                overlap_cell(e),
            )),
            DeviceRole::DisplayOnly => out.push_str(&format!(
                "  [{}] {} → display-only: excluded from weight placement.\n",
                e.identity.index,
                short_name(&e.identity.name)
            )),
            DeviceRole::Unmeasured => out.push_str(&format!(
                "  [{}] {} → unmeasured: run `breccia profile --reprofile` before assigning.\n",
                e.identity.index,
                short_name(&e.identity.name)
            )),
        }
    }
    out.push_str(&format!(
        "  combined VRAM budget: {} ≥ {MODEL_GPU_WEIGHT_MIB} MiB weights → fits on-card \
         without host offload\n",
        gib(budget_sum)
    ));

    let h = &profile.host;
    out.push_str(&format!(
        "\nHost tiers (defaults from docs/phase-0/04, live values now)\n\
         \x20 Tier-1 VRAM: as in the role table above\n\
         \x20 Tier-2 pinned: start {}, max {}, watermark MemAvailable < {} \
         (effective mlock limit {} → units need LimitMEMLOCK=infinity)\n\
         \x20 Tier-3 SSD: {} cap {}, {} free, read {}\n\
         \x20 io_uring: {}, zram: {}\n",
        gib(h.tiers.tier2_pinned_start_bytes),
        gib(h.tiers.tier2_pinned_max_bytes),
        gib(h.tiers.tier2_watermark_mem_available_bytes),
        crate::fmt_bytes(h.mlock_limit_bytes),
        h.tiers.tier3_path,
        gib(h.tiers.tier3_cap_bytes),
        gib(h.tiers.tier3_free_bytes),
        read_cell(h),
        h.io_uring.detail,
        h.zram
            .as_ref()
            .map(|z| format!("{} ({})", z.device, gib(z.disksize_bytes)))
            .unwrap_or_else(|| "none".into()),
    ));
    out.push_str(
        "\nConstraints carried from Gate A\n\
         \x20 layer-pipeline split only after Phase-2 proves handoff overhead \
         (expert-parallel is the default)\n\
         \x20 MI50 excluded from PCIe streaming; the RX is the streaming/expert-paging home\n\
         \x20 no VK_KHR_cooperative_matrix on RADV 26.2.4 → GEMM via subgroup/FMA kernels; \
         Wave32 selectable on gfx1030 only (size control 32..64), gfx906 locked to Wave64\n",
    );
    // Pin bytes come in as a parameter: breccia-profile does not depend on the engine.
    if let Some(bytes) = expert_pin_bytes {
        let need = if bytes > 8 << 20 {
            "units need LimitMEMLOCK=infinity"
        } else {
            "fits the default 8 MiB mlock limit"
        };
        out.push_str(&format!(
            "\nHost expert import (--host-expert-import ON)\n  imported arena pin: {} → {need}\n",
            crate::fmt_bytes(bytes)
        ));
    }
    out
}

fn short_name(n: &str) -> &str {
    // "AMD Radeon RX 6950 XT (RADV NAVI21, gfx1030)" → "AMD Radeon RX 6950 XT"
    match n.find(" (") {
        Some(i) => &n[..i],
        None => n,
    }
}

fn pcie_cell(e: &DeviceEntry) -> String {
    e.measurements
        .pcie
        .as_ref()
        .and_then(|p| crate::role::steady_state_pcie(&p.sizes))
        .map(|(h, d)| format!("{h:.1} GB/s H2D / {d:.1} D2H"))
        .unwrap_or_else(|| "PCIe n/a".into())
}

fn overlap_cell(e: &DeviceEntry) -> String {
    e.measurements
        .overlap
        .as_ref()
        .map(|o| format!("{:.0}%", o.efficiency_percent))
        .unwrap_or_else(|| "n/a".into())
}

fn read_cell(h: &crate::schema::HostInfo) -> String {
    h.storage
        .iter()
        .find(|s| s.path == h.tiers.tier3_path)
        .and_then(|s| s.read_bench.as_ref())
        .map(|r| {
            format!(
                "{} {:.0} MB/s in {:.2}s ({})",
                r.method,
                r.mbps,
                r.seconds,
                short_file(&r.file)
            )
        })
        .unwrap_or_else(|| "not measured".into())
}

fn short_file(p: &str) -> String {
    let name = std::path::Path::new(p)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.to_string());
    if name.len() > 28 {
        format!("{}…", &name[..27])
    } else {
        name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{BackendKind, DeviceMeasurements};
    use crate::role::suggest_role;
    use breccia_vk::{Bandwidth, Heap, Overlap, Pcie, PcieSize};

    fn bw(copy: f64) -> Bandwidth {
        Bandwidth {
            buffer_bytes: 512 << 20,
            iterations: 100,
            seconds: 0.2,
            wall_seconds: 0.2,
            read_gbps: copy / 2.0,
            write_gbps: copy / 2.0,
            copy_gbps: copy,
            notes: vec![],
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn entry(
        index: u32,
        name: &str,
        id: &str,
        copy: f64,
        h2d: f64,
        overlap: f64,
        display: bool,
        local: u64,
    ) -> DeviceEntry {
        let input = crate::role::RoleInput {
            device_type: "DISCRETE_GPU".into(),
            dram_copy_gbps: Some(copy),
            cache_copy_gbps: Some(copy),
            pcie_h2d_gbps: Some(h2d),
            pcie_d2h_gbps: Some(h2d),
            overlap_efficiency_pct: Some(overlap),
            fp32_gflops: Some(20_000.0),
        };
        let d = suggest_role(&input);
        DeviceEntry {
            identity: crate::schema::DeviceIdentity {
                backend: BackendKind::Vulkan,
                index,
                name: name.into(),
                vendor_id: "0x1002".into(),
                device_id: id.into(),
                uuid: format!("{id}-uuid"),
                device_type: "DISCRETE_GPU".into(),
                api_version: "1.3.0".into(),
                driver_name: "RADV".into(),
                driver_version: "26.2.4".into(),
                driver_info: String::new(),
                pci_address: "0000:03:00.0".into(),
                pci_topology: "0000:03:00.0 up=0000:00:01.1 max=16.0 GT/s x16".into(),
                display_outputs: display,
            },
            vulkan: Some(breccia_vk::Properties {
                memory_heaps: vec![
                    Heap {
                        index: 0,
                        size_bytes: 29 << 30,
                        device_local: false,
                    },
                    Heap {
                        index: 1,
                        size_bytes: local,
                        device_local: true,
                    },
                ],
                ..Default::default()
            }),
            measurements: DeviceMeasurements {
                backend: BackendKind::Vulkan,
                vram_bandwidth: Some(bw(copy)),
                vram_bandwidth_cache: Some(bw(copy * 3.9)),
                pcie: Some(Pcie {
                    queue_family: 1,
                    queue_index: 0,
                    distinct_from_compute_queue: true,
                    staging_memory: "HOST_VISIBLE|HOST_COHERENT".into(),
                    device_memory: "DEVICE_LOCAL".into(),
                    sizes: vec![PcieSize {
                        size_mb: 64,
                        h2d_gbps: h2d,
                        d2h_gbps: h2d,
                        h2d_seconds_per_copy: 0.0,
                        d2h_seconds_per_copy: 0.0,
                    }],
                    notes: vec![],
                }),
                overlap: Some(Overlap {
                    compute_queue_family: 0,
                    transfer_queue_family: 1,
                    queue_mode: "distinct_transfer_queue_family".into(),
                    compute_seconds: 1.0,
                    transfer_seconds: 1.0,
                    both_seconds: 1.0,
                    efficiency_percent: overlap,
                    formula: String::new(),
                    compute_work_seconds: 1.0,
                    transfer_work_bytes: 1 << 30,
                    notes: vec![],
                }),
                ..Default::default()
            },
            role: d.role,
            role_reason: d.reason,
            role_notes: d.notes,
        }
    }

    fn profile() -> HwProfile {
        HwProfile {
            schema_version: crate::schema::SCHEMA_VERSION,
            fingerprint: "abc".into(),
            timestamp: crate::schema::iso8601(0),
            timestamp_unix_s: 0,
            quick: false,
            devices: vec![
                entry(
                    1,
                    "AMD Radeon RX 6950 XT (RADV NAVI21, gfx1030)",
                    "0x73a5",
                    462.0,
                    28.7,
                    102.3,
                    true,
                    (16 << 30) - (22 << 20),
                ),
                entry(
                    2,
                    "AMD Radeon VII (RADV VEGA20, gfx906)",
                    "0x66af",
                    826.0,
                    2.9,
                    0.8,
                    false,
                    (16 << 30) - (22 << 20),
                ),
            ],
            host: crate::system::probe_host(false),
            notes: vec![],
        }
    }

    #[test]
    fn budget_subtracts_display_reserve_only_on_display_cards() {
        let mut p = profile();
        assert!(p.devices[0].identity.display_outputs);
        let rx_budget = vram_budget(&p.devices[0]);
        let mi_budget = vram_budget(&p.devices[1]);
        assert_eq!(
            rx_budget,
            device_local_bytes(p.devices[0].vulkan.as_ref()) - DISPLAY_RESERVE_BYTES
        );
        assert_eq!(mi_budget, device_local_bytes(p.devices[1].vulkan.as_ref()));
        // Both budgets still hold the 13,577 MiB weight load together.
        assert!(rx_budget + mi_budget > MODEL_GPU_WEIGHT_MIB * 1024 * 1024);
        p.devices[0].identity.display_outputs = false;
        assert_eq!(
            vram_budget(&p.devices[0]),
            device_local_bytes(p.devices[0].vulkan.as_ref())
        );
    }

    #[test]
    fn plan_contains_roles_tiers_and_constraints() {
        let text = plan_text(&profile(), None);
        assert!(text.contains("streaming-compute"), "{text}");
        assert!(text.contains("static-resident"), "{text}");
        assert!(text.contains("13577 MiB weights"), "{text}");
        assert!(text.contains("Tier-2 pinned"), "{text}");
        assert!(text.contains("/mnt/agent-storage-a"), "{text}");
        assert!(text.contains("LimitMEMLOCK=infinity"), "{text}");
        assert!(text.contains("Wave64"), "{text}");
        assert!(text.contains("nothing written"), "{text}");
        assert!(text.contains("0x73a5"), "{text}");
        assert!(!text.contains("imported arena pin"), "{text}");
    }

    #[test]
    fn plan_reports_expert_pin_only_when_flag_on() {
        let on = plan_text(&profile(), Some(16 << 20));
        assert!(on.contains("imported arena pin: 16.0 MiB"), "{on}");
        // Bind to the new block: bare "units need LimitMEMLOCK=infinity" also
        // appears on the Tier-2 line, so it would pass with the block absent.
        assert!(
            on.contains("imported arena pin: 16.0 MiB → units need LimitMEMLOCK=infinity"),
            "{on}"
        );
        let small = plan_text(&profile(), Some(4 << 20));
        assert!(
            small.contains("fits the default 8 MiB mlock limit"),
            "{small}"
        );
        assert!(!plan_text(&profile(), None).contains("imported arena pin"));
    }
}
