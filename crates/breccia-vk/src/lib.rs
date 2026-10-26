//! `breccia-vk` — Vulkan device discovery, raw micro-benchmarks, and the Phase-2
//! asynchronous transfer layer.
//!
//! This crate is the reusable port of `harnesses/vk-probe/` (the Phase-0.2 probe
//! that produced `docs/phase-0/02-vulkan-raw-probe.md`). It provides:
//!
//! * [`run_probe`] — enumerate devices, then (optionally) measure VRAM bandwidth,
//!   FMA compute (fp32/fp16), PCIe H2D/D2H and compute/transfer overlap;
//! * [`transfer::TransferRing`] — the engine-facing DMA ring: [`transfer`] is the
//!   timeline-semaphore ring / pinned staging / compute-transfer overlap primitive
//!   that M3 (expert streaming) and M5 (KV paging) drive, and [`transfer_bench`] is
//!   the M1b benchmark that proves it (`breccia bench transfer`);
//! * [`ctx::Host`] — instance/device plumbing (headless, no surface);
//! * [`report`] — serde types for the measurement results;
//! * [`pci`] — stable PCIe topology strings for the hardware fingerprint.
//!
//! # Environment trap
//!
//! This machine exports `MESA_VK_DEVICE_SELECT=1002:164e!` (desktop session pins
//! the iGPU), which hides **both discrete GPUs** from every Vulkan process.
//! [`ctx::Host::new`] unsets the variable before instance creation and records a
//! warning in [`probe::ProbeOutcome::warnings`] (also asserted by a unit test).

pub mod bench;
pub mod ctx;
pub mod pci;
pub mod probe;
pub mod report;
pub mod transfer;
pub mod transfer_bench;

pub use ctx::{api_version_string, driver_version_string, Dev, Host};
pub use probe::{run_discovery, run_probe, ProbeOptions, ProbeOutcome};
pub use report::*;
pub use transfer::{RingConfig, RingStats, TransferRing};
pub use transfer_bench::{
    run_transfer_bench, Churn, DeviceSelector, DeviceTransfer, RingSizeStat, RingThroughput,
    TransferBenchOptions, TransferCaps, TransferEnvironment, TransferQueue, TransferReport,
    TRANSFER_SCHEMA_VERSION,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Run-for-real enumeration test (this machine): without clearing
    /// `MESA_VK_DEVICE_SELECT` only the Raphael iGPU would show up.
    #[test]
    fn enumerates_three_devices_on_this_host() {
        let host = match Host::new() {
            Ok(h) => h,
            Err(e) => panic!("libvulkan must load on this machine: {e}"),
        };
        let devs = host.devices().expect("physical device enumeration");
        assert!(
            devs.len() >= 3,
            "expected 3 Vulkan devices (iGPU + RX 6950 XT + Radeon VII), got {}: {:?}",
            devs.len(),
            devs.iter().map(|d| d.name.as_str()).collect::<Vec<_>>()
        );
        assert!(
            devs.iter()
                .any(|d| d.vendor_id == 0x1002 && d.device_id == 0x73a5),
            "RX 6950 XT (1002:73a5) missing — MESA_VK_DEVICE_SELECT not cleared?"
        );
        assert!(
            devs.iter()
                .any(|d| d.vendor_id == 0x1002 && d.device_id == 0x66af),
            "Radeon VII (1002:66af) missing — MESA_VK_DEVICE_SELECT not cleared?"
        );
        for d in &devs {
            assert!(!d.uuid.is_empty(), "device {} has no deviceUUID", d.name);
            assert!(
                d.exts.iter().any(|e| e == "VK_KHR_timeline_semaphore"),
                "{} lacks VK_KHR_timeline_semaphore",
                d.name
            );
        }
        host.destroy();
    }

    /// The environment trap itself: a set variable is cleared and reported.
    #[test]
    fn device_select_env_is_cleared_and_reported() {
        // Safe under edition 2021; this test deliberately sets the trap first.
        std::env::set_var("MESA_VK_DEVICE_SELECT", "1002:164e!");
        let host = ctx::Host::new().expect("libvulkan loads");
        assert!(
            host.device_select_was_set,
            "Host::new must report the MESA_VK_DEVICE_SELECT trap"
        );
        assert!(
            std::env::var("MESA_VK_DEVICE_SELECT").is_err(),
            "variable must be unset before instance creation"
        );
        host.destroy();
    }

    /// The report types must round-trip through JSON (they feed hw_profile.json).
    #[test]
    fn report_json_round_trip() {
        let rep = report::DeviceReport {
            device: report::DeviceId {
                index: 1,
                name: "AMD Radeon RX 6950 XT (RADV NAVI21, gfx1030)".into(),
                vendor_id: "0x1002".into(),
                device_id: "0x73a5".into(),
                uuid: "b3f0..".into(),
                pci_address: "0000:03:00.0".into(),
                device_type: "DISCRETE_GPU".into(),
                api_version: "1.3.0".into(),
                driver_version: "26.2.4".into(),
                driver_name: "RADV".into(),
                driver_info: "Mesa 26.2.4".into(),
            },
            ..Default::default()
        };
        let s = serde_json::to_string_pretty(&rep).expect("serialize");
        let back: report::DeviceReport = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(rep, back);
    }
}
