//! Backend seam: one trait + backend-neutral measurement types, so a future HIP
//! backend can be ingested by `breccia-profile` without touching the schema.
//!
//! Phase 1 implements [`VulkanBackend`] only. The HIP side later is a second
//! `BackendProbe` impl returning `BackendKind::Hip` devices with the same
//! measurement shapes (VRAM bandwidth, FMA GFLOPS, PCIe, overlap) — the role
//! heuristic and `hw_profile.json` stay unchanged.

use crate::schema::DeviceIdentity;
use anyhow::Result;
use breccia_vk::{ProbeOptions, Properties};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    #[default]
    Vulkan,
    Hip,
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendKind::Vulkan => f.write_str("vulkan"),
            BackendKind::Hip => f.write_str("hip"),
        }
    }
}

/// Backend-neutral measurement payload stored in `hw_profile.json`.
///
/// The measurement structs are the (ash-free) serde types from `breccia-vk`;
/// a HIP implementation maps its numbers into the same shapes.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Default)]
pub struct DeviceMeasurements {
    pub backend: BackendKind,
    pub vram_bandwidth: Option<breccia_vk::Bandwidth>,
    pub vram_bandwidth_cache: Option<breccia_vk::Bandwidth>,
    pub compute: Option<breccia_vk::Compute>,
    pub pcie: Option<breccia_vk::Pcie>,
    pub overlap: Option<breccia_vk::Overlap>,
    pub notes: Vec<String>,
    pub error: Option<String>,
}

/// What one backend reports per device: who it is, its discovery detail (backend
/// specific, hence optional), and what was measured.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendDevice {
    pub identity: DeviceIdentity,
    pub vulkan: Option<Properties>,
    pub measurements: DeviceMeasurements,
}

/// The seam. `discover` is the cheap path (identity only, used for the
/// fingerprint / cache validation); `probe` is the calibration path and also
/// returns backend-level warnings (environment traps, etc.).
pub trait BackendProbe {
    fn backend(&self) -> BackendKind;
    fn discover(&self) -> Result<Vec<DeviceIdentity>>;
    fn probe(&self, opts: &ProbeOptions) -> Result<BackendRun>;
}

#[derive(Debug, Clone, Default)]
pub struct BackendRun {
    pub devices: Vec<BackendDevice>,
    pub warnings: Vec<String>,
}

/// Phase-1 implementation: `breccia-vk` (RADV/ash, port of `harnesses/vk-probe`).
pub struct VulkanBackend;

fn topology_of(address: &str) -> String {
    if address.is_empty() {
        String::new()
    } else {
        breccia_vk::pci::topology_string(address)
    }
}

impl BackendProbe for VulkanBackend {
    fn backend(&self) -> BackendKind {
        BackendKind::Vulkan
    }

    fn discover(&self) -> Result<Vec<DeviceIdentity>> {
        let (reports, _warnings) = breccia_vk::run_discovery()?;
        Ok(reports
            .iter()
            .map(|r| {
                let mut id = identity_from(&r.device);
                id.display_outputs = crate::system::has_display_outputs(&id.pci_address);
                id
            })
            .collect())
    }

    fn probe(&self, opts: &ProbeOptions) -> Result<BackendRun> {
        let outcome = breccia_vk::run_probe(opts)?;
        let warnings = outcome.warnings.clone();
        let devices = outcome
            .devices
            .iter()
            .map(|r| {
                let mut identity = identity_from(&r.device);
                identity.display_outputs =
                    crate::system::has_display_outputs(&identity.pci_address);
                BackendDevice {
                    identity,
                    vulkan: Some(r.properties.clone()),
                    measurements: DeviceMeasurements {
                        backend: BackendKind::Vulkan,
                        vram_bandwidth: r.vram_bandwidth_gbps.clone(),
                        vram_bandwidth_cache: r.vram_bandwidth_cache_gbps.clone(),
                        compute: r.compute_gflops.clone(),
                        pcie: r.pcie.clone(),
                        overlap: r.overlap.clone(),
                        notes: r.notes.clone(),
                        error: r.error.clone(),
                    },
                }
            })
            .collect();
        Ok(BackendRun { devices, warnings })
    }
}

fn identity_from(d: &breccia_vk::DeviceId) -> DeviceIdentity {
    DeviceIdentity {
        backend: BackendKind::Vulkan,
        index: d.index,
        name: d.name.clone(),
        vendor_id: d.vendor_id.clone(),
        device_id: d.device_id.clone(),
        uuid: d.uuid.clone(),
        device_type: d.device_type.clone(),
        api_version: d.api_version.clone(),
        driver_name: d.driver_name.clone(),
        driver_version: d.driver_version.clone(),
        driver_info: d.driver_info.clone(),
        pci_address: d.pci_address.clone(),
        pci_topology: topology_of(&d.pci_address),
        display_outputs: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_kind_serde_is_lowercase() {
        assert_eq!(
            serde_json::to_string(&BackendKind::Vulkan).unwrap(),
            "\"vulkan\""
        );
        assert_eq!(serde_json::to_string(&BackendKind::Hip).unwrap(), "\"hip\"");
        let back: BackendKind = serde_json::from_str("\"hip\"").unwrap();
        assert_eq!(back, BackendKind::Hip);
    }

    #[test]
    fn discover_lists_the_discrete_gpus_on_this_host() {
        // Cheap path: identity only, no benchmarks (and MESA trap cleared inside).
        let ids = VulkanBackend.discover().expect("discovery");
        assert!(ids.len() >= 3, "expected 3 devices, got {}", ids.len());
        assert!(ids.iter().any(|i| i.device_id == "0x73a5"), "RX 6950 XT");
        assert!(ids.iter().any(|i| i.device_id == "0x66af"), "Radeon VII");
        for id in &ids {
            assert!(!id.uuid.is_empty(), "{} has no uuid", id.name);
            assert!(!id.pci_topology.is_empty(), "{} has no topology", id.name);
        }
        // The RX on this machine hangs off a CPU root port with x16 lanes.
        let rx = ids.iter().find(|i| i.device_id == "0x73a5").unwrap();
        assert!(rx.pci_topology.contains("x16"), "{}", rx.pci_topology);
    }

    #[test]
    fn measurements_default_to_empty_not_wrong() {
        let m = DeviceMeasurements::default();
        assert_eq!(m.backend, BackendKind::Vulkan);
        assert!(m.vram_bandwidth.is_none() && m.overlap.is_none());
        let s = serde_json::to_string(&m).unwrap();
        assert!(s.contains("vram_bandwidth"));
    }
}
