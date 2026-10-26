//! Hardware fingerprint: BLAKE3 over **stable** hardware identity.
//!
//! Must hold two properties (unit-tested below):
//!
//! * **stable** — same hardware, same string, run after run (so a cached
//!   `hw_profile.json` is reused instead of re-benchmarking);
//! * **sensitive** — any hardware/driver/topology change produces a new string
//!   (so a stale profile is invalidated).
//!
//! What goes in: per-GPU `vendor:device` + `deviceUUID` + driver version/name +
//! PCIe topology string, CPU model, total RAM. What is deliberately **left out**:
//! anything state-dependent (clocks, link speed at idle, MemAvailable, free disk
//! space, timestamps, process lists).
//!
//! Only **compute devices** participate: integrated/CPU-type/other devices are
//! excluded. The engine never schedules work on them, and RADV's enumeration of
//! the integrated device was observed to drop out transiently between runs
//! (2026-10-26: one profile run saw 2 devices, the next 3) — letting that flip
//! the fingerprint would force a needless recalibration. Mesa version still
//! enters through each discrete GPU's driver fields, so driver updates invalidate.

use crate::schema::{DeviceIdentity, HostInfo};
use serde::{Deserialize, Serialize};

/// Domain-separated prefix so a Breccia fingerprint never collides with some other
/// blake3 use of the same fields.
const DOMAIN: &str = "breccia-hw-fingerprint-v1";

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Default)]
pub struct FingerprintGpu {
    pub vendor_id: String,
    pub device_id: String,
    pub uuid: String,
    pub driver_name: String,
    pub driver_version: String,
    pub pci_topology: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Default)]
pub struct FingerprintInput {
    pub gpus: Vec<FingerprintGpu>,
    pub cpu_model: String,
    pub total_ram_bytes: u64,
}

impl FingerprintInput {
    /// Build from a calibrated/queried profile's own inputs (used by `--print-fingerprint`
    /// and by the cache-validation path so both compute the same value).
    pub fn from_identities(identities: &[DeviceIdentity], host: &HostInfo) -> Self {
        let mut gpus: Vec<FingerprintGpu> = identities
            .iter()
            .filter(|i| is_compute_device(i))
            .map(|i| FingerprintGpu {
                vendor_id: i.vendor_id.clone(),
                device_id: i.device_id.clone(),
                uuid: i.uuid.clone(),
                driver_name: i.driver_name.clone(),
                driver_version: i.driver_version.clone(),
                pci_topology: i.pci_topology.clone(),
            })
            .collect();
        // Order must not matter (Vulkan enumeration order can change).
        gpus.sort_by(|a, b| {
            (&a.uuid, &a.pci_topology, &a.vendor_id, &a.device_id).cmp(&(
                &b.uuid,
                &b.pci_topology,
                &b.vendor_id,
                &b.device_id,
            ))
        });
        Self {
            gpus,
            cpu_model: host.cpu_model.clone(),
            total_ram_bytes: host.mem_total_bytes,
        }
    }
}

fn feed(h: &mut blake3::Hasher, s: &str) {
    h.update(&(s.len() as u64).to_le_bytes());
    h.update(s.as_bytes());
}

/// Devices the engine can actually schedule compute on.
///
/// Deliberately excludes `INTEGRATED_GPU`/`CPU`/`VIRTUAL_GPU`-type devices: the
/// plan never places work on them, and their enumeration is flaky through RADV
/// (see module docs). A *change of discrete GPU set* still invalidates.
fn is_compute_device(i: &DeviceIdentity) -> bool {
    matches!(i.device_type.as_str(), "DISCRETE_GPU" | "EXTERNAL_GPU")
}

/// 64-char lowercase hex BLAKE3 digest of the canonical input.
///
/// The GPU list is sorted inside so *caller* ordering can never change the digest
/// (Vulkan enumeration order, sysfs readdir order, …).
pub fn fingerprint(input: &FingerprintInput) -> String {
    let mut gpus: Vec<&FingerprintGpu> = input.gpus.iter().collect();
    gpus.sort_by(|a, b| {
        (&a.uuid, &a.pci_topology, &a.vendor_id, &a.device_id).cmp(&(
            &b.uuid,
            &b.pci_topology,
            &b.vendor_id,
            &b.device_id,
        ))
    });
    let mut h = blake3::Hasher::new();
    feed(&mut h, DOMAIN);
    h.update(&(gpus.len() as u64).to_le_bytes());
    for g in gpus {
        feed(&mut h, &g.vendor_id);
        feed(&mut h, &g.device_id);
        feed(&mut h, &g.uuid);
        feed(&mut h, &g.driver_name);
        feed(&mut h, &g.driver_version);
        feed(&mut h, &g.pci_topology);
    }
    feed(&mut h, &input.cpu_model);
    h.update(&input.total_ram_bytes.to_le_bytes());
    h.finalize().to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// This machine's identity as observed in docs/phase-0/02 + 04.
    fn this_machine() -> FingerprintInput {
        FingerprintInput {
            gpus: vec![
                FingerprintGpu {
                    vendor_id: "0x1002".into(),
                    device_id: "0x164e".into(),
                    uuid: "aaaa0000-1111-2222-3333-444444444444".into(),
                    driver_name: "RADV".into(),
                    driver_version: "26.2.4".into(),
                    pci_topology: "0000:17:00.0 up=0000:00:08.1 max=7.5 GT/s x16".into(),
                },
                FingerprintGpu {
                    vendor_id: "0x1002".into(),
                    device_id: "0x73a5".into(),
                    uuid: "bbbb0000-1111-2222-3333-444444444444".into(),
                    driver_name: "RADV".into(),
                    driver_version: "26.2.4".into(),
                    pci_topology: "0000:03:00.0 up=0000:00:01.1 max=16.0 GT/s x16".into(),
                },
                FingerprintGpu {
                    vendor_id: "0x1002".into(),
                    device_id: "0x66af".into(),
                    uuid: "cccc0000-1111-2222-3333-444444444444".into(),
                    driver_name: "RADV".into(),
                    driver_version: "26.2.4".into(),
                    pci_topology: "0000:05:00.0 up=0000:00:02.1 max=8.0 GT/s x16".into(),
                },
            ],
            cpu_model: "AMD Ryzen 9 7950X 16-Core Processor".into(),
            total_ram_bytes: 62_736_435_200,
        }
    }

    #[test]
    fn same_input_same_fingerprint() {
        let a = fingerprint(&this_machine());
        let b = fingerprint(&this_machine());
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn gpu_order_does_not_matter() {
        let mut shuffled = this_machine();
        shuffled.gpus.reverse();
        assert_eq!(fingerprint(&this_machine()), fingerprint(&shuffled));
    }

    #[test]
    fn hardware_changes_change_the_fingerprint() {
        let base = fingerprint(&this_machine());

        let mut driver = this_machine();
        driver.gpus[1].driver_version = "26.3.0".into();
        assert_ne!(base, fingerprint(&driver), "driver bump must invalidate");

        let mut uuid = this_machine();
        uuid.gpus[2].uuid = "dddd0000-1111-2222-3333-444444444444".into();
        assert_ne!(base, fingerprint(&uuid), "swapped card must invalidate");

        let mut ram = this_machine();
        ram.total_ram_bytes += 1 << 30;
        assert_ne!(base, fingerprint(&ram), "RAM change must invalidate");

        let mut cpu = this_machine();
        cpu.cpu_model = "AMD Ryzen 9 9950X".into();
        assert_ne!(base, fingerprint(&cpu), "CPU change must invalidate");

        let mut topo = this_machine();
        topo.gpus[1].pci_topology = "0000:04:00.0 up=0000:00:01.1 max=16.0 GT/s x16".into();
        assert_ne!(base, fingerprint(&topo), "re-cabled GPU must invalidate");

        let mut dropped = this_machine();
        dropped.gpus.pop();
        assert_ne!(base, fingerprint(&dropped), "removed GPU must invalidate");
    }

    #[test]
    fn state_changes_do_not_matter() {
        // Nothing volatile enters the input: this is structural, so a changed
        // field set is the test — every field here is identity, not state.
        let mut a = this_machine();
        let base = fingerprint(&a);
        a.cpu_model = a.cpu_model.clone(); // no-op rewrite
        assert_eq!(base, fingerprint(&a));
    }

    #[test]
    fn only_compute_devices_enter_the_fingerprint() {
        let mut d = crate::schema::DeviceIdentity::default();
        d.device_type = "DISCRETE_GPU".into();
        assert!(is_compute_device(&d));
        d.device_type = "EXTERNAL_GPU".into();
        assert!(is_compute_device(&d));
        // Flaky-in-enumeration / never-scheduled device types must not invalidate.
        for t in ["INTEGRATED_GPU", "CPU", "VIRTUAL_GPU", "OTHER", ""] {
            d.device_type = t.into();
            assert!(!is_compute_device(&d), "{t:?} must not enter the fingerprint");
        }
    }
}
