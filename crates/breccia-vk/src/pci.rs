//! Stable PCIe topology strings for the hardware fingerprint.
//!
//! The fingerprint must change when hardware changes but **not** when the machine
//! merely re-trains its link (idle `current_link_speed` drops to 2.5 GT/s), so the
//! topology string uses only static facts: the device address, its upstream bridge,
//! and the negotiated *maximum* link capability (`max_link_*`, from the card's
//! PCIe capabilities — not the current state).

use std::fs;

const SYS_PCI: &str = "/sys/bus/pci/devices";

/// `VK_EXT_pci_bus_info` fields → canonical `dddd:bb:dd.f` address.
pub fn address_from_bus_info(bus: u32, device: u32, function: u32) -> String {
    format!("0000:{bus:02x}:{device:02x}.{function}")
}

/// Fallback discovery: scan sysfs for a device with this vendor:device id pair.
/// Returns the lexicographically first match (deterministic across runs).
pub fn find_address_by_ids(vendor_id: u32, device_id: u32) -> Option<String> {
    let want_v = format!("0x{vendor_id:04x}");
    let want_d = format!("0x{device_id:04x}");
    let mut matches = Vec::new();
    let entries = fs::read_dir(SYS_PCI).ok()?;
    for e in entries.flatten() {
        let v = fs::read_to_string(e.path().join("vendor")).unwrap_or_default();
        let d = fs::read_to_string(e.path().join("device")).unwrap_or_default();
        if v.trim() == want_v && d.trim() == want_d {
            matches.push(e.file_name().to_string_lossy().into_owned());
        }
    }
    matches.sort();
    matches.into_iter().next()
}

/// Gather the static topology facts for `address` from sysfs.
/// Unreadable parts become `"?"`; a missing address returns `"?"`.
pub fn topology_string(address: &str) -> String {
    if address.is_empty() {
        return "?".to_string();
    }
    let dev_dir = format!("{SYS_PCI}/{address}");
    let bridge = fs::canonicalize(&dev_dir)
        .ok()
        .and_then(|p| p.parent().map(|q| q.to_path_buf()))
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .filter(|n| n.contains(':') && !n.starts_with("pci"))
        .unwrap_or_else(|| "?".to_string());
    let speed = read_trimmed(format!("{dev_dir}/max_link_speed")).unwrap_or_else(|| "?".into());
    let width = read_trimmed(format!("{dev_dir}/max_link_width")).unwrap_or_else(|| "?".into());
    topology_from_parts(address, &bridge, &speed, &width)
}

/// Pure formatter (unit-tested): `0000:03:00.0 up=0000:00:01.1 max=16.0 GT/s x16`.
pub fn topology_from_parts(
    address: &str,
    bridge: &str,
    max_speed: &str,
    max_width: &str,
) -> String {
    format!("{address} up={bridge} max={max_speed} x{max_width}")
}

fn read_trimmed(path: String) -> Option<String> {
    fs::read_to_string(path).ok().map(|s| s.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bus_info_address_format() {
        assert_eq!(address_from_bus_info(3, 0, 0), "0000:03:00.0");
        assert_eq!(address_from_bus_info(0x41, 0x1f, 3), "0000:41:1f.3");
    }

    #[test]
    fn topology_is_stable_and_readable() {
        let s = topology_from_parts("0000:03:00.0", "0000:00:01.1", "16.0 GT/s", "16");
        assert_eq!(s, "0000:03:00.0 up=0000:00:01.1 max=16.0 GT/s x16");
        // Unknown parts must not panic and must round into the same shape.
        let u = topology_from_parts("0000:05:00.0", "?", "?", "?");
        assert!(u.starts_with("0000:05:00.0 up=?"));
    }

    #[test]
    fn topology_on_this_machine_is_resolved() {
        // This host has at least the iGPU + two discrete cards in sysfs.
        let addr = find_address_by_ids(0x1002, 0x73a5).expect("RX 6950 XT in sysfs");
        let topo = topology_string(&addr);
        assert!(topo.contains(&addr), "{topo}");
        assert!(
            !topo.contains("max=? x?"),
            "topology facts unreadable: {topo}"
        );
    }
}
