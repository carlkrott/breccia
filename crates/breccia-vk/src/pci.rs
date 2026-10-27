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

/// Normalise a user-supplied PCI address to the sysfs / `VK_EXT_pci_bus_info`
/// spelling `dddd:bb:dd.f`, so `3:0.0`, `03:00.0` and `0000:03:00.0` are one key.
///
/// Vulkan ordinals are not stable on this host (RADV intermittently omits the iGPU
/// and reorders enumeration), so PCI identity is the only safe device selector.
pub fn normalize_address(address: &str) -> String {
    fn pad_dev(s: &str) -> String {
        match s.split_once('.') {
            Some((d, f)) => format!("{d:0>2}.{f}"),
            None => format!("{s:0>2}"),
        }
    }
    let t = address.trim().to_ascii_lowercase();
    if t.is_empty() {
        return t;
    }
    match t.split(':').collect::<Vec<_>>().as_slice() {
        [domain, bus, dev] => format!("{domain:0>4}:{bus:0>2}:{}", pad_dev(dev)),
        [bus, dev] => format!("0000:{bus:0>2}:{}", pad_dev(dev)),
        _ => t,
    }
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

// ---------------------------------------------------------------------------
// The driver's own VRAM counters (the `VK_EXT_memory_budget` fallback)
// ---------------------------------------------------------------------------

/// amdgpu's `mem_info_vram_total` / `mem_info_vram_used` for `address`, in bytes.
///
/// This is the *driver's* figure, not a guess: it is the same number
/// `amdgpu_top`/`radeontop` show, and it is system-wide (every process). Use it
/// only when `VK_EXT_memory_budget` is absent, and say so in the report — the
/// arena's reserve then has to cover other processes' current use as well.
pub fn vram_used(address: &str) -> Option<(u64, u64)> {
    let dev = format!("{SYS_PCI}/{}", normalize_address(address));
    let total: u64 = read_trimmed(format!("{dev}/mem_info_vram_total"))?
        .parse()
        .ok()?;
    let used: u64 = read_trimmed(format!("{dev}/mem_info_vram_used"))?
        .parse()
        .ok()?;
    Some((total, used))
}

/// Does `address` currently drive a display?
///
/// Read from sysfs: any `card*-<connector>/status` **whose DRM card belongs to
/// `address`** reading `connected`. This is the fact the expert arena's *display
/// reserve* keys on — hard-coding a PCI address in the reserve rule is exactly the
/// host-specific assumption that made the arena a block list in the first place
/// (and it is already stale on this box: the connected outputs are on
/// `0000:03:00.0`, while the plan text in `AGENTS.md` says the MI50).
///
/// The connector directories live under `/sys/class/drm/`, **not** under the PCI
/// device's own `drm/` node (that one holds only `cardN`/`renderD*`), so the match
/// is made by canonicalising — the kernel's connector path goes through the PCI
/// device, and that is the identity we want.
///
/// `None` when sysfs cannot answer (no DRM nodes at all): the caller must then
/// treat the reserve conservatively and say it could not verify.
pub fn drives_a_display(address: &str) -> Option<bool> {
    let want = fs::canonicalize(format!("{SYS_PCI}/{}", normalize_address(address))).ok()?;
    let entries = fs::read_dir("/sys/class/drm").ok()?;
    let mut saw_connector = false;
    let mut saw_card = false;
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if !name.starts_with("card") {
            continue;
        }
        if !name.contains('-') {
            saw_card = true;
            continue; // the `cardN` node itself
        }
        if name.contains("Writeback") {
            continue; // not an output
        }
        let Ok(real) = fs::canonicalize(e.path()) else {
            continue;
        };
        if !real.starts_with(&want) {
            continue; // another card's connector
        }
        saw_connector = true;
        let status = read_trimmed(format!("{}/status", e.path().display())).unwrap_or_default();
        if status == "connected" {
            return Some(true);
        }
    }
    if saw_card {
        // A card with no connector *subdirectory* at all is an unknown, not a
        // "headless" answer: some kernels only expose connectors for cards that
        // have a display controller bound.
        if saw_connector {
            Some(false)
        } else {
            None
        }
    } else {
        None
    }
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
    fn address_normalisation_is_stable() {
        for (input, want) in [
            ("0000:03:00.0", "0000:03:00.0"),
            ("03:00.0", "0000:03:00.0"),
            ("3:0.0", "0000:03:00.0"),
            (" 0000:0E:00.0 ", "0000:0e:00.0"),
            ("0000:05:00.0", "0000:05:00.0"),
        ] {
            assert_eq!(normalize_address(input), want, "input {input:?}");
        }
        // round-trips what address_from_bus_info produces
        assert_eq!(
            normalize_address(&address_from_bus_info(0x41, 0x1f, 3)),
            "0000:41:1f.3"
        );
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

    /// The arena's display reserve is keyed on this: every discrete card must be
    /// *answerable* (Some), because `unknown` falls back to the display reserve.
    /// Which one is connected is a live fact (and it is not the card the plan text
    /// assumes), so it is printed rather than asserted.
    #[test]
    fn the_display_fact_is_answerable_per_card() {
        for (vid, did) in [(0x1002u32, 0x73a5u32), (0x1002, 0x66af)] {
            let Some(addr) = find_address_by_ids(vid, did) else {
                continue; // the artifact is host-specific; skip what is absent
            };
            let d = drives_a_display(&addr);
            println!("{addr} drives_a_display={d:?} vram={:?}", vram_used(&addr));
            assert!(d.is_some(), "{addr}: no connector information at all");
            assert!(
                vram_used(&addr).is_some(),
                "{addr}: no amdgpu vram counters"
            );
        }
    }
}
