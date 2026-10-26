//! One Vulkan device (instance + logical device) for the kernels.
//!
//! Everything here is `breccia_vk::ctx`: instance/device-queue selection, the
//! `MESA_VK_DEVICE_SELECT` trap handling, buffer allocation and timestamp
//! timing are reused as-is (docs/phase-0 `02`, docs/phase-2 `01`) — this module
//! only picks a device, keeps its identity for the report, and owns the
//! teardown order (device before instance).

use anyhow::{anyhow, Context, Result};
use breccia_vk::ctx::{self, Dev, Gpu, Host};

/// Identity of the opened device, for reports.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DeviceInfo {
    pub index: u32,
    pub name: String,
    pub driver: String,
    pub pci_address: String,
    pub device_id: String,
    pub subgroup_size: u32,
    pub timestamp_period_ns: f64,
}

/// An open Vulkan device.
pub struct Device {
    inner: Option<(Host, Gpu)>,
    pub info: DeviceInfo,
    /// Wall time spent in instance + device creation.
    pub open_seconds: f64,
}

impl Device {
    /// Open the Vulkan device selected by `selector`.
    ///
    /// * `None` — first discrete GPU (the RX 6950 XT on this host).
    /// * `Some("0000:03:00.0")` — by PCI address (stable; Vulkan ordinals are
    ///   not stable here, docs/phase-1/profile-report.md).
    /// * `Some("1")` — by runtime enumeration index.
    pub fn open(selector: Option<&str>) -> Result<Device> {
        let t0 = std::time::Instant::now();
        let host = Host::new().map_err(|e| anyhow!("{e}"))?;
        let devs: Vec<Dev> = host.devices().map_err(|e| anyhow!("{e}"))?;
        if devs.is_empty() {
            host.destroy();
            return Err(anyhow!(
                "no Vulkan device enumerated — is MESA_VK_DEVICE_SELECT hiding the GPUs?"
            ));
        }
        let dev = resolve(&devs, selector)?;
        let info = DeviceInfo {
            index: dev.index,
            name: dev.name.clone(),
            driver: format!(
                "{} {}",
                dev.driver_name,
                ctx::driver_version_string(dev.driver_version)
            ),
            pci_address: dev.pci_address.clone(),
            device_id: format!("{:04x}:{:04x}", dev.vendor_id, dev.device_id),
            subgroup_size: dev.subgroup_size,
            timestamp_period_ns: f64::from(dev.timestamp_period_ns),
        };
        let gpu = match Gpu::create(&host, dev, dev.index) {
            Ok(g) => g,
            Err(e) => {
                host.destroy();
                return Err(anyhow!("{e}"));
            }
        };
        Ok(Device {
            inner: Some((host, gpu)),
            info,
            open_seconds: t0.elapsed().as_secs_f64(),
        })
    }

    pub fn gpu(&self) -> &Gpu {
        &self.inner.as_ref().expect("device open").1
    }
}

fn resolve<'a>(devs: &'a [Dev], selector: Option<&str>) -> Result<&'a Dev> {
    match selector {
        None => devs
            .iter()
            .find(|d| d.device_type == ash::vk::PhysicalDeviceType::DISCRETE_GPU)
            .or_else(|| devs.first())
            .ok_or_else(|| anyhow!("no device")),
        Some(spec) => {
            let t = spec.trim();
            if t.is_empty() {
                return Err(anyhow!("empty device selector"));
            }
            if t.contains(':') {
                let want = breccia_vk::pci::normalize_address(t);
                devs.iter().find(|d| d.pci_address == want).ok_or_else(|| {
                    anyhow!(
                        "no Vulkan device at PCI {want}; enumerated: {}",
                        devs.iter()
                            .map(|d| format!("{} [{}]", d.name, d.pci_address))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                })
            } else {
                let idx: u32 = t.parse().with_context(|| {
                    format!("--device {t:?} is neither an index nor a PCI address")
                })?;
                devs.iter()
                    .find(|d| d.index == idx)
                    .ok_or_else(|| anyhow!("no Vulkan device with index {idx}"))
            }
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        if let Some((host, gpu)) = self.inner.take() {
            gpu.destroy();
            host.destroy();
        }
    }
}
