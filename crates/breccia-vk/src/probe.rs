//! Probe orchestration: enumerate → (optionally) benchmark → report.
//!
//! This is the reusable entry point behind `breccia profile`; the selection rules
//! and progress output match `harnesses/vk-probe` so numbers stay comparable with
//! `docs/phase-0/02-vulkan-raw-probe.md`.

use crate::bench;
use crate::ctx::{api_version_string, Dev, Gpu, Host};
use crate::report::DeviceReport;

/// Benchmark window sizing. `quick` shrinks buffers and timing targets
/// (≈ 4-6× faster, noisier) — intended for CI-speed runs.
#[derive(Debug, Clone, Default)]
pub struct ProbeOptions {
    /// Short benchmark windows (fewer iterations, smaller buffers).
    pub quick: bool,
    /// Include integrated GPUs (default: discrete GPUs only).
    pub all_devices: bool,
    /// Run only Vulkan device index N.
    pub device: Option<u32>,
    /// Bench every discovered device including the iGPU.
    pub bench_integrated: bool,
}

#[derive(Debug, Clone)]
pub struct ProbeOutcome {
    /// One entry per *benchmarked* device (identity + properties always present;
    /// measurement fields are `None` with a reason in `notes` when a sub-test failed).
    pub devices: Vec<DeviceReport>,
    /// `vkEnumerateInstanceVersion` string, e.g. "1.4.363".
    pub instance_api_version: String,
    /// Environment hazards encountered while probing (e.g. MESA_VK_DEVICE_SELECT).
    pub warnings: Vec<String>,
    /// Every device the instance enumerated (index → name), even unbenchmarked ones.
    pub discovered: Vec<(u32, String)>,
    pub quick: bool,
}

/// Select the devices to benchmark: explicit index > all_devices > discrete-only
/// (falling back to "everything that is not an integrated GPU").
fn select_devices(devices: &[Dev], opts: &ProbeOptions) -> Result<Vec<u32>, String> {
    if let Some(n) = opts.device {
        if n as usize >= devices.len() {
            return Err(format!(
                "--device {n} out of range (0..{})",
                devices.len().saturating_sub(1)
            ));
        }
        return Ok(vec![n]);
    }
    if opts.all_devices {
        return Ok((0..devices.len() as u32).collect());
    }
    let discrete: Vec<u32> = devices
        .iter()
        .filter(|d| d.device_type == ash::vk::PhysicalDeviceType::DISCRETE_GPU)
        .map(|d| d.index)
        .collect();
    if !discrete.is_empty() {
        return Ok(discrete);
    }
    let non_igpu: Vec<u32> = devices
        .iter()
        .filter(|d| d.device_type != ash::vk::PhysicalDeviceType::INTEGRATED_GPU)
        .map(|d| d.index)
        .collect();
    if !non_igpu.is_empty() {
        return Ok(non_igpu);
    }
    if opts.bench_integrated {
        return Ok((0..devices.len() as u32).collect());
    }
    Ok(Vec::new())
}

/// Run device discovery + benchmarks for the selected devices on this machine.
///
/// Clears `MESA_VK_DEVICE_SELECT` before instance creation (see the crate docs).
/// Use [`run_discovery`] when only identity/properties are needed (fingerprint path).
pub fn run_probe(opts: &ProbeOptions) -> anyhow::Result<ProbeOutcome> {
    let host = Host::new().map_err(|e| anyhow::anyhow!("breccia-vk: {e}"))?;
    let mut warnings = Vec::new();
    if host.device_select_was_set {
        warnings.push(format!(
            "MESA_VK_DEVICE_SELECT={} was set in the environment and has been cleared: it \
             would otherwise pin enumeration to a single device and hide both discrete GPUs \
             (docs/PROJECT-SETUP.md)",
            host.device_select_value.as_deref().unwrap_or("")
        ));
    }

    let devices = host
        .devices()
        .map_err(|e| anyhow::anyhow!("breccia-vk: {e}"))?;
    if devices.is_empty() {
        host.destroy();
        return Err(anyhow::anyhow!(
            "breccia-vk: no Vulkan physical devices found (is the RADV driver visible?)"
        ));
    }
    let discovered: Vec<(u32, String)> =
        devices.iter().map(|d| (d.index, d.name.clone())).collect();
    let instance_api_version = api_version_string(host.api_version);

    let selected = match select_devices(&devices, opts) {
        Ok(s) => s,
        Err(e) => {
            host.destroy();
            return Err(anyhow::anyhow!("breccia-vk: {e}"));
        }
    };
    if selected.is_empty() {
        host.destroy();
        return Err(anyhow::anyhow!(
            "breccia-vk: no devices selected for benchmarking (try all_devices)"
        ));
    }

    println!(
        "breccia-vk: instance API {instance_api_version}, {} device(s) enumerated, \
         benchmarking {:?}{}",
        devices.len(),
        selected,
        if opts.quick { " [quick mode]" } else { "" }
    );

    let mut results: Vec<DeviceReport> = Vec::new();
    for dev in &devices {
        // Identity + properties for **every** device, benchmarked or not: the
        // fingerprint (and hw_profile.json) must cover the same device set as the
        // discovery-only fast path, otherwise the cached profile never matches.
        if !selected.contains(&dev.index) {
            let mut rep = DeviceReport {
                device: bench::device_id(dev),
                properties: bench::properties(dev),
                notes: Vec::new(),
                ..Default::default()
            };
            rep.notes
                .push("not benchmarked (selection: discrete GPUs only)".to_string());
            results.push(rep);
            continue;
        }
        let gpu = match Gpu::create(&host, dev, dev.index) {
            Ok(g) => g,
            Err(e) => {
                let mut rep = DeviceReport {
                    device: bench::device_id(dev),
                    properties: bench::properties(dev),
                    notes: Vec::new(),
                    ..Default::default()
                };
                rep.error = Some(e);
                results.push(rep);
                continue;
            }
        };
        let rep = bench::run_device(dev, &gpu, opts.quick);
        results.push(rep);
        gpu.destroy();
    }
    host.destroy();

    let any_ok = results
        .iter()
        .any(|r| r.vram_bandwidth_gbps.is_some() || r.compute_gflops.is_some() || r.pcie.is_some());
    if !any_ok {
        let errs = results
            .iter()
            .filter_map(|r| r.error.clone())
            .collect::<Vec<_>>();
        return Err(anyhow::anyhow!(
            "breccia-vk: no benchmark completed successfully: {errs:?}"
        ));
    }

    Ok(ProbeOutcome {
        devices: results,
        instance_api_version,
        warnings,
        discovered,
        quick: opts.quick,
    })
}

/// Discovery only (no benchmarks) — used by the fingerprint fast path.
/// Still clears `MESA_VK_DEVICE_SELECT`.
pub fn run_discovery() -> anyhow::Result<(Vec<DeviceReport>, Vec<String>)> {
    let host = Host::new().map_err(|e| anyhow::anyhow!("breccia-vk: {e}"))?;
    let mut warnings = Vec::new();
    if host.device_select_was_set {
        warnings.push(format!(
            "MESA_VK_DEVICE_SELECT={} was set and has been cleared before enumeration",
            host.device_select_value.as_deref().unwrap_or("")
        ));
    }
    let devices = host
        .devices()
        .map_err(|e| anyhow::anyhow!("breccia-vk: {e}"))?;
    let reports = devices
        .iter()
        .map(|d| DeviceReport {
            device: bench::device_id(d),
            properties: bench::properties(d),
            ..Default::default()
        })
        .collect();
    host.destroy();
    Ok((reports, warnings))
}
