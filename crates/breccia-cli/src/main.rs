//! `breccia` CLI entry (`breccia profile|serve|bench`).
//!
//! Phase 1 wires `breccia profile` (hardware discovery + calibration →
//! `hw_profile.json`) on top of `breccia-vk` + `breccia-profile`. Phase 2 adds
//! `breccia bench transfer` (the M1b asynchronous transfer harness: DMA ring
//! throughput, timeline-semaphore usage, host-memory capability, and the M3-shaped
//! expert-slot churn that decides whether DMA hides under compute).
//!
//! ```text
//! breccia profile [--reprofile] [--quick] [--json PATH] [--dry-run-plan] [--print-fingerprint]
//! breccia bench transfer [--device <profile-index|pci>]... [--json PATH] [--quick]
//! breccia bench gemv [--type T]... [--rows N --cols N] [--device <idx|pci>]... [--json PATH] [--quick] [--real]
//! ```
//!
//! Behaviour: with a cached profile whose fingerprint matches (and whose
//! benchmark window is at least as long as requested) `profile` prints the cached
//! summary and exits 0 without touching the GPU; otherwise it calibrates and
//! writes. `--dry-run-plan` prints the role table + IQ3_XXS placement sketch and
//! writes nothing.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use breccia_vk::DeviceSelector;

#[derive(Parser)]
#[command(
    name = "breccia",
    version,
    about = "Breccia — hybrid HIP/Vulkan inference engine",
    propagate_version = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Hardware discovery + calibration → hw_profile.json (M1a)
    Profile {
        /// Ignore a matching cached profile and re-run the calibration
        #[arg(long)]
        reprofile: bool,
        /// Short benchmark windows (CI-speed, noisier numbers)
        #[arg(long)]
        quick: bool,
        /// Write the profile here (default: $XDG_CONFIG_HOME/breccia/hw_profile.json);
        /// also the path the cached fast path is read from
        #[arg(long, value_name = "PATH")]
        json: Option<PathBuf>,
        /// Print the device role table + layer-placement sketch, write nothing
        #[arg(long)]
        dry_run_plan: bool,
        /// Print the hardware fingerprint (BLAKE3) and exit
        #[arg(long)]
        print_fingerprint: bool,
    },
    /// Phase 2: OpenAI-compatible agent server (parsing surface present now)
    Serve {
        /// Resolve and print the backend/device/placement plan, launch nothing
        #[arg(long)]
        dry_run_plan: bool,
        /// llama.cpp/HIP-compatible arguments — the production
        /// qwen-dual-serve.sh invocation works verbatim (ROCmN aliases and all)
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Benchmark harnesses
    Bench {
        #[command(subcommand)]
        kind: Option<BenchKind>,
    },
}

#[derive(Subcommand)]
enum BenchKind {
    /// M1b: async transfer ring — H2D throughput, overlap, host-memory caps
    Transfer {
        /// Device by cached-profile index or PCI address (repeatable);
        /// default = every discrete GPU
        #[arg(long = "device", value_name = "PROFILE_INDEX|PCI")]
        device: Vec<String>,
        /// Write the full JSON report here
        #[arg(long, value_name = "PATH")]
        json: Option<PathBuf>,
        /// Short benchmark windows (CI-speed, noisier numbers)
        #[arg(long)]
        quick: bool,
    },
    /// Phase 3: fused dequant+GEMV decode kernels — GB/s vs bytes read
    Gemv {
        /// Quant type, repeatable (default: every implemented type)
        #[arg(long = "type", value_name = "IQ3_S")]
        types: Vec<String>,
        /// Output rows (default: the model's decode shapes)
        #[arg(long)]
        rows: Option<usize>,
        /// Row length in elements (default: paired with --rows)
        #[arg(long)]
        cols: Option<usize>,
        /// Device by cached-profile index or PCI address (repeatable);
        /// default = every discrete GPU
        #[arg(long = "device", value_name = "PROFILE_INDEX|PCI")]
        device: Vec<String>,
        /// Write the full JSON report here
        #[arg(long, value_name = "PATH")]
        json: Option<PathBuf>,
        /// Short benchmark windows (CI-speed, noisier numbers)
        #[arg(long)]
        quick: bool,
        /// Run the kernel on a real expert tensor from the GGUF (mmap, no copy)
        #[arg(long)]
        real: bool,
        /// GGUF glob for --real (default: the IQ3_XXS shards)
        #[arg(long, value_name = "GLOB")]
        gguf: Option<String>,
        /// Block (layer) index for --real
        #[arg(long, default_value_t = 0)]
        layer: usize,
        /// Expert role for --real: gate | up | down
        #[arg(long, default_value = "gate")]
        role: String,
        /// Restrict --real to one expert (adds a CPU conformance check)
        #[arg(long, value_name = "E")]
        expert: Option<usize>,
        /// Force a split-K factor (default: automatic — see docs/phase-3 §4.2)
        #[arg(long, value_name = "N")]
        splits: Option<u32>,
        /// Real mode on a named tensor instead of an expert role
        /// (e.g. output.weight for Q5_K)
        #[arg(long, value_name = "NAME")]
        tensor: Option<String>,
    },
}

fn main() {
    let cli = Cli::parse();
    if let Err(e) = run(cli) {
        eprintln!("breccia: {e:#}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Profile {
            reprofile,
            quick,
            json,
            dry_run_plan,
            print_fingerprint,
        } => profile(ProfileArgs {
            reprofile,
            quick,
            json,
            dry_run_plan,
            print_fingerprint,
        }),
        Command::Serve { dry_run_plan, args } => serve(dry_run_plan, args),
        Command::Bench { kind: None } => {
            println!(
                "bench: available harnesses\n  \
                 breccia bench transfer [--device <idx|pci>]... [--json PATH] [--quick]\n  \
                 breccia bench gemv [--type IQ3_S]... [--rows N --cols N] [--device <idx|pci>]... \\\n                 \
                 [--json PATH] [--quick] [--real [--layer N] [--role gate|up|down] [--expert E]]"
            );
            Ok(())
        }
        Command::Bench {
            kind:
                Some(BenchKind::Transfer {
                    device,
                    json,
                    quick,
                }),
        } => bench_transfer(device, json, quick),
        Command::Bench {
            kind:
                Some(BenchKind::Gemv {
                    types,
                    rows,
                    cols,
                    device,
                    json,
                    quick,
                    real,
                    gguf,
                    layer,
                    role,
                    expert,
                    splits,
                    tensor,
                }),
        } => bench_gemv(GemvArgs {
            types,
            rows,
            cols,
            device,
            json,
            quick,
            real,
            gguf,
            layer,
            role,
            expert,
            splits,
            tensor,
        }),
    }
}

struct ProfileArgs {
    reprofile: bool,
    quick: bool,
    json: Option<PathBuf>,
    dry_run_plan: bool,
    print_fingerprint: bool,
}

fn profile(a: ProfileArgs) -> Result<()> {
    let path = a
        .json
        .clone()
        .unwrap_or_else(breccia_profile::default_profile_path);

    // Cheap identity pass first: decides the cached fast path (and answers
    // --print-fingerprint) without running any benchmark.
    let current = breccia_profile::current_fingerprint()
        .context("hardware discovery failed (is libvulkan/RADV visible?)")?;

    if a.print_fingerprint {
        println!("{}", current.fingerprint);
        return Ok(());
    }

    // Cached fast path.
    if !a.reprofile {
        if let Some(cached) = breccia_profile::load_profile(&path)? {
            if breccia_profile::cache_usable(&cached, &current.fingerprint, a.quick) {
                print!("{}", breccia_profile::summary_text(&cached));
                if a.dry_run_plan {
                    print!("{}", breccia_profile::plan::plan_text(&cached));
                }
                println!(
                    "cached profile reused (fingerprint match): {}",
                    path.display()
                );
                return Ok(());
            }
        }
    }

    // Calibrate.
    println!(
        "calibrating hardware profile ({} mode)…",
        if a.quick { "quick" } else { "full" }
    );
    let profile =
        breccia_profile::calibrate(a.quick).context("calibration failed (Vulkan benchmarks)")?;

    if a.dry_run_plan {
        print!("{}", breccia_profile::plan::plan_text(&profile));
        println!("dry run: nothing written (would write {})", path.display());
        return Ok(());
    }

    breccia_profile::save_profile(&profile, &path)?;
    print!("{}", breccia_profile::summary_text(&profile));
    println!("wrote {}", path.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// serve — llama.cpp/HIP-compatible launch surface (Phase 1: plan only)
// ---------------------------------------------------------------------------

/// Map `GPU[n] -> PCI bus` from rocm-smi, in HIP ordinal order.
///
/// The ordinal orders differ between backends on this machine (Vulkan's device 0
/// is the iGPU while HIP's device 0 is the RX 6950 XT), so a `ROCmN` alias must
/// never be resolved by position alone. PCI identity is the stable key.
fn hip_pci_buses() -> Option<Vec<String>> {
    let out = std::process::Command::new("rocm-smi")
        .arg("--showbus")
        .output()
        .ok()?;
    if !out.status.success() && out.stdout.is_empty() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut buses: Vec<(usize, String)> = Vec::new();
    for line in text.lines() {
        let Some((lhs, rhs)) = line.split_once(':') else {
            continue;
        };
        let Some(idx_txt) = lhs.trim().strip_prefix("GPU[") else {
            continue;
        };
        let Ok(idx) = idx_txt.trim_end_matches(']').parse::<usize>() else {
            continue;
        };
        let Some(bus) = rhs.split("PCI Bus:").nth(1) else {
            continue;
        };
        let mut bus = bus.trim().to_ascii_lowercase();
        // normalise "0000:0e:00.0" / "0000:0E:00.0" to the sysfs spelling
        if let Some(rest) = bus.strip_prefix("0000:") {
            bus = format!("0000:{rest}");
        }
        buses.push((idx, bus));
    }
    if buses.is_empty() {
        return None;
    }
    buses.sort_by_key(|(i, _)| *i);
    Some(buses.into_iter().map(|(_, b)| b).collect())
}

// ---------------------------------------------------------------------------
// bench transfer — M1b asynchronous transfer harness
// ---------------------------------------------------------------------------

fn bench_transfer(specs: Vec<String>, json: Option<PathBuf>, quick: bool) -> Result<()> {
    let profile = breccia_profile::load_profile(&breccia_profile::default_profile_path())
        .ok()
        .flatten();
    let mut devices = Vec::new();
    for spec in &specs {
        devices.push(selector_for(spec, profile.as_ref())?);
    }

    let report =
        breccia_vk::run_transfer_bench(&breccia_vk::TransferBenchOptions { quick, devices })
            .context("transfer benchmark failed")?;

    if let Some(path) = &json {
        let text = serde_json::to_string_pretty(&report).context("serializing transfer report")?;
        if let Some(dir) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        std::fs::write(path, text + "\n").with_context(|| format!("writing {}", path.display()))?;
    }

    print_transfer_summary(&report);
    if let Some(path) = &json {
        println!("\nwrote {}", path.display());
    }
    Ok(())
}

/// Turn one `--device` argument into a selector that survives enumeration-order
/// changes: a cached-profile index is mapped to its PCI address first.
fn selector_for(
    spec: &str,
    profile: Option<&breccia_profile::schema::HwProfile>,
) -> Result<DeviceSelector> {
    let t = spec.trim();
    if t.is_empty() {
        anyhow::bail!("--device needs a profile index or a PCI address");
    }
    if t.contains(':') {
        return Ok(DeviceSelector::Pci(breccia_vk::pci::normalize_address(t)));
    }
    let idx: u32 = t.parse().map_err(|_| {
        anyhow::anyhow!(
            "--device {t:?} is neither a profile index nor a PCI address (dddd:bb:dd.f)"
        )
    })?;
    if let Some(p) = profile {
        if let Some(entry) = p.devices.iter().find(|d| d.identity.index == idx) {
            if !entry.identity.pci_address.is_empty() {
                println!(
                    "device {idx} -> {} [pci {}] (resolved through the cached profile; Vulkan \
                     ordinals are not stable on this host)",
                    entry.identity.name, entry.identity.pci_address
                );
                return Ok(DeviceSelector::Pci(entry.identity.pci_address.clone()));
            }
        } else {
            println!(
                "note: --device {idx} is not in the cached profile ({} device(s)); using the \
                 runtime Vulkan index",
                p.devices.len()
            );
        }
    }
    Ok(DeviceSelector::Index(idx))
}

fn print_transfer_summary(rep: &breccia_vk::TransferReport) {
    println!(
        "\ntransfer harness ({} mode, schema v{}), {} device(s)",
        if rep.environment.quick {
            "quick"
        } else {
            "full"
        },
        rep.schema_version,
        rep.devices.len()
    );
    println!(
        "  host {} · instance API {} · {}",
        rep.environment.hostname, rep.environment.instance_api_version, rep.environment.timestamp
    );
    if !rep.environment.instance_layers.is_empty() {
        println!("  instance layers: {:?}", rep.environment.instance_layers);
    }
    for w in &rep.warnings {
        println!("  warning: {w}");
    }

    for d in &rep.devices {
        println!("\n[{}] {}", d.device.index, d.device.name);
        println!(
            "  pci {} · {} {} · {}",
            if d.device.pci_address.is_empty() {
                "?"
            } else {
                &d.device.pci_address
            },
            d.device.driver_name,
            d.device.driver_version,
            d.device.device_type
        );
        println!(
            "  queues: compute family {} ({}q) · transfer family {} ({}q, index {}) · {}",
            d.queue.compute_family,
            d.queue.compute_family_queue_count,
            d.queue.transfer_family,
            d.queue.transfer_family_queue_count,
            d.queue.transfer_queue_index,
            d.queue.mode
        );
        println!(
            "  timeline semaphores: supported={} used={}",
            d.capabilities.timeline_semaphore_supported, d.capabilities.timeline_semaphore_used
        );
        let hm = &d.capabilities.host_memory;
        println!(
            "  external memory host: listed={} enabled={} align={} B import={}{}{}",
            hm.extension_listed,
            hm.extension_enabled,
            hm.min_imported_host_pointer_alignment,
            if hm.import_ok { "ok" } else { "unavailable" },
            if hm.import_verified {
                " (verified)"
            } else {
                ""
            },
            hm.h2d_gbps
                .map(|v| format!(" · {v:.1} GB/s zero-copy H2D"))
                .unwrap_or_default()
        );
        if let Some(r) = &d.ring {
            for s in &r.sizes {
                println!(
                    "  ring {:>3} MiB x{} slots: {:6.2} GB/s DMA, {:6.2} GB/s with host staging \
                     ({} rounds, {} slot stalls, {:.3} s blocked)",
                    s.size_mb,
                    s.slots,
                    s.dma_gbps,
                    s.staged_gbps,
                    s.rounds,
                    s.stall_count,
                    s.stall_seconds
                );
            }
        }
        if let Some(c) = &d.churn {
            println!(
                "  churn ({} rounds, {} MiB, depth {}): stream {:.3} s · compute-only {:.3} s · \
                 transfer-only {:.3} s",
                c.rounds,
                c.bytes_total >> 20,
                c.pipeline_depth,
                c.stream_seconds,
                c.compute_only_seconds,
                c.transfer_only_seconds
            );
            println!(
                "  overlap {:.1}% · consumer stall {:.2} ms/MB ({:.3} s total) · effective H2D \
                 {:.2} GB/s (full rate {:.2} GB/s)",
                c.overlap_efficiency_percent,
                c.consumer_stall_ms_per_mb,
                c.consumer_stall_seconds,
                c.stream_h2d_gbps,
                c.full_rate_h2d_gbps
            );
        }
        for n in &d.notes {
            println!("  note: {n}");
        }
        if let Some(e) = &d.error {
            println!("  ERROR: {e}");
        }
    }
}

fn serve(dry_run: bool, args: Vec<String>) -> Result<()> {
    if args.is_empty() {
        println!(
            "serve: phase-2 (no arguments given).\n\
             Pass the llama.cpp/HIP flag surface, e.g.:\n  \
             breccia serve --dry-run-plan -m model.gguf --device ROCm0,ROCm1 -ngl 99 -c 32768"
        );
        return Ok(());
    }

    let plan = breccia_flags::parse_llama_server_args(&args)
        .map_err(|e| anyhow::anyhow!("launch flags: {e}"))?;

    println!("backend policy : {:?} (Vulkan = primary)", plan.backend);
    if let Some(m) = &plan.model {
        println!("model          : {}", m.display());
    }
    if let Some(b) = &plan.binary {
        println!(
            "binary         : {} (reference; breccia executes in-process later)",
            b.display()
        );
    }
    if let Some(c) = plan.ctx_size {
        let np = plan.parallel.unwrap_or(1);
        println!(
            "context        : {c} tokens, np={np}, kv={:?}/{:?}, fa={:?}",
            plan.cache_type_k, plan.cache_type_v, plan.flash_attn
        );
    }
    if let Some(ngl) = plan.gpu_layers {
        println!(
            "gpu layers     : {ngl}, tensor-split {:?}, ubatch {:?}",
            plan.tensor_split, plan.ubatch_size
        );
    }
    if plan.cpu_moe {
        println!(
            "moe            : cpu_moe=true, moe-cache-mib={:?}, no_host={}",
            plan.moe_cache_mib, plan.no_host
        );
    }
    if let Some(p) = &plan.slot_save_path {
        println!("slot-save      : {}", p.display());
    }
    if let Some(mm) = &plan.mmproj {
        println!(
            "mmproj         : {} (device {:?})",
            mm.display(),
            plan.mmproj_device
        );
    }
    println!("listen         : {:?}:{:?}", plan.host, plan.port);

    // ---- device resolution: alias -> backend index, by PCI identity ----
    let profile = breccia_profile::load_profile(&breccia_profile::default_profile_path())
        .ok()
        .flatten();
    let hip_buses = if plan
        .devices
        .iter()
        .any(|d| d.backend_hint == breccia_flags::BackendHint::Hip)
    {
        hip_pci_buses()
    } else {
        None
    };

    println!("\ndevice resolution");
    for (i, dev) in plan.devices.iter().enumerate() {
        let alias = format!("{:?}{}", dev.backend_hint, dev.index);
        match dev.backend_hint {
            breccia_flags::BackendHint::Hip => {
                let pci = hip_buses
                    .as_ref()
                    .and_then(|b| b.get(dev.index as usize))
                    .cloned();
                match (pci, &profile) {
                    (Some(pci), Some(p)) => {
                        match p.devices.iter().find(|d| d.identity.pci_address.eq_ignore_ascii_case(&pci)) {
                            Some(entry) => println!(
                                "  [{i}] {alias:>8} (HIP {}) -> {}  [pci {}] — bound by PCI identity at launch",
                                dev.index, entry.identity.name, pci
                            ),
                            None => println!(
                                "  [{i}] {alias:>8} -> pci {pci} not found in profile — profile is stale, re-run `breccia profile --reprofile`",
                            ),
                        }
                    }
                    _ => println!(
                        "  [{i}] {alias:>8} -> ordinal mapping (no profile / rocm-smi) — identity must be verified",
                    ),
                }
            }
            breccia_flags::BackendHint::Vulkan => {
                // Vulkan ordinals are NOT stable across runs on this host (RADV
                // intermittently omits the iGPU and reorders enumeration), so a
                // VulkanN alias is informational only.
                let name = profile
                    .as_ref()
                    .and_then(|p| p.devices.iter().find(|d| d.identity.index == dev.index))
                    .map(|d| format!("{} [pci {}]", d.identity.name, d.identity.pci_address))
                    .unwrap_or_else(|| "<not in cached profile>".to_string());
                println!(
                    "  [{i}] {alias:>8} -> calibration-time ordinal {}. {name}; \
                     runtime binds by identity because Vulkan enumeration order is unstable",
                    dev.index
                );
            }
            breccia_flags::BackendHint::Cpu => println!("  [{i}] {alias:>8} -> host CPU"),
        }
    }

    // ---- tensor overrides: compile and report ----
    match plan.validated_overrides() {
        Ok(v) => println!(
            "\noverrides      : {} compiled ({} tensor regexes)",
            v.len(),
            plan.override_tensors.len()
        ),
        Err(errs) => {
            println!("\noverrides      : {} FAILED to compile", errs.len());
            for e in errs {
                println!("  {e}");
            }
            anyhow::bail!("override regex validation failed");
        }
    }

    for w in plan.warnings() {
        // The alias warning states an ordinal remap, but ordinals are not stable on
        // this host (RADV enumeration order varies). The identity resolution above is
        // authoritative; restate the alias fact without a misleading ordinal.
        if matches!(w.code, breccia_flags::WarningCode::HipAliasOnVulkan) {
            continue;
        }
        println!("warning        : {w}");
    }
    if plan
        .warnings()
        .iter()
        .any(|w| matches!(w.code, breccia_flags::WarningCode::HipAliasOnVulkan))
    {
        println!(
            "note           : HIP/CUDA device aliases accepted; backend stays Vulkan \
             (binding by PCI identity, ordinals deliberately ignored)"
        );
    }

    println!(
        "\n{}",
        if dry_run {
            "dry run: nothing launched, nothing written"
        } else {
            "serve: phase-2 (plan parsed; execution lands with the engine)"
        }
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pci_spec_is_normalised() {
        assert_eq!(
            selector_for("03:00.0", None).expect("pci spec"),
            DeviceSelector::Pci("0000:03:00.0".into())
        );
        assert_eq!(
            selector_for(" 0000:0E:00.0 ", None).expect("full pci spec"),
            DeviceSelector::Pci("0000:0e:00.0".into())
        );
    }

    #[test]
    fn numeric_spec_falls_back_to_runtime_index() {
        // Without a cached profile the index is used verbatim (with a printed note
        // when a profile exists but does not carry that index).
        assert_eq!(
            selector_for("1", None).expect("index"),
            DeviceSelector::Index(1)
        );
    }

    #[test]
    fn garbage_spec_is_rejected() {
        assert!(selector_for("rx6950", None).is_err());
        assert!(selector_for("  ", None).is_err());
    }
}

// ---------------------------------------------------------------------------
// bench gemv — Phase 3 fused dequant+GEMV (decode path)
// ---------------------------------------------------------------------------

struct GemvArgs {
    types: Vec<String>,
    rows: Option<usize>,
    cols: Option<usize>,
    device: Vec<String>,
    json: Option<PathBuf>,
    quick: bool,
    real: bool,
    gguf: Option<String>,
    layer: usize,
    role: String,
    expert: Option<usize>,
    splits: Option<u32>,
    tensor: Option<String>,
}

/// Everything a `bench gemv` JSON file contains: one run per device, because
/// each device needs its own instance/logical device and its own numbers.
#[derive(serde::Serialize)]
struct GemvBundle {
    schema_version: u32,
    runs: Vec<breccia_kernels::GemvReport>,
}

fn bench_gemv(args: GemvArgs) -> Result<()> {
    use breccia_kernels::{GemvBenchOptions, QuantType};

    let types: Vec<QuantType> = args
        .types
        .iter()
        .map(|t| t.parse::<QuantType>().map_err(|e| anyhow::anyhow!(e)))
        .collect::<Result<_>>()?;

    let shapes = match (args.rows, args.cols) {
        (Some(r), Some(c)) => vec![breccia_kernels::Shape::new(r, c)],
        (None, None) => Vec::new(),
        _ => anyhow::bail!("--rows and --cols must be given together"),
    };

    let profile = breccia_profile::load_profile(&breccia_profile::default_profile_path())
        .ok()
        .flatten();
    let devices: Vec<String> = if args.device.is_empty() {
        breccia_kernels::bench::discrete_devices()?
    } else {
        args.device
            .iter()
            .map(|spec| {
                Ok(match selector_for(spec, profile.as_ref())? {
                    DeviceSelector::Pci(pci) => pci,
                    DeviceSelector::Index(i) => i.to_string(),
                })
            })
            .collect::<Result<_>>()?
    };

    let mut runs = Vec::new();
    for dev in &devices {
        let opts = GemvBenchOptions {
            types: types.clone(),
            shapes: shapes.clone(),
            devices: vec![dev.clone()],
            quick: args.quick,
            real: args.real,
            real_model: args.gguf.clone(),
            real_layer: args.layer,
            real_role: args.role.clone(),
            real_expert: args.expert,
            splits: args.splits,
            real_tensor: args.tensor.clone(),
        };
        let report = breccia_kernels::run_gemv_bench(&opts)
            .with_context(|| format!("GEMV benchmark on {dev}"))?;
        print_gemv_summary(&report);
        runs.push(report);
    }

    if let Some(path) = &args.json {
        let bundle = GemvBundle {
            schema_version: breccia_kernels::bench::GEMV_SCHEMA_VERSION,
            runs,
        };
        let text = serde_json::to_string_pretty(&bundle).context("serializing the GEMV report")?;
        if let Some(dir) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        std::fs::write(path, text + "\n").with_context(|| format!("writing {}", path.display()))?;
        println!("\nwrote {}", path.display());
    }
    Ok(())
}

fn print_gemv_summary(rep: &breccia_kernels::GemvReport) {
    println!(
        "\nGEMV decode benchmark ({} mode, schema v{})",
        if rep.quick { "quick" } else { "full" },
        rep.schema_version
    );
    println!(
        "  host {} · {} · device {} ({}) {} · subgroup {}",
        rep.environment.hostname,
        rep.environment.timestamp,
        rep.device.name,
        rep.device.device_id,
        rep.device.pci_address,
        rep.device.subgroup_size
    );
    if rep.environment.vulkan_validation {
        println!("  validation layer: enabled");
    }
    for w in &rep.warnings {
        println!("  warning: {w}");
    }

    if !rep.real.is_empty() {
        for c in &rep.real {
            println!(
                "\n  {} ({} {}) as {}x{} — {} MiB of packed weights{}",
                c.tensor,
                c.ty,
                c.shape,
                c.shape.rows,
                c.shape.cols,
                c.weight_bytes >> 20,
                c.expert
                    .map(|e| format!(", expert {e} of {}", c.experts))
                    .unwrap_or_else(|| format!(", all {} experts", c.experts))
            );
            println!(
                "    {:8.3} ms/token-equivalent · {:7.1} GB/s{} · {:.0} tok/s if this were \
                 the whole token",
                c.seconds_per_iter * 1e3,
                c.gbps,
                c.reference_gbps
                    .map(|bw| format!(" ({:.0}% of {bw:.0} GB/s)", 100.0 * c.gbps / bw))
                    .unwrap_or_default(),
                c.tokens_per_s_equivalent
            );
            println!(
                "    upload {:.2} s (one-off) · {} iters · timestamps {}",
                c.upload_seconds,
                c.iters,
                if c.used_timestamps {
                    "yes"
                } else {
                    "no (wall clock)"
                }
            );
            if let Some(rel) = c.conformance_max_rel {
                println!(
                    "    conformance vs CPU reference ({} rows): max relative error {:.2e}",
                    c.conformance_rows, rel
                );
            }
        }
        return;
    }

    println!(
        "\n  {:<8} {:>12} {:>6} {:>10} {:>11} {:>9} {:>9}",
        "type", "shape", "bpw", "MiB read", "ms/GEMV", "GB/s", "of DRAM"
    );
    for c in &rep.synthetic {
        println!(
            "  {:<8} {:>12} {:>6.2} {:>10.2} {:>11.3} {:>9.1} {:>8} {:>5}",
            c.ty,
            format!("{}x{}", c.shape.rows, c.shape.cols),
            c.bits_per_value,
            c.weight_bytes as f64 / (1 << 20) as f64,
            c.seconds_per_iter * 1e3,
            c.gbps,
            c.percent_of_reference
                .map(|p| format!("{p:.0}%"))
                .unwrap_or_else(|| "-".to_string()),
            c.splits
        );
    }
}
