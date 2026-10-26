//! `breccia` CLI entry (`breccia profile|serve|bench`).
//!
//! Phase 1 wires `breccia profile` (hardware discovery + calibration →
//! `hw_profile.json`) on top of `breccia-vk` + `breccia-profile`. The
//! llama.cpp/HIP-compatible launch surface (`breccia-flags`) and the server
//! arrive with Phase 2, which is where `serve`/`bench` become real.
//!
//! ```text
//! breccia profile [--reprofile] [--quick] [--json PATH] [--dry-run-plan] [--print-fingerprint]
//! ```
//!
//! Behaviour: with a cached profile whose fingerprint matches (and whose
//! benchmark window is at least as long as requested) it prints the cached
//! summary and exits 0 without touching the GPU; otherwise it calibrates and
//! writes. `--dry-run-plan` prints the role table + IQ3_XXS placement sketch and
//! writes nothing.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

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
    /// Phase 2: benchmark harness (not implemented yet)
    Bench {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        _args: Vec<String>,
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
        Command::Bench { .. } => {
            println!("bench: phase-2");
            Ok(())
        }
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
