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
    /// Phase 2: OpenAI-compatible agent server (not implemented yet)
    Serve {
        /// Accepted and ignored for now (llama.cpp-compatible surface arrives in Phase 2)
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        _args: Vec<String>,
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
        Command::Serve { .. } => {
            println!("serve: phase-2");
            Ok(())
        }
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
