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
//! breccia bench gemv --experts 1,2,4,8,10,16 [--type T]... [--rows N --cols N]   # batched MoE decode
//! breccia bench ops --op rmsnorm [--tokens 1,256] [--device <idx|pci>] [--json PATH]
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

// The `bench gemv` variant carries the whole decode-benchmark surface (paths,
// names, expert counts); this is a parse-only enum built once per process, so
// boxing it to equalise variant sizes would only add noise.
#[allow(clippy::large_enum_variant)]
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
#[allow(clippy::large_enum_variant)] // parse-only, built once per process (same reason as `Command`)
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
        /// Batched multi-expert (MoE decode) mode: expert counts to batch in one
        /// dispatch, e.g. --experts 1,2,4,8,10,16 (repeatable). Compares against
        /// the per-expert loop in the same run.
        #[arg(long, value_name = "N[,N...]", value_delimiter = ',')]
        experts: Vec<usize>,
        /// Dot-product accumulator: f32 (default, the conformance bar) or f16
        /// (research path — measured error ~1e-2, see docs/phase-3/03)
        #[arg(long, value_name = "f32|f16", default_value = "f32")]
        accum: String,
    },
    /// Phase 3: elementwise/utility op kernels — latency and effective GB/s
    Ops {
        /// One op per invocation (AGENTS.md GPU crash safety);
        /// default = every op in sequence
        #[arg(long = "op", value_name = "OP")]
        ops: Vec<String>,
        /// Batch sizes (decode = 1, prefill chunk = 256)
        #[arg(long, value_name = "N[,N...]", value_delimiter = ',', default_values_t = [1usize, 256])]
        tokens: Vec<usize>,
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
    /// Phase 3 (CPU): host resident-expert lane — dequant+GEMV GB/s, MoE token
    CpuExperts {
        /// Quant type, repeatable (default: every type the model's experts use)
        #[arg(long = "type", value_name = "IQ2_XS")]
        types: Vec<String>,
        /// Worker threads for the top-k token pass (default: physical cores/2,
        /// capped at 16)
        #[arg(long, value_name = "N")]
        threads: Option<usize>,
        /// Routed experts per block in the token pass (the model uses 10)
        #[arg(long, default_value_t = 10)]
        topk: usize,
        /// GGUF glob for the model (default: the IQ3_XXS shards)
        #[arg(long, value_name = "GLOB")]
        gguf: Option<String>,
        /// Expert slices per per-type cell (default: 96, 16 with --quick)
        #[arg(long, value_name = "N")]
        type_experts: Option<usize>,
        /// Measure only the first N blocks (0 = all 48)
        #[arg(long, default_value_t = 0)]
        layers: usize,
        /// Token-pass repetitions; the best is reported, the first is the cold one
        #[arg(long, default_value_t = 3)]
        reps: usize,
        /// Write the full JSON report here
        #[arg(long, value_name = "PATH")]
        json: Option<PathBuf>,
        /// Short runs (fewer experts, fewer reps, no scalar token pass)
        #[arg(long)]
        quick: bool,
    },
    /// Phase 4: the forward-pass graph — one token, then greedy generation
    Token {
        /// Tokens to generate after the prompt (greedy, temperature 0)
        #[arg(long, default_value_t = 32)]
        gen: usize,
        /// Prompt token ids, comma-separated (`--prompt-ids 1,2,3`). With
        /// `--prompt-ids-file`, the file's ids are used instead.
        #[arg(long, value_name = "ID,ID,...")]
        prompt_ids: Option<String>,
        /// A file of whitespace/comma-separated prompt ids
        #[arg(long, value_name = "PATH")]
        prompt_ids_file: Option<PathBuf>,
        /// Device plan: comma-separated PCI addresses (`0000:03:00.0,0000:0e:00.0`),
        /// runtime indices, or `default` (first discrete). Repeatable; the
        /// default is one device. **The first run of new code must be
        /// MI50-only** (`--devices 0000:0e:00.0`, AGENTS.md).
        #[arg(long = "devices", value_name = "PCI[,PCI]", value_delimiter = ',')]
        devices: Vec<String>,
        /// Measure this many independent generations (the best is reported)
        #[arg(long, default_value_t = 3)]
        runs: usize,
        /// Host expert-lane threads (default: physical cores/2, capped at 16)
        #[arg(long, value_name = "N")]
        threads: Option<usize>,
        /// KV cells (and positions) the session may use
        #[arg(long, default_value_t = 256)]
        ctx: usize,
        /// Digests of every block's activations (for diffing runs)
        #[arg(long)]
        debug_layers: bool,
        /// Sampling temperature (`0` = greedy, the parity path)
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
        /// GGUF glob for the model (default: the IQ3_XXS shards)
        #[arg(long, value_name = "GLOB")]
        gguf: Option<String>,
        /// Per-stage timers (host-clock deltas around synchronous dispatches)
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        timers: bool,
        /// Write the full JSON report here
        #[arg(long, value_name = "PATH")]
        json: Option<PathBuf>,
        /// Write the prompt's last-token logits here (`id logit` per line, the
        /// `--top-logits` best) — the row a parity diff is made against
        #[arg(long, value_name = "PATH")]
        dump_logits: Option<PathBuf>,
        /// How many logits `--dump-logits` keeps
        #[arg(long, default_value_t = 256)]
        top_logits: usize,
        /// Record each GPU phase into one command buffer (experimental: the
        /// mechanism is verified bit-identical, the graph integration is not)
        #[arg(long)]
        batching: bool,
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
                 [--json PATH] [--quick] [--real [--layer N] [--role gate|up|down] [--expert E]]\n                 \
                 [--experts 1,2,4,8,10,16] [--accum f32|f16]\n  \
                 breccia bench ops --op rmsnorm|rope|swiglu|softmax|router|emb_gather|hc|add|gdn|qsa \\\n                 \
                 [--tokens 1,256] [--device <idx|pci>] [--json PATH] [--quick]\n  \
                 breccia bench cpu-experts [--type IQ2_XS]... [--threads N] [--topk 10] \\\n                 \
                 [--type-experts N] [--layers N] [--reps N] [--gguf GLOB] [--json PATH] [--quick]\n  \
                 breccia bench token --gen 32 --prompt-ids \"1,2,3\" [--devices PCI[,PCI]] [--json PATH] \\\n                 \
                 [--debug-layers] [--runs 3] [--threads N] [--ctx 256] [--temperature T]"
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
                    experts,
                    accum,
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
            experts,
            accum,
        }),
        Command::Bench {
            kind:
                Some(BenchKind::Ops {
                    ops,
                    tokens,
                    device,
                    json,
                    quick,
                }),
        } => bench_ops(OpsArgs {
            ops,
            tokens,
            device,
            json,
            quick,
        }),
        Command::Bench {
            kind:
                Some(BenchKind::CpuExperts {
                    types,
                    threads,
                    topk,
                    gguf,
                    type_experts,
                    layers,
                    reps,
                    json,
                    quick,
                }),
        } => bench_cpu_experts(CpuExpertsArgs {
            types,
            threads,
            topk,
            gguf,
            type_experts,
            layers,
            reps,
            json,
            quick,
        }),
        Command::Bench {
            kind:
                Some(BenchKind::Token {
                    gen,
                    prompt_ids,
                    prompt_ids_file,
                    devices,
                    runs,
                    threads,
                    ctx,
                    debug_layers,
                    temperature,
                    gguf,
                    timers,
                    json,
                    dump_logits,
                    top_logits,
                    batching,
                }),
        } => bench_token(TokenArgs {
            gen,
            prompt_ids,
            prompt_ids_file,
            devices,
            runs,
            threads,
            ctx,
            debug_layers,
            temperature,
            gguf,
            timers,
            json,
            dump_logits,
            top_logits,
            batching,
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
    experts: Vec<usize>,
    accum: String,
}

/// Everything a `bench gemv` JSON file contains: one run per device, because
/// each device needs its own instance/logical device and its own numbers.
#[derive(serde::Serialize)]
struct GemvBundle {
    schema_version: u32,
    runs: Vec<breccia_kernels::GemvReport>,
}

fn bench_gemv(args: GemvArgs) -> Result<()> {
    use breccia_kernels::{Accum, GemvBenchOptions, QuantType};

    let accum: Accum = args
        .accum
        .parse::<Accum>()
        .map_err(|e| anyhow::anyhow!(e))?;

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
            experts: args.experts.clone(),
            accum,
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
    if let Some(acc) = rep
        .synthetic
        .first()
        .map(|c| c.accum.clone())
        .or_else(|| rep.real.first().map(|c| c.accum.clone()))
        .or_else(|| rep.batch.first().map(|c| c.accum.clone()))
    {
        if acc != "f32" {
            println!("  accumulator: {acc} (non-default; f32 is the conformance path)");
        }
    }
    for w in &rep.warnings {
        println!("  warning: {w}");
    }

    if !rep.batch.is_empty() {
        println!(
            "\n  batched multi-expert (one dispatch over N experts, per-expert loop measured in the same run)"
        );
        println!(
            "  {:<8} {:>11} {:>4} {:>9} {:>11} {:>11} {:>8} {:>9} {:>8}",
            "type",
            "expert",
            "N",
            "x-layout",
            "loop ms",
            "batched ms",
            "speedup",
            "loop GB/s",
            "bt GB/s"
        );
        for c in &rep.batch {
            println!(
                "  {:<8} {:>11} {:>4} {:>9} {:>11.3} {:>11.3} {:>7.2}x {:>9.1} {:>8.1}{}",
                c.ty,
                c.shape.to_string(),
                c.experts,
                c.x_layout,
                c.loop_seconds * 1e3,
                c.batched_seconds * 1e3,
                c.speedup,
                c.loop_gbps,
                c.batched_gbps,
                c.percent_of_reference
                    .map(|p| format!(" ({p:.0}% of DRAM)"))
                    .unwrap_or_default()
            );
            if let (Some(t), Some(rel)) = (&c.tensor, c.conformance_max_rel) {
                println!(
                    "           {} · experts {:?} · {} MiB read · conformance {} rows max rel {:.2e}{}",
                    t,
                    c.expert_ids.clone().unwrap_or_default(),
                    c.total_bytes >> 20,
                    c.conformance_rows.unwrap_or(0),
                    rel,
                    c.upload_seconds
                        .map(|s| format!(" · upload {s:.2} s"))
                        .unwrap_or_default()
                );
            }
        }
        return;
    }

    if !rep.real.is_empty() {
        for c in &rep.real {
            println!(
                "\n  {} ({} {}) as {}x{} — {} of packed weights{}",
                c.tensor,
                c.ty,
                c.shape,
                c.shape.rows,
                c.shape.cols,
                if c.weight_bytes >= 1 << 20 {
                    format!("{} MiB", c.weight_bytes >> 20)
                } else {
                    format!("{} KiB", c.weight_bytes >> 10)
                },
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
            if let Some([start, end]) = c.file_range {
                println!(
                    "    file bytes 0x{start:x}..0x{end:x} ({:.0} KiB){}",
                    (end - start) as f64 / 1024.0,
                    if c.folded_rows {
                        " · rows folded to a whole kernel block"
                    } else {
                        ""
                    }
                );
            }
        }
        return;
    }

    println!(
        "\n  {:<8} {:>12} {:>6} {:>10} {:>11} {:>9} {:>9} {:>9}",
        "type", "shape", "bpw", "MiB read", "ms/GEMV", "GB/s", "of DRAM", "max rel"
    );
    for c in &rep.synthetic {
        println!(
            "  {:<8} {:>12} {:>6.2} {:>10.2} {:>11.3} {:>9.1} {:>8} {:>5} {:>9.2e}",
            c.ty,
            format!("{}x{}", c.shape.rows, c.shape.cols),
            c.bits_per_value,
            c.weight_bytes as f64 / (1 << 20) as f64,
            c.seconds_per_iter * 1e3,
            c.gbps,
            c.percent_of_reference
                .map(|p| format!("{p:.0}%"))
                .unwrap_or_else(|| "-".to_string()),
            c.splits,
            c.conformance_max_rel.unwrap_or(f64::NAN),
        );
    }
}

// ---------------------------------------------------------------------------
// bench ops — Phase 3 elementwise/utility op kernels
// ---------------------------------------------------------------------------

struct OpsArgs {
    ops: Vec<String>,
    tokens: Vec<usize>,
    device: Vec<String>,
    json: Option<PathBuf>,
    quick: bool,
}

/// Everything a `bench ops` JSON file contains: one run per device, because each
/// device needs its own instance/logical device and its own numbers.
#[derive(serde::Serialize)]
struct OpsBundle {
    schema_version: u32,
    runs: Vec<breccia_kernels::OpsReport>,
}

fn bench_ops(args: OpsArgs) -> Result<()> {
    use breccia_kernels::{OpKind, OpsBenchOptions};

    let ops: Vec<OpKind> = args
        .ops
        .iter()
        .map(|s| s.parse::<OpKind>().map_err(|e| anyhow::anyhow!(e)))
        .collect::<Result<_>>()?;
    let tokens: Vec<usize> = args.tokens.iter().copied().filter(|t| *t > 0).collect();
    if tokens.is_empty() {
        anyhow::bail!("--tokens needs at least one non-zero batch size");
    }

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
        let opts = OpsBenchOptions {
            ops: ops.clone(),
            devices: vec![dev.clone()],
            quick: args.quick,
            tokens: tokens.clone(),
        };
        let report = breccia_kernels::run_ops_bench(&opts)
            .with_context(|| format!("ops benchmark on {dev}"))?;
        print_ops_summary(&report);
        runs.push(report);
    }

    if let Some(path) = &args.json {
        let bundle = OpsBundle {
            schema_version: breccia_kernels::ops::OPS_SCHEMA_VERSION,
            runs,
        };
        let text = serde_json::to_string_pretty(&bundle).context("serializing the ops report")?;
        if let Some(dir) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        std::fs::write(path, text + "\n").with_context(|| format!("writing {}", path.display()))?;
        println!("\nwrote {}", path.display());
    }
    Ok(())
}

fn print_ops_summary(rep: &breccia_kernels::OpsReport) {
    let e = &rep.environment;
    println!(
        "\nop-kernel benchmark ({} mode, schema v{})",
        if rep.quick { "quick" } else { "full" },
        rep.schema_version
    );
    println!(
        "  host {} · {} · device {} ({}) {} · subgroup {}",
        e.hostname,
        e.timestamp,
        rep.device.name,
        rep.device.device_id,
        rep.device.pci_address,
        rep.device.subgroup_size
    );
    println!(
        "  geometry: n_embd {} · hc {}/{} · head_dim {}/n_rot {} · {} experts top-{} · n_ff {} · n_head {}",
        e.n_embd, e.hc, e.hc_lr, e.head_dim, e.n_rot, e.n_expert, e.top_k, e.n_ff, e.n_head
    );
    if e.vulkan_validation {
        println!("  validation layer: enabled");
    }
    for w in &rep.warnings {
        println!("  warning: {w}");
    }

    println!(
        "\n  {:<11} {:<20} {:>5} {:>12} {:>10} {:>9} {:>9} {:>10}",
        "op", "case", "T", "units", "µs", "GB/s", "GFLOP/s", "MiB moved"
    );
    for c in &rep.cells {
        println!(
            "  {:<11} {:<20} {:>5} {:>12} {:>10.2} {:>9.1} {:>9.1} {:>10.2}",
            c.op,
            c.case,
            c.tokens,
            format!("{} {}", c.units, c.units_name),
            c.us,
            c.gbps,
            c.gops,
            c.bytes as f64 / (1 << 20) as f64
        );
    }
    println!(
        "  (µs is one dispatch, best of {} windows; GB/s = bytes read+written / time; \
         timestamps {})",
        rep.cells.first().map(|c| c.us_all.len()).unwrap_or(0),
        if rep.cells.iter().all(|c| c.used_timestamps) {
            "on"
        } else {
            "partly wall clock"
        }
    );
}

// ---------------------------------------------------------------------------
// bench cpu-experts (Phase 3, host resident-expert lane)
// ---------------------------------------------------------------------------

struct CpuExpertsArgs {
    types: Vec<String>,
    threads: Option<usize>,
    topk: usize,
    gguf: Option<String>,
    type_experts: Option<usize>,
    layers: usize,
    reps: usize,
    json: Option<PathBuf>,
    quick: bool,
}

#[derive(serde::Serialize)]
struct CpuExpertsBundle {
    schema_version: u32,
    runs: Vec<breccia_kernels::cpu::CpuExpertReport>,
}

fn bench_cpu_experts(args: CpuExpertsArgs) -> Result<()> {
    use breccia_kernels::cpu::{self, CpuExpertBenchOptions};
    use breccia_kernels::QuantType;

    let types: Vec<QuantType> = args
        .types
        .iter()
        .map(|t| t.parse::<QuantType>().map_err(|e| anyhow::anyhow!(e)))
        .collect::<Result<_>>()?;

    let opts = CpuExpertBenchOptions {
        threads: args.threads.unwrap_or(0),
        types,
        topk: args.topk,
        model_glob: args
            .gguf
            .clone()
            .unwrap_or_else(|| cpu::DEFAULT_CPU_MODEL_GLOB.to_string()),
        quick: args.quick,
        reps: args.reps,
        type_experts: args.type_experts.unwrap_or(0),
        layers: args.layers,
        json: args.json.clone(),
    };
    let report = cpu::run_cpu_expert_bench(&opts).context("CPU expert lane benchmark")?;
    cpu::print_summary(&report);

    if let Some(path) = &args.json {
        let bundle = CpuExpertsBundle {
            schema_version: cpu::CPU_EXPERTS_SCHEMA_VERSION,
            runs: vec![report],
        };
        let text = serde_json::to_string_pretty(&bundle).context("serializing the report")?;
        if let Some(dir) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        std::fs::write(path, text + "\n").with_context(|| format!("writing {}", path.display()))?;
        println!("\nwrote {}", path.display());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// breccia bench token — Phase 4: the forward-pass graph
// ---------------------------------------------------------------------------

/// The IQ3_XXS shards, the only artifact this machine has.
const DEFAULT_MODEL_GLOB: &str = "/srv/strata-data/models/IQ3_XXS/*-0000?-of-00002.gguf";

struct TokenArgs {
    gen: usize,
    prompt_ids: Option<String>,
    prompt_ids_file: Option<PathBuf>,
    devices: Vec<String>,
    runs: usize,
    threads: Option<usize>,
    ctx: usize,
    debug_layers: bool,
    temperature: f32,
    gguf: Option<String>,
    timers: bool,
    json: Option<PathBuf>,
    dump_logits: Option<PathBuf>,
    top_logits: usize,
    batching: bool,
}

/// Prompt ids from `--prompt-ids` or `--prompt-ids-file` (comma/space separated).
fn parse_prompt_ids(a: &TokenArgs) -> Result<Vec<u32>> {
    let text = match (&a.prompt_ids_file, &a.prompt_ids) {
        (Some(path), _) => std::fs::read_to_string(path)
            .with_context(|| format!("reading the prompt ids from {}", path.display()))?,
        (None, Some(s)) => s.clone(),
        (None, None) => anyhow::bail!(
            "no prompt: pass --prompt-ids \"1,2,3\" (or --prompt-ids-file PATH). \
             Tokenize real text with the production server: \
             curl -s localhost:8164/tokenize -d '{{\"content\":\"...\"}}'"
        ),
    };
    let mut out = Vec::new();
    for tok in text.split(|c: char| c == ',' || c.is_whitespace()) {
        let t = tok.trim();
        if t.is_empty() {
            continue;
        }
        let id: u32 = t
            .parse()
            .with_context(|| format!("prompt id {t:?} is not a token id"))?;
        out.push(id);
    }
    if out.is_empty() {
        anyhow::bail!("the prompt id list is empty");
    }
    Ok(out)
}

/// Reset and feed every prompt token; returns the last token's logits.
fn feed_prompt(sess: &mut breccia_engine::Session<'_>, prompt: &[u32]) -> Result<Vec<f32>> {
    sess.reset()?;
    let mut logits = Vec::new();
    for (pos, &t) in prompt.iter().enumerate() {
        logits = sess.token(t, pos, None)?;
    }
    Ok(logits)
}

fn bench_token(a: TokenArgs) -> Result<()> {
    let prompt = parse_prompt_ids(&a)?;
    let glob = a
        .gguf
        .clone()
        .unwrap_or_else(|| DEFAULT_MODEL_GLOB.to_string());
    println!(
        "breccia bench token: {} prompt ids, {} to generate, {} run(s)",
        prompt.len(),
        a.gen,
        a.runs
    );

    // ---- load -------------------------------------------------------------
    let model = breccia_engine::Model::load(
        &glob,
        &breccia_engine::LoadOptions {
            devices: a.devices.clone(),
            ctx_capacity: a.ctx,
            verbose: true,
        },
    )
    .context("loading the model")?;
    print!("{}", model.stats.banner(&model.cfg));
    breccia_engine::graph::assert_head(&model)?;

    let cfg = breccia_engine::SessionCfg {
        ctx_capacity: a.ctx,
        threads: a.threads.unwrap_or(0),
        timers: a.timers,
        batching: a.batching,
    };
    let mut sess = breccia_engine::Session::new(&model, &cfg).context("building the session")?;
    sess.sampling = breccia_engine::Sampling {
        temperature: a.temperature,
        top_k: 0,
        seed: 0x9e37_79b9_7f4a_7c15,
    };
    println!(
        "session  {} expert threads ({}) · {} MiB state + scratch · {} compiled pipelines",
        sess.threads(),
        breccia_kernels::cpu::path().name(),
        sess.session_bytes() >> 20,
        model.keys.len()
    );

    // ---- A. one token, twice: finite logits and a stable top-1 -------------
    let t_first = std::time::Instant::now();
    let l1 = feed_prompt(&mut sess, &prompt).context("the first forward pass")?;
    let first_seconds = t_first.elapsed().as_secs_f64();
    let l2 = feed_prompt(&mut sess, &prompt).context("the repeat forward pass")?;
    let s1 = breccia_engine::LogitSummary::of_top(&l1, a.top_logits.max(10));
    let s2 = breccia_engine::LogitSummary::of(&l2);
    if let Some(path) = &a.dump_logits {
        // `id logit` per line, best first: the row a parity diff reads.
        let mut text = String::new();
        for (id, logit) in &s1.top {
            text.push_str(&format!("{id} {logit:.6}\n"));
        }
        if let Some(dir) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))?;
        println!(
            "             dumped the top {} logits to {}",
            s1.top.len(),
            path.display()
        );
    }
    let max_delta = l1
        .iter()
        .zip(l2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!(
        "\nfirst token  logits[{}] finite={} min={:.3} max={:.3} mean={:.4} rms={:.4} \
         distinct={:.3}\n             top-1 {} ({:.4}) · top-2 {} ({:.4}) · gap {:.4}",
        s1.vocab,
        s1.all_finite,
        s1.min,
        s1.max,
        s1.mean,
        s1.rms,
        s1.distinct_ratio,
        s1.top1,
        s1.top1_logit,
        s1.top2,
        s1.top2_logit,
        s1.top1_logit - s1.top2_logit
    );
    println!("             top-10 {:?}", s1.top10);
    println!(
        "             repeat: top-1 {} (stable: {}) · max |Δlogit| {:.3e} · {:.2} s/token \
         (prompt {})",
        s2.top1,
        s1.top1 == s2.top1,
        max_delta,
        first_seconds / prompt.len() as f64,
        prompt.len()
    );
    if !s1.all_finite {
        anyhow::bail!("the first token's logits are not all finite");
    }
    if s1.distinct_ratio < 0.5 {
        anyhow::bail!(
            "only {:.1}% of the logits are distinct — that looks like a broken head, not a model",
            s1.distinct_ratio * 100.0
        );
    }

    // ---- B/C. measured generations ----------------------------------------
    let mut runs = Vec::with_capacity(a.runs);
    let mut stages_acc: Vec<f64> = vec![0.0; breccia_engine::STAGES.len()];
    let mut traces: Vec<breccia_engine::LayerTrace> = Vec::new();
    let mut first_step_logits = Vec::new();
    let mut last_logits = Vec::new();
    for run in 0..a.runs.max(1) {
        let dbg: Option<&mut Vec<breccia_engine::LayerTrace>> = if a.debug_layers && run == 0 {
            Some(&mut traces)
        } else {
            None
        };
        let gen = sess
            .generate(&prompt, a.gen, dbg)
            .with_context(|| format!("generation run {run}"))?;
        let row = breccia_engine::RunReport {
            run,
            decode_seconds: gen.decode_seconds,
            forwards: gen.forwards,
            tokens_per_s: gen.tokens_per_s(),
            ms_per_token: gen.ms_per_token(),
            prompt_ids: prompt.clone(),
            generated: gen.tokens.clone(),
            expert_bytes_per_token: gen.bytes_per_token(),
        };
        for (i, (_, ms)) in sess.timers.table(gen.forwards).iter().enumerate() {
            stages_acc[i] += ms;
        }
        println!(
            "run {run}  {:>6.2} tok/s  {:>6.2} ms/token  {} forwards  experts {} MiB/token  {:?}",
            row.tokens_per_s,
            row.ms_per_token,
            row.forwards,
            row.expert_bytes_per_token >> 20,
            &row.generated[..row.generated.len().min(8)]
        );
        if run == 0 {
            first_step_logits = gen.first_step_logits.clone();
        }
        last_logits = gen.last_logits.clone();
        runs.push(row);
    }

    let runs_n = runs.len().max(1) as f64;
    for v in stages_acc.iter_mut() {
        *v /= runs_n;
    }
    let total_ms: f64 = stages_acc.iter().sum();
    let stages: Vec<breccia_engine::StageRow> = breccia_engine::STAGES
        .iter()
        .zip(stages_acc.iter())
        .map(|(name, ms)| breccia_engine::StageRow {
            stage: (*name).to_string(),
            ms_per_token: *ms,
            share: if total_ms > 0.0 { *ms / total_ms } else { 0.0 },
        })
        .collect();
    let mut order: Vec<usize> = (0..stages.len()).collect();
    order.sort_by(|&a, &b| {
        stages[b]
            .ms_per_token
            .partial_cmp(&stages[a].ms_per_token)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let hotspots: Vec<String> = order
        .iter()
        .take(6)
        .map(|&i| {
            format!(
                "{} {:.3} ms ({:.1}%)",
                stages[i].stage,
                stages[i].ms_per_token,
                stages[i].share * 100.0
            )
        })
        .collect();

    let forwards: usize = runs.iter().map(|r| r.forwards).sum();
    let seconds: f64 = runs.iter().map(|r| r.decode_seconds).sum();
    let tps = if seconds > 0.0 {
        forwards as f64 / seconds
    } else {
        0.0
    };
    let mspt = if forwards > 0 {
        seconds * 1e3 / forwards as f64
    } else {
        0.0
    };

    println!("\nper stage (ms/token, mean of {} run(s)):", runs.len());
    for row in &stages {
        if row.ms_per_token > 0.0 {
            println!(
                "  {:<16} {:>7.3}  {:>5.1}%",
                row.stage,
                row.ms_per_token,
                row.share * 100.0
            );
        }
    }
    println!(
        "\ndecode  {tps:.2} tok/s  ({mspt:.2} ms/token)  vs {:.2} tok/s parity = {:.1}%",
        breccia_engine::PARITY_TOKENS_PER_S,
        tps / breccia_engine::PARITY_TOKENS_PER_S * 100.0
    );

    // ---- the report -------------------------------------------------------
    // The parity anchor is the prompt's last-token row; `first_step` is kept for
    // the "did the second step drift" question.
    let first_summary = s1.clone();
    let _ = &first_step_logits;
    let notes = vec![
        "Expert lane is host-only (-cmoe): no -ot per-block GPU expert overrides (M3).".to_string(),
        "The router and every BF16-weight projection see an f32 activation; llama.cpp ".to_string()
            + "rounds the activation to BF16 inside the BF16 matmul (measured 8.1e-3 logit "
            + "error in strata, docs/specs/qwen4exp-architecture.md §8).",
        "Quantised projections (attn_qkv, ssm_out, attn_q/k/v/output) use an f32 activation; "
            .to_string()
            + "the production kernels use Q8_K/Q8_0 (0.66-1.41% measured error, §14).",
        "The PLE device block runs on the host (no kernel yet); ple_key/ple_value stay on the GPU."
            .to_string(),
        "KV cache is f16; the production service uses q4_0 K/V (-ctk q4_0 -ctv q4_0).".to_string(),
        "No batched prefill: the prompt is fed one token at a time, so prefill tok/s is not "
            .to_string()
            + "the prefill number (decode tok/s is unaffected).",
    ];
    let report = breccia_engine::TokenReport {
        schema_version: breccia_engine::TOKEN_SCHEMA_VERSION,
        model: glob.clone(),
        architecture: model.cfg.arch.clone(),
        blocks: model.cfg.n_layers,
        gdn_blocks: model.cfg.layer_counts().0,
        qsa_blocks: model.cfg.layer_counts().1,
        vocab: model.cfg.vocab,
        devices: model
            .stats
            .devices
            .iter()
            .map(|d| format!("{} [{}]", d.name, d.pci))
            .collect(),
        load_plan: model.stats.plan.clone(),
        load_seconds: model.stats.open_seconds,
        upload_seconds: model.stats.upload_seconds,
        dense_bytes: model.stats.dense_bytes,
        head_bytes: model.stats.head_bytes,
        session_bytes: sess.session_bytes(),
        rss_bytes: model.stats.rss_bytes,
        rss_peak_bytes: model.stats.rss_peak_bytes,
        expert_bytes_mapped: model.stats.expert_bytes_mapped,
        ple_bytes_mapped: model.stats.ple_bytes_mapped,
        temperature: a.temperature,
        threads: sess.threads(),
        ctx_capacity: a.ctx,
        runs,
        tokens_per_s: tps,
        ms_per_token: mspt,
        stages,
        first_token: first_summary,
        hotspots,
        debug_layers: traces,
        notes,
    };
    println!("\nhotspots  {}", report.hotspots.join(" · "));
    let _ = last_logits;
    let _ = max_delta;

    if let Some(path) = &a.json {
        let text = serde_json::to_string_pretty(&report).context("serializing the report")?;
        if let Some(dir) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        std::fs::write(path, text + "\n").with_context(|| format!("writing {}", path.display()))?;
        println!("wrote {}", path.display());
    }
    Ok(())
}
