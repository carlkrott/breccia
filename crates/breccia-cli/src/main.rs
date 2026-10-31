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
        /// One-time per-GPU prefill discovery: A/B the merged-prefill flags
        /// (`BRECCIA_PF_*`) on each discrete GPU and store the verdict the engine
        /// uses as their default. Re-run after a driver or shader change.
        #[arg(long)]
        prefill: bool,
        /// Model for `--prefill` (shard glob or path)
        #[arg(long, value_name = "GLOB", default_value = DEFAULT_MODEL_GLOB)]
        model: String,
        /// PCI address(es) for `--prefill` (default: every discrete GPU in the profile)
        #[arg(long, value_name = "PCI")]
        devices: Vec<String>,
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
    /// Compression fabric F2/F3: GPU rANS decode rate and delivered GB/s through the ring
    Codec {
        /// .brcf frame files from docs/coord/codecframes.py (repeatable)
        #[arg(long = "frames", value_name = "PATH", required = true)]
        frames: Vec<PathBuf>,
        /// Device by cached-profile index or PCI address (repeatable)
        #[arg(long = "device", value_name = "PROFILE_INDEX|PCI")]
        device: Vec<String>,
        /// Write the full JSON report here
        #[arg(long, value_name = "PATH")]
        json: Option<PathBuf>,
        /// Short benchmark windows (noisier numbers)
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
    /// Phase 4: N concurrent agent-style streams (different prompts, long
    /// prefixes) — aggregate/per-agent rate, token latency p50/p95, per-card and
    /// host-lane utilisation
    Agents {
        /// Concurrent agents (one sequence each; they cycle over `--devices`)
        #[arg(long, default_value_t = 4)]
        streams: usize,
        /// Tokens to generate per agent (greedy)
        #[arg(long, default_value_t = 32)]
        gen: usize,
        /// Prompt length for the synthetic agent prefixes
        #[arg(long, default_value_t = 300)]
        prefix: usize,
        /// Device plan, comma-separated PCI addresses
        #[arg(
            long,
            value_name = "PCI[,PCI]",
            default_value = "0000:03:00.0,0000:0e:00.0"
        )]
        devices: String,
        /// Host expert-lane threads (shared by every agent)
        #[arg(long)]
        threads: Option<usize>,
        /// Write the JSON report here
        #[arg(long, value_name = "PATH")]
        json: Option<PathBuf>,
        /// Record a block's readback inside the block's own batch
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        inbatch_read: bool,
        /// Prefill chunk: feed the prompt `N` tokens at a time with the grouped
        /// expert lane. `0` = token by token. Absent: with `--prefill-merged`, each
        /// card's discovered chunk (`breccia profile --prefill`), else 0
        #[arg(long)]
        prefill_chunk: Option<usize>,
        /// Merged prefill (§8v): route a chunk once and run **one multi-user GEMM
        /// per distinct expert** over the resident (`--gpu-experts`) arena instead
        /// of one GEMV per token
        #[arg(long, default_value_t = false)]
        prefill_merged: bool,
        /// Extra complete generations before the measured window, so the page
        /// cache holds this run's experts. `0` = the cold number; `1` is the
        /// steady-state one (the product-relevant figure for agents). The storage
        /// bytes re-read are printed on the `cache` line either way.
        #[arg(long, default_value_t = 0)]
        warm: usize,
        /// Blocks whose routed experts live in VRAM, per stream (`;`-separated)
        #[arg(long, value_name = "N[,N]")]
        gpu_experts: Option<String>,
        /// Give every agent the reference prompt: the parity run (all agents must
        /// produce the server's tokens bit-identically) instead of the mixed
        /// workload table.
        #[arg(long)]
        same_prompt: bool,
        /// `--gpu-experts auto`: bytes kept free on a device that currently drives
        /// a display (read from sysfs, not assumed)
        #[arg(long, default_value_t = 768, value_name = "MIB")]
        arena_display_reserve_mib: u64,
        /// `--gpu-experts auto`: bytes kept free on a headless device (session
        /// scratch ~120 MiB + allocator slack)
        #[arg(long, default_value_t = 256, value_name = "MIB")]
        arena_headroom_mib: u64,
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
        /// Record each GPU phase into one command buffer and submit it once.
        /// On by default; `--batching false` is the reference path (and the A/B)
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        batching: bool,
        /// Blocks per device, comma separated (`--split-blocks 21,27`). Default:
        /// balance the dense weight bytes across the devices.
        #[arg(long, value_name = "N[,N]")]
        split_blocks: Option<String>,
        /// Blocks whose routed experts live in VRAM (`-ot`-equivalent), each on
        /// the device that owns the block. `auto` sizes the arena from measured
        /// free VRAM instead (a display reserve on a card that drives an output,
        /// a smaller headroom elsewhere). Production mapping:
        /// `0,1,2,4,8,18,20,22,23,24,35`
        #[arg(long, value_name = "auto|N[,N]")]
        gpu_experts: Option<String>,
        /// Where the LM head + final mixer live: `auto` (default — the hardware
        /// profile's compute-role device, else the device that owns block 47), an
        /// **ordinal into `--devices`** (`0` = the *first* `--devices` entry, so
        /// `--head-device 0` with `--devices 0e,03` is the MI50), or a **PCI
        /// address** (`0000:03:00.0`), which cannot be misread.
        #[arg(long, value_name = "auto|N|PCI")]
        head_device: Option<String>,
        /// `--gpu-experts auto`: bytes kept free on a device that currently drives
        /// a display (read from sysfs, not assumed)
        #[arg(long, default_value_t = 768, value_name = "MIB")]
        arena_display_reserve_mib: u64,
        /// `--gpu-experts auto`: bytes kept free on a headless device (session
        /// scratch ~120 MiB + allocator slack)
        #[arg(long, default_value_t = 256, value_name = "MIB")]
        arena_headroom_mib: u64,
        /// Independent sequences decoded in lockstep, one per device: `2` puts
        /// one sequence on each card and interleaves them at the block level so
        /// each expert lane runs under the other stream's GPU phase. `1` is the
        /// single-stream graph; more streams than devices puts two on one card.
        #[arg(long, default_value_t = 1)]
        streams: usize,
        /// Record a block's readback as copies inside the block's own batch and
        /// read the staging mapping, instead of a submission of its own.
        /// **On by default**: in a two-stream pair the GPU has slack, and the
        /// removed round trip is worth 4-7 % of the aggregate (§8p).
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        inbatch_read: bool,
        /// Prefill chunk: feed the prompt `N` tokens at a time with the grouped
        /// expert lane (§8s). `0` = token by token. Absent: with `--prefill-merged`,
        /// each card's discovered chunk (`breccia profile --prefill`), else 0
        #[arg(long)]
        prefill_chunk: Option<usize>,
        /// Merged prefill (§8v): one multi-user GEMM per distinct resident expert
        #[arg(long, default_value_t = false)]
        prefill_merged: bool,
        /// Extra complete generations before the measured window (`0` = cold)
        #[arg(long, default_value_t = 0)]
        warm: usize,
        /// A **second prompt** generated in the *same* process after the measured
        /// runs, e.g. `--prompt-ids <long> --then-prompt-ids <short>`. This is the
        /// session-reuse case (`scripts/session-reset-test.sh`): its tokens must be
        /// identical to a fresh process's, which is what makes a missing state reset
        /// a deterministic failure instead of an intermittent one.
        #[arg(long, value_name = "ID,ID,...")]
        then_prompt_ids: Option<String>,
        /// D4: import host-lane expert bytes zero-copy via
        /// `VK_EXT_external_memory_host` (an mlock'd host arena; needs
        /// `LimitMEMLOCK=infinity` for the streaming set). Default OFF.
        #[arg(long)]
        host_expert_import: bool,
        /// R29 card 1: after the arena upload, drop the uploaded source pages
        /// (fadvise DONTNEED) and WILLNEED the streamed/host-lane expert
        /// ranges. Default OFF.
        #[arg(long)]
        page_hint_trim: bool,
    },
}
fn main() {
    let cli = Cli::parse();
    // §12 defaults: cached host staging (§11f) everywhere. Set before any thread starts.
    if std::env::var_os("BRECCIA_STAGING_CACHED").is_none() {
        std::env::set_var("BRECCIA_STAGING_CACHED", "1");
    }
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
            prefill,
            model,
            devices,
        } => profile(ProfileArgs {
            reprofile,
            quick,
            json,
            dry_run_plan,
            print_fingerprint,
            prefill,
            model,
            devices,
        }),
        Command::Serve { dry_run_plan, args } => serve(dry_run_plan, args),
        Command::Bench { kind: None } => {
            println!(
                "bench: available harnesses\n  \
                 breccia bench transfer [--device <idx|pci>]... [--json PATH] [--quick]\n  \
                 breccia bench codec --frames F.brcf... [--device <idx|pci>]... [--json PATH] [--quick]\n  \
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
                Some(BenchKind::Codec {
                    frames,
                    device,
                    json,
                    quick,
                }),
        } => bench_codec(frames, device, json, quick),
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
                Some(BenchKind::Agents {
                    streams,
                    gen,
                    prefix,
                    devices,
                    threads,
                    json,
                    inbatch_read,
                    gpu_experts,
                    same_prompt,
                    prefill_chunk,
                    prefill_merged,
                    warm,
                    arena_display_reserve_mib,
                    arena_headroom_mib,
                }),
        } => {
            let devs: Vec<String> = devices
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            let (prefill_chunk, prefill_chunks) =
                resolve_prefill_chunks(prefill_chunk, prefill_merged, &devs, 0);
            bench_agents(AgentArgs {
            streams,
            gen,
            prefix,
            devices,
            threads,
            json,
            inbatch_read,
            gpu_experts,
            same_prompt,
            prefill_chunk,
            prefill_chunks,
            prefill_merged,
            warm,
            arena_display_reserve_mib,
            arena_headroom_mib,
        })
        }
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
                    split_blocks,
                    gpu_experts,
                    head_device,
                    arena_display_reserve_mib,
                    arena_headroom_mib,
                    streams,
                    inbatch_read,
                    prefill_chunk,
                    prefill_merged,
                    warm,
                    then_prompt_ids,
                    host_expert_import,
                    page_hint_trim,
                }),
        } => {
            let (prefill_chunk, prefill_chunks) =
                resolve_prefill_chunks(prefill_chunk, prefill_merged, &devices, ctx);
            bench_token(TokenArgs {
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
            split_blocks,
            gpu_experts,
            head_device,
            arena_display_reserve_mib,
            arena_headroom_mib,
            streams,
            inbatch_read,
            prefill_chunk,
            prefill_chunks,
            prefill_merged,
            warm,
            then_prompt_ids,
            host_expert_import,
            page_hint_trim,
        })
        }
    }
}

struct AgentArgs {
    streams: usize,
    gen: usize,
    prefix: usize,
    devices: String,
    threads: Option<usize>,
    json: Option<PathBuf>,
    inbatch_read: bool,
    gpu_experts: Option<String>,
    same_prompt: bool,
    /// The smallest of `prefill_chunks` (what a multi-device model runs at).
    prefill_chunk: usize,
    /// Per `devices` entry.
    prefill_chunks: Vec<usize>,
    prefill_merged: bool,
    warm: usize,
    arena_display_reserve_mib: u64,
    arena_headroom_mib: u64,
}

struct ProfileArgs {
    reprofile: bool,
    quick: bool,
    json: Option<PathBuf>,
    dry_run_plan: bool,
    print_fingerprint: bool,
    prefill: bool,
    model: String,
    devices: Vec<String>,
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
                    print!("{}", breccia_profile::plan::plan_text(&cached, None));
                }
                println!(
                    "cached profile reused (fingerprint match): {}",
                    path.display()
                );
                if a.prefill && !a.dry_run_plan {
                    return profile_prefill(&path, &a.model, &a.devices);
                }
                return Ok(());
            }
        }
    }

    // Calibrate.
    println!(
        "calibrating hardware profile ({} mode)…",
        if a.quick { "quick" } else { "full" }
    );
    let mut profile =
        breccia_profile::calibrate(a.quick).context("calibration failed (Vulkan benchmarks)")?;
    // A recalibration keeps each GPU's prefill verdict; its own staleness check
    // (driver + shader hash) decides whether it still applies.
    if let Ok(Some(old)) = breccia_profile::load_profile(&path) {
        for e in &mut profile.devices {
            e.prefill = old
                .devices
                .iter()
                .find(|o| same_pci(&o.identity.pci_address, &e.identity.pci_address))
                .and_then(|o| o.prefill.clone());
        }
    }

    if a.dry_run_plan {
        print!("{}", breccia_profile::plan::plan_text(&profile, None));
        println!("dry run: nothing written (would write {})", path.display());
        return Ok(());
    }

    breccia_profile::save_profile(&profile, &path)?;
    print!("{}", breccia_profile::summary_text(&profile));
    println!("wrote {}", path.display());
    if a.prefill {
        return profile_prefill(&path, &a.model, &a.devices);
    }
    // First calibration on a box (a GPU without a verdict): the one-time prefill
    // check runs too, best-effort, so a missing model does not fail the install.
    let unchecked = profile
        .devices
        .iter()
        .any(|e| e.identity.device_type == "DISCRETE_GPU" && e.prefill.is_none());
    if unchecked {
        if let Err(e) = profile_prefill(&path, &a.model, &a.devices) {
            eprintln!(
                "breccia: prefill discovery skipped ({e:#}); run `breccia profile --prefill --model ...`"
            );
        }
    }
    Ok(())
}

fn same_pci(a: &str, b: &str) -> bool {
    !a.is_empty() && breccia_vk::pci::normalize_address(a) == breccia_vk::pci::normalize_address(b)
}

/// `--prefill-chunk` per `devs` entry: an explicit value wins (0 included);
/// absent, `--prefill-merged` uses every card's discovered `best_chunk`, all or
/// nothing (a card without one sends every card back to token by token). A
/// discovered chunk is capped at the largest power of two <= `ctx`: a bigger one
/// only grows the chunk scratch and shrinks the arena (MI50 2048 vs 1024 at ctx
/// 1100: byte-identical logits, prefill 15.0 vs 12.2 s, §11l item 5).
fn prefill_chunks_for(
    flag: Option<usize>,
    merged: bool,
    devs: &[String],
    ctx: usize,
    best_chunk: impl Fn(&str) -> Option<usize>,
) -> (Vec<usize>, &'static str) {
    let n = devs.len().max(1);
    if let Some(c) = flag {
        return (vec![c; n], "--prefill-chunk");
    }
    if !merged {
        return (vec![0; n], "default");
    }
    let found: Option<Vec<usize>> = devs
        .iter()
        .map(|d| {
            best_chunk(d)
                .filter(|&c| c > 0)
                .map(|c| if ctx > 0 { c.min(1 << ctx.ilog2()) } else { c })
        })
        .collect();
    match found {
        Some(v) if !v.is_empty() => (v, "discovered per card"),
        _ => (
            vec![0; n],
            "no discovered chunk for every card: token by token \
             (run `breccia profile --prefill` or pass --prefill-chunk)",
        ),
    }
}

/// [`prefill_chunks_for`] against the saved hardware profile; returns the smallest
/// chunk (a multi-device model's) and the per-card list, and says where they came from.
fn resolve_prefill_chunks(
    flag: Option<usize>,
    merged: bool,
    devs: &[String],
    ctx: usize,
) -> (usize, Vec<usize>) {
    let prof = breccia_profile::load_profile(&breccia_profile::default_profile_path())
        .ok()
        .flatten();
    let (chunks, src) = prefill_chunks_for(flag, merged, devs, ctx, |d| {
        let e = prof.as_ref()?.devices.iter().find(|e| same_pci(&e.identity.pci_address, d))?;
        Some(e.prefill.as_ref()?.best_chunk)
    });
    if flag.is_none() && merged {
        println!("prefill chunk {chunks:?} ({src})");
    }
    (chunks.iter().copied().min().unwrap_or(0), chunks)
}

/// Stream `s`'s chunk, from its card (`s % devs`), and its `auto` arena grown by
/// the scratch every stream on that card allocates after the arenas are chosen.
fn stream_chunk_arena(
    spec: breccia_engine::ArenaSpec,
    chunks: &[usize],
    s: usize,
    per_card: usize,
    ctx: usize,
) -> (usize, breccia_engine::ArenaSpec) {
    let chunk = chunks[s % chunks.len()];
    let spec = match spec {
        breccia_engine::ArenaSpec::Auto(o) => {
            // Every stream on the card allocates its own scratch.
            breccia_engine::ArenaSpec::Auto((0..per_card).fold(o, |o, _| o.with_prefill_chunk(chunk, ctx)))
        }
        other => other,
    };
    (chunk, spec)
}

/// KL(p || q) over two logit vectors (softmax each), in nats.
fn logits_kl(p: &[f32], q: &[f32]) -> f64 {
    let lsm = |x: &[f32]| {
        let m = x.iter().fold(f32::MIN, |a, &b| a.max(b)) as f64;
        let z = x.iter().map(|&v| (v as f64 - m).exp()).sum::<f64>().ln() + m;
        x.iter().map(move |&v| v as f64 - z).collect::<Vec<f64>>()
    };
    let (lp, lq) = (lsm(p), lsm(q));
    lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum()
}

fn argmax(x: &[f32]) -> usize {
    (0..x.len()).fold(0, |b, i| if x[i] > x[b] { i } else { b })
}

/// `breccia profile --prefill`: per discrete GPU, A/B the merged-prefill flags
/// in-process and store a verdict in the profile (`DeviceEntry.prefill`).
///
/// Arms on a 701-token synthetic prompt (one 512 chunk + a ragged 189, so both
/// the full and a ragged T run through every chunked kernel): per-token reference,
/// all flags off, all on, and all on minus each flag. A flag stays on when the
/// all-on arm matches the per-token reference (KL <= 1e-3, or no worse than
/// all-off, and the same top-1) and dropping it does not make prefill faster.
fn profile_prefill(path: &std::path::Path, model_glob: &str, devices: &[String]) -> Result<()> {
    use breccia_engine::graph::PF_FLAGS;
    const CHUNK: usize = 512;
    let mut prof = breccia_profile::load_profile(path)?
        .with_context(|| format!("no hardware profile at {}", path.display()))?;
    let targets: Vec<String> = if devices.is_empty() {
        prof.devices
            .iter()
            .filter(|e| e.identity.device_type == "DISCRETE_GPU" && !e.identity.pci_address.is_empty())
            .map(|e| e.identity.pci_address.clone())
            .collect()
    } else {
        devices.to_vec()
    };
    if targets.is_empty() {
        anyhow::bail!("no discrete GPU in {} (pass --devices PCI)", path.display());
    }
    for addr in targets {
        println!("prefill discovery on {addr} ({model_glob})");
        let model = breccia_engine::Model::load(
            model_glob,
            &breccia_engine::LoadOptions {
                devices: vec![addr.clone()],
                ctx_capacity: 1024,
                profile: Some(path.to_path_buf()),
                // Without an arena the experts run on the CPU lane at per-token
                // speed and no flag shows an effect; block 0 matches `--gpu-experts 0`.
                arena: breccia_engine::ArenaSpec::parse("0").map_err(anyhow::Error::msg)?,
                verbose: false,
                ..Default::default()
            },
        )
        .with_context(|| format!("loading the model on {addr}"))?;
        breccia_engine::graph::assert_head(&model)?;
        let prompt = synthetic_prefix(
            &[760, 13017, 1433, 364, 279, 5447],
            701,
            0x5eed,
            model.cfg.vocab as u32,
        );
        let mut sess = breccia_engine::Session::new(
            &model,
            &breccia_engine::SessionCfg {
                ctx_capacity: 1024,
                threads: 0,
                timers: false,
                batching: true,
                inbatch_read: true,
                prefill_chunk: CHUNK,
                prefill_merged: true,
                probe: false,
                pool: None,
            },
        )?;
        // One arm: (best of two wall times in ms, last-token logits).
        let arm = |sess: &mut breccia_engine::Session<'_>, pf: [bool; 5], chunk: usize, reps: usize| {
            sess.pf[0] = pf;
            let mut best = f64::MAX;
            let mut logits = Vec::new();
            for _ in 0..reps {
                let t = std::time::Instant::now();
                logits = feed_prompt(sess, &prompt, chunk)?;
                best = best.min(t.elapsed().as_secs_f64() * 1e3);
            }
            anyhow::Ok((best, logits))
        };
        let (_, reference) = arm(&mut sess, [false; 5], 0, 1)?;
        arm(&mut sess, [true; 5], CHUNK, 1)?; // warm-up: pipelines, workspaces
        let (off_ms, off) = arm(&mut sess, [false; 5], CHUNK, 2)?;
        let (on_ms, on) = arm(&mut sess, [true; 5], CHUNK, 2)?;
        let top = argmax(&reference);
        let kl_off = logits_kl(&reference, &off);
        let kl_on = logits_kl(&reference, &on);
        let on_ok = argmax(&on) == top && (kl_on <= 1e-3 || kl_on <= kl_off * 1.1);
        println!(
            "  per-token ref top-1 {top}; all-off {off_ms:.0} ms KL {kl_off:.2e}; \
             all-on {on_ms:.0} ms KL {kl_on:.2e} top-1 {}",
            if argmax(&on) == top { "same" } else { "DIFFERENT" }
        );
        let mut flags = Vec::new();
        for (i, name) in PF_FLAGS.iter().enumerate() {
            let mut pf = [true; 5];
            pf[i] = false;
            let (ms, lg) = arm(&mut sess, pf, CHUNK, 2)?;
            let kl = logits_kl(&reference, &lg);
            let (on, reason) = if !on_ok {
                (false, "all-on arm failed parity".to_string())
            } else if on_ms > ms * 1.02 {
                (false, format!("slower: {on_ms:.0} ms with vs {ms:.0} ms without"))
            } else {
                (true, format!("{on_ms:.0} ms with vs {ms:.0} ms without"))
            };
            println!("  {name:<20} {} ({reason}; KL without {kl:.2e})", if on { "on " } else { "OFF" });
            flags.push(breccia_profile::schema::PrefillFlag {
                name: name.to_string(),
                on,
                kl,
                top1_same: argmax(&lg) == top,
                ms_without: ms / prompt.len() as f64,
                reason,
            });
        }
        let verdict = breccia_profile::schema::PrefillVerdict {
            driver_version: model.devices[0].info.driver.clone(),
            spirv_hash: breccia_kernels::SPIRV_HASH.to_string(),
            timestamp: breccia_profile::schema::iso8601(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_secs()),
            ),
            model: model_glob.to_string(),
            tokens: prompt.len(),
            chunk: CHUNK,
            off_ms: off_ms / prompt.len() as f64,
            on_ms: on_ms / prompt.len() as f64,
            flags,
            best_chunk: 0,
        };
        drop(sess);
        drop(model);
        let entry = prof
            .devices
            .iter_mut()
            .find(|e| same_pci(&e.identity.pci_address, &addr))
            .with_context(|| format!("{addr} is not in {}", path.display()))?;
        entry.prefill = Some(verdict);
        // Saved per device, so a crash on the second GPU keeps the first verdict.
        breccia_profile::save_profile(&prof, path)?;
        println!("  verdict stored in {}", path.display());
        // The sweep loads the model with the flags just stored.
        let best = profile_prefill_chunk(path, model_glob, &addr)?;
        if let Some(v) = prof
            .devices
            .iter_mut()
            .find(|e| same_pci(&e.identity.pci_address, &addr))
            .and_then(|e| e.prefill.as_mut())
        {
            v.best_chunk = best;
        }
        breccia_profile::save_profile(&prof, path)?;
        println!("  best chunk {best} stored");
    }
    Ok(())
}

/// The prefill chunk one card runs fastest at. Each candidate gets a fresh model
/// and session (workspaces are sized at `Session::new`) under the `auto` arena
/// with that chunk's scratch reserve, so a bigger chunk also pays for the expert
/// blocks it pushes out of VRAM. A candidate must match the 512 arm (KL <= 1e-3,
/// same top-1) and beat the smaller winner by more than 2 % to replace it.
fn profile_prefill_chunk(path: &std::path::Path, model_glob: &str, addr: &str) -> Result<usize> {
    const CANDIDATES: [usize; 3] = [512, 1024, 2048];
    // Planner splits evenly up to 1.5x the target: 3x700 / 2x1050 / 1x2100.
    const TOKENS: usize = 2100;
    let mut base: Option<Vec<f32>> = None;
    let mut best = (0usize, f64::MAX);
    for chunk in CANDIDATES {
        let model = breccia_engine::Model::load(
            model_glob,
            &breccia_engine::LoadOptions {
                devices: vec![addr.to_string()],
                ctx_capacity: TOKENS + 64,
                profile: Some(path.to_path_buf()),
                // A display card keeps 2.5 GiB free (desktop + other tenants).
                arena: breccia_engine::ArenaSpec::Auto(
                    breccia_engine::ArenaOpts {
                        display_reserve: 2560 << 20,
                        ..Default::default()
                    }
                    .with_prefill_chunk(chunk, TOKENS + 64),
                ),
                verbose: false,
                ..Default::default()
            },
        )
        .with_context(|| format!("loading the model on {addr} for chunk {chunk}"))?;
        let prompt = synthetic_prefix(
            &[760, 13017, 1433, 364, 279, 5447],
            TOKENS,
            0x5eed,
            model.cfg.vocab as u32,
        );
        let mut sess = breccia_engine::Session::new(
            &model,
            &breccia_engine::SessionCfg {
                ctx_capacity: TOKENS + 64,
                threads: 0,
                timers: false,
                batching: true,
                inbatch_read: true,
                prefill_chunk: chunk,
                prefill_merged: true,
                probe: false,
                pool: None,
            },
        )?;
        feed_prompt(&mut sess, &prompt, chunk)?; // warm-up: pipelines, workspaces
        let mut ms = f64::MAX;
        let mut logits = Vec::new();
        for _ in 0..2 {
            let t = std::time::Instant::now();
            logits = feed_prompt(&mut sess, &prompt, chunk)?;
            ms = ms.min(t.elapsed().as_secs_f64() * 1e3 / prompt.len() as f64);
        }
        let (kl, same) = match &base {
            Some(b) => (logits_kl(b, &logits), argmax(b) == argmax(&logits)),
            None => (0.0, true),
        };
        let ok = kl <= 1e-3 && same;
        let wins = ok && ms < best.1 * 0.98;
        println!(
            "  chunk {chunk:>4}: {ms:.2} ms/token, KL vs 512 {kl:.2e}, top-1 {}{}",
            if same { "same" } else { "DIFFERENT" },
            if wins { " <- best so far" } else { "" }
        );
        if base.is_none() {
            base = Some(logits);
        }
        if wins {
            best = (chunk, ms);
        }
    }
    Ok(best.0)
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

fn bench_codec(
    frames: Vec<PathBuf>,
    specs: Vec<String>,
    json: Option<PathBuf>,
    quick: bool,
) -> Result<()> {
    let profile = breccia_profile::load_profile(&breccia_profile::default_profile_path())
        .ok()
        .flatten();
    let mut devices = Vec::new();
    for spec in &specs {
        devices.push(selector_for(spec, profile.as_ref())?);
    }
    let report = breccia_vk::run_codec_bench(&breccia_vk::CodecBenchOptions {
        quick,
        devices,
        frames,
    })
    .context("codec benchmark failed")?;
    if let Some(path) = &json {
        let text = serde_json::to_string_pretty(&report).context("serializing codec report")?;
        if let Some(dir) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        std::fs::write(path, text + "\n").with_context(|| format!("writing {}", path.display()))?;
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

/// Whether a GPU timestamp probe is on. `Session::gpu_busy_s` returns 0 when it
/// is not, which the multi-stream reports used to print as "0% utilised".
fn gpu_busy_measured() -> bool {
    std::env::var_os("BRECCIA_GPU_TIME").is_some() || breccia_kernels::batch_cmd::probe::enabled()
}

const NOT_MEASURED: &str = "not measured (set BRECCIA_GPU_TIME=1)";

fn busy_summary(measured: bool, busy_s: f64, wall_s: f64) -> String {
    if !measured {
        return NOT_MEASURED.into();
    }
    format!(
        "{:.0} ms of {:.0} ms = {:.0}% utilised",
        busy_s * 1e3,
        wall_s * 1e3,
        busy_s / wall_s.max(1e-9) * 100.0
    )
}

fn busy_per_step(measured: bool, busy_s: f64, steps: u64) -> String {
    if !measured {
        return NOT_MEASURED.into();
    }
    format!("{:.1} ms/step", busy_s * 1e3 / steps.max(1) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_chunk_flag_wins_then_discovered_all_or_nothing() {
        let devs = vec!["0000:03:00.0".to_string(), "0000:0e:00.0".to_string()];
        let both = |d: &str| Some(if d.ends_with("03:00.0") { 2048 } else { 1024 });
        let one = |d: &str| d.ends_with("03:00.0").then_some(2048);
        assert_eq!(prefill_chunks_for(Some(0), true, &devs, 0, both).0, vec![0, 0]);
        assert_eq!(prefill_chunks_for(Some(512), false, &devs, 0, both).0, vec![512, 512]);
        assert_eq!(prefill_chunks_for(None, false, &devs, 0, both).0, vec![0, 0]);
        assert_eq!(prefill_chunks_for(None, true, &devs, 0, both).0, vec![2048, 1024]);
        assert_eq!(prefill_chunks_for(None, true, &devs, 0, one).0, vec![0, 0]);
        assert_eq!(prefill_chunks_for(None, true, &devs, 0, |_| Some(0)).0, vec![0, 0]);
        assert_eq!(prefill_chunks_for(None, true, &[], 0, both).0, vec![0]);
        // ctx caps a discovered chunk (largest power of two <= ctx), never the flag.
        assert_eq!(prefill_chunks_for(None, true, &devs, 1100, both).0, vec![1024, 1024]);
        assert_eq!(prefill_chunks_for(None, true, &devs, 600, both).0, vec![512, 512]);
        assert_eq!(prefill_chunks_for(None, true, &devs, 4200, both).0, vec![2048, 1024]);
        assert_eq!(prefill_chunks_for(Some(2048), true, &devs, 1100, both).0, vec![2048, 2048]);
    }

    #[test]
    fn unmeasured_busy_never_prints_a_percentage() {
        assert!(!busy_summary(false, 0.0, 2.0).contains('%'));
        assert!(!busy_per_step(false, 0.0, 8).contains("ms/step"));
        assert!(busy_summary(true, 1.0, 2.0).contains("50% utilised"));
        assert_eq!(busy_per_step(true, 0.8, 8), "100.0 ms/step");
    }

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

    /// D4: the flag is parsed off by default and carried into the engine's
    /// `LoadOptions` field by `bench token` (`host_expert_import: a.host_expert_import`).
    #[test]
    fn host_expert_import_flag_parses_off_by_default() {
        let parse = |extra: &[&str]| -> bool {
            let mut argv = vec!["breccia", "bench", "token", "--prompt-ids", "1"];
            argv.extend_from_slice(extra);
            match Cli::try_parse_from(argv).expect("cli parses").command {
                Command::Bench {
                    kind:
                        Some(BenchKind::Token {
                            host_expert_import, ..
                        }),
                } => host_expert_import,
                _ => panic!("expected bench token"),
            }
        };
        assert!(!parse(&[]));
        assert!(parse(&["--host-expert-import"]));
    }

    /// Defect: `--gpu-experts auto` was dropped on the multi path (every stream got
    /// an empty list). Every multi group must resolve through `ArenaSpec::parse`,
    /// and a bad token or empty group must error, never become an arena-off stream.
    #[test]
    fn multi_arena_specs_are_strict() {
        let opts = breccia_engine::ArenaOpts::default();
        assert_eq!(
            multi_arena_specs("auto", 2, opts).unwrap(),
            vec![breccia_engine::ArenaSpec::Auto(opts)]
        );
        assert_eq!(
            multi_arena_specs("0,1;2,3", 2, opts).unwrap(),
            vec![
                breccia_engine::ArenaSpec::List(vec![0, 1]),
                breccia_engine::ArenaSpec::List(vec![2, 3]),
            ]
        );
        for bad in ["0,1;foo", "0,1;", "auto;0,1", "0,x,2", "1,x"] {
            assert!(multi_arena_specs(bad, 2, opts).is_err(), "{bad} must error");
        }
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
    split_blocks: Option<String>,
    gpu_experts: Option<String>,
    head_device: Option<String>,
    arena_display_reserve_mib: u64,
    arena_headroom_mib: u64,
    streams: usize,
    inbatch_read: bool,
    /// The smallest of `prefill_chunks` (what a multi-device model runs at).
    prefill_chunk: usize,
    /// Per `devices` entry.
    prefill_chunks: Vec<usize>,
    prefill_merged: bool,
    warm: usize,
    then_prompt_ids: Option<String>,
    /// D4: import host-lane expert bytes via VK_EXT_external_memory_host
    host_expert_import: bool,
    /// R29 card 1: page-cache hints around the expert upload
    page_hint_trim: bool,
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
fn feed_prompt(
    sess: &mut breccia_engine::Session<'_>,
    prompt: &[u32],
    chunk: usize,
) -> Result<Vec<f32>> {
    sess.reset()?;
    let mut logits = Vec::new();
    if chunk > 0 {
        let mut pos = 0usize;
        for len in breccia_engine::prefill_plan(prompt.len(), chunk) {
            logits = sess.prefill(&prompt[pos..pos + len], pos)?;
            pos += len;
        }
        return Ok(logits);
    }
    for (pos, &t) in prompt.iter().enumerate() {
        logits = sess.token(t, pos, None)?;
    }
    Ok(logits)
}

/// `BRECCIA_MTP_SPEC_BENCH=1,2,3,gate`: for each arm, ABAB against plain decode
/// (re-prefilling each time), greedy parity compared in-process. `BRECCIA_MTP_GEN`
/// tokens (default 64), `BRECCIA_MTP_REPS` pairs per arm (default 2).
fn mtp_spec_bench(
    model: &breccia_engine::model::Model,
    sess: &mut breccia_engine::Session<'_>,
    prompt: &[u32],
    chunk: usize,
    arms: &str,
) -> Result<()> {
    use breccia_engine::mtp_spec::{generate, Policy};
    let env_n = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let (gen, reps) = (env_n("BRECCIA_MTP_GEN", 64), env_n("BRECCIA_MTP_REPS", 2));
    let dir = std::env::var("BRECCIA_MTP_DIR").unwrap_or_else(|_| "/srv/strata-data/mtp/rt".into());
    sess.spec_enable(4).context("spec_enable(4)")?;
    let mut d = breccia_engine::mtp_draft::MtpDraft::open(model, 0, &dir, prompt.len() + gen + 16)?;
    println!("mtp-spec  draft {} MiB device, prompt {} gen {gen} reps {reps}", d.bytes >> 20, prompt.len());
    for arm in arms.split(',') {
        let pol = Policy::parse(arm).with_context(|| format!("BRECCIA_MTP_SPEC_BENCH arm {arm:?}"))?;
        for rep in 0..reps {
            let (a, sa) = generate(sess, None, prompt, chunk, gen, Policy::Plain)?;
            let (b, sb) = generate(sess, Some(&mut d), prompt, chunk, gen, pol)?;
            let (ta, tb) = (a.len() as f64 / sa.decode_s, b.len() as f64 / sb.decode_s);
            println!(
                "mtp-spec  arm {arm} rep {rep}: plain {ta:.2} tok/s | spec {tb:.2} tok/s ({:.3}x) | greedy {} ({} tokens)",
                tb / ta,
                if a == b { "IDENTICAL".to_string() } else {
                    format!("DIFFER at {:?}", a.iter().zip(&b).position(|(x, y)| x != y))
                },
                b.len()
            );
            println!("mtp-spec  arm {arm} rep {rep}: {sb}");
        }
    }
    Ok(())
}

/// `BRECCIA_MTP_DRAFT_BENCH=N`: time N draft steps (synthetic R, real MTP
/// weights, zeroed draft KV) on device 0. `BRECCIA_MTP_DIR` overrides the weights dir.
fn mtp_draft_bench(model: &breccia_engine::model::Model, prompt_len: usize, n: usize) -> Result<()> {
    let dir = std::env::var("BRECCIA_MTP_DIR").unwrap_or_else(|_| "/srv/strata-data/mtp/rt".into());
    let t0 = std::time::Instant::now();
    let mut d = breccia_engine::mtp_draft::MtpDraft::open(model, 0, &dir, prompt_len + 64)?;
    println!("mtp-draft  built in {:.1} s, {} MiB device", t0.elapsed().as_secs_f64(), d.bytes >> 20);
    let r: Vec<f32> = (0..10240).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.02).collect();
    d.set_r(&r)?;
    let mut rows: Vec<[f64; 4]> = Vec::new();
    let mut tok = 1000u32;
    for i in 0..n {
        let (t, ms) = d.step(tok, prompt_len + i % 32)?;
        println!("mtp-draft  step {i:2}: tok {t:6} gather {:.2} record {:.2} gpu {:.2} read {:.2} ms", ms[0], ms[1], ms[2], ms[3]);
        tok = t;
        rows.push(ms);
    }
    let med = |c: usize| {
        let mut v: Vec<f64> = rows.iter().skip(2).map(|r| r[c]).collect();
        v.sort_by(|a, b| a.total_cmp(b));
        v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
    };
    println!(
        "mtp-draft  median (skip 2 warmup): gather {:.2} record {:.2} gpu {:.2} read {:.2}  step(no gather) {:.2} ms",
        med(0), med(1), med(2), med(3), med(1) + med(2) + med(3)
    );
    Ok(())
}

/// `BRECCIA_SPEC_VERIFY_BENCH=T` (needs `--batching`, `--prefill-chunk` with
/// `prefill_cap >= T`, one device): time one decode step against one T-row
/// verify at the same position, check the verify rows against the decode steps,
/// and check that `rollback_to(row)` leaves the state a plain decode would have.
/// Runs on the state `feed_prompt` left; `first` is the prompt's last logits.
fn spec_verify_bench(
    sess: &mut breccia_engine::Session<'_>,
    first: &[f32],
    prompt_len: usize,
    t: usize,
) -> Result<()> {
    const N: usize = 16; // decode steps timed; N >= T + 9 covers the rollback check
    const REPS: usize = 5;
    let median = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    let p = prompt_len;
    sess.spec_enable(t)?;
    sess.spec_save_base()?;
    // Plain greedy: g[i + 1] follows g[i]; d[i] is the logits after feeding g[i].
    let mut g = vec![argmax(first) as u32];
    let (mut dec_ms, mut d) = (Vec::new(), Vec::new());
    for i in 0..N {
        let t0 = std::time::Instant::now();
        let l = sess.token(g[i], p + i, None)?;
        dec_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        g.push(argmax(&l) as u32);
        if i < t {
            d.push(l);
        }
    }
    let decode_ms = median(&mut dec_ms);

    let (mut ver_ms, mut rows) = (Vec::new(), Vec::new());
    for rep in 0..=REPS {
        sess.spec_restore_base()?;
        let t0 = std::time::Instant::now();
        rows = sess.verify(&g[..t], p)?;
        if rep > 0 {
            ver_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        }
    }
    let verify_ms = median(&mut ver_ms);
    let top1 = (0..t).filter(|&i| argmax(&rows[i]) as u32 == g[i + 1]).count();
    let max_diff = (0..t)
        .flat_map(|i| rows[i].iter().zip(&d[i]).map(|(a, b)| (a - b).abs()))
        .fold(0f32, f32::max);
    println!("spec-verify rows vs decode steps: max |logit diff| {max_diff:.3e}");

    // Accept `row` drafts, then decode on: must match plain greedy from there.
    let mut pass = true;
    for row in 0..t {
        sess.spec_restore_base()?;
        sess.verify(&g[..t], p)?;
        sess.rollback_to(row)?;
        let mut cur = g[row + 1];
        let mut got = Vec::new();
        for j in 0..8 {
            let l = sess.token(cur, p + row + 1 + j, None)?;
            cur = argmax(&l) as u32;
            got.push(cur);
        }
        let ok = got == g[row + 2..row + 10];
        // Early divergence = bad snapshot; late = decode-vs-prefill numerics.
        let first_bad = got.iter().zip(&g[row + 2..]).position(|(a, b)| a != b);
        println!(
            "spec-verify rollback_to({row}) then 8 greedy: {} (first divergence {first_bad:?})",
            if ok { "match" } else { "MISMATCH" }
        );
        pass &= ok;
    }
    sess.spec_restore_base()?;
    println!(
        "spec-verify T={t} decode_ms={decode_ms:.2} verify_ms={verify_ms:.2} ratio={:.2} top1={top1}/{t} rollback={}",
        verify_ms / decode_ms,
        if pass { "PASS" } else { "FAIL" }
    );
    Ok(())
}

/// Two (or more) sequences, one per device, interleaved at the block level.
///
/// The number that matters is the **aggregate** token rate: each stream's own
/// latency is the block-pair time and does not improve, but every expert lane now
/// runs while the other stream's GPU phase is executing
/// (`docs/phase-4/01-forward-pass.md` §8o).
/// `--gpu-experts` for the multi-stream path: `;` separates streams, `,` lists
/// blocks, and `auto` alone gives every stream the measured-free-VRAM rule.
/// Every group goes through `ArenaSpec::parse`; a bad token or an empty group is
/// an error, never a silently-empty (arena-off) stream (defect r367-plateau).
fn multi_arena_specs(
    spec: &str,
    n_streams: usize,
    opts: breccia_engine::ArenaOpts,
) -> anyhow::Result<Vec<breccia_engine::ArenaSpec>> {
    use breccia_engine::ArenaSpec;
    let groups: Vec<&str> = spec.split(';').map(str::trim).collect();
    if groups.iter().all(|g| g.eq_ignore_ascii_case("auto")) {
        // shortcut: streams size their arena in load order against the free VRAM left at their turn, so two streams on one card share by load order, not an even split; upgrade if a dual-card receipt shows imbalance.
        let _ = n_streams;
        return Ok(vec![ArenaSpec::Auto(opts)]);
    }
    let mut out = Vec::with_capacity(groups.len());
    for g in &groups {
        if g.is_empty() {
            anyhow::bail!("--gpu-experts {spec:?}: empty stream group");
        }
        if g.eq_ignore_ascii_case("auto") {
            anyhow::bail!("--gpu-experts {spec:?}: 'auto' cannot be mixed with block lists");
        }
        for t in g.split(',') {
            if t.trim().parse::<usize>().is_err() {
                anyhow::bail!(
                    "--gpu-experts {spec:?}: {:?} is not a block number",
                    t.trim()
                );
            }
        }
        out.push(ArenaSpec::parse(g).map_err(anyhow::Error::msg)?);
    }
    Ok(out)
}

fn bench_token_multi(a: &TokenArgs, prompt: &[u32], glob: &str) -> Result<()> {
    let n_streams = a.streams;
    // §12 defaults for multi-stream: concurrent prompt passes (§11j) and the first
    // card as routed-expert helper (§11j item 2, §12.4). Env set to anything wins.
    for k in ["BRECCIA_MULTI_PREFILL_CONCURRENT", "BRECCIA_PF_EXPERT_DEVICE"] {
        if std::env::var_os(k).is_none() {
            std::env::set_var(k, "1");
        }
    }
    // §12.13: the micro-batch pipeline lost on streams-2 (the RX serves its own lane
    // too: 6.11 / 6.58 s vs 6.53 / 6.54 s), so multi-stream keeps it off unless set.
    if std::env::var_os("BRECCIA_PF_MICRO").is_none() {
        std::env::set_var("BRECCIA_PF_MICRO", "0");
    }
    if prompt.len() + a.gen + 1 > a.ctx {
        anyhow::bail!(
            "prompt {} + gen {} + 1 exceeds --ctx {}",
            prompt.len(),
            a.gen,
            a.ctx
        );
    }
    let devs: Vec<String> = a.devices.clone();
    if devs.is_empty() {
        anyhow::bail!("--streams {n_streams} needs at least one --devices entry");
    }
    // `--gpu-experts` is a per-stream list; `;` separates streams (the last group
    // repeats). Two streams on the same card are allowed: their GPU phases then
    // queue behind each other, which is what the scaling curve measures. `auto`
    // asks for the measured-free-VRAM rule; a non-numeric group is an error
    // (hard-parse like the single path), and a numeric spelling that survives
    // only because `filter_map` dropped its tokens never silently becomes "off".
    let groups: Vec<breccia_engine::ArenaSpec> = match a.gpu_experts.as_deref() {
        Some(spec) => {
            let opts = breccia_engine::ArenaOpts {
                display_reserve: a.arena_display_reserve_mib << 20,
                headroom: a.arena_headroom_mib << 20,
            };
            multi_arena_specs(spec, n_streams, opts)?
        }
        // Measured default for a two-stream pair (§8p): the production mapping's
        // nine hottest blocks. It does not fit once two streams share a card, so it
        // is not used above two streams.
        None if n_streams == 2 && devs.len() >= 2 => vec![breccia_engine::ArenaSpec::List(vec![
            0, 1, 2, 4, 8, 18, 20, 22, 23,
        ])],
        None => vec![breccia_engine::ArenaSpec::Off],
    };
    let (stream_chunks, mut stream_specs): (Vec<usize>, Vec<breccia_engine::ArenaSpec>) = (0
        ..n_streams)
        .map(|s| {
            let g = groups.get(s).or_else(|| groups.last()).cloned().unwrap_or_default();
            stream_chunk_arena(g, &a.prefill_chunks, s, n_streams.div_ceil(devs.len()), a.ctx)
        })
        .unzip();
    // Item 2 (§11j): `BRECCIA_PF_EXPERT_DEVICE=1` also opens the first `--devices`
    // card in every other stream's model as device 1 with no blocks, so that
    // stream's streamed-expert GEMMs run there (the §11f host-staged path). Its
    // chunk scratch there is allocated with its session, after every arena is
    // chosen, so the helper card's own streams reserve room for it up front.
    let helper = std::env::var("BRECCIA_PF_EXPERT_DEVICE").as_deref() == Ok("1");
    let helped = |s: usize| helper && devs[s % devs.len()] != devs[0];
    for s in (0..n_streams).filter(|&s| devs[s % devs.len()] == devs[0]) {
        if let breccia_engine::ArenaSpec::Auto(o) = &stream_specs[s] {
            let o = (0..n_streams)
                .filter(|&h| helped(h))
                .fold(*o, |o, h| {
                    // shortcut: a helper venue spills to GTT (RX helper GEMMs 75 -> 275 ms)
                    // unless its fold gets a margin; fixed 800 MiB measured at 1k only
                    // (reserve 2560 ok, 1536 still spills), kernel mechanism open.
                    // Re-measure before longer prompts.
                    let f = o.with_prefill_chunk(stream_chunks[h], a.ctx);
                    let extra = std::env::var("BRECCIA_HELPER_MARGIN_MIB")
                        .ok()
                        .and_then(|v| v.parse::<u64>().ok())
                        .unwrap_or(800)
                        << 20;
                    breccia_engine::ArenaOpts {
                        display_reserve: f.display_reserve + extra,
                        headroom: f.headroom + extra,
                    }
                });
            stream_specs[s] = breccia_engine::ArenaSpec::Auto(o);
        }
    }
    println!(
        "breccia bench token --streams {n_streams}: {} prompt ids, {} to generate each, \
         one device per stream",
        prompt.len(),
        a.gen
    );
    // G1 (`BRECCIA_SHARED_DEVICE=1`, docs/coord/G1-shared-device.md): streams on
    // the same card share one loaded model instead of each loading a copy, which
    // spills the second copy to GTT. Off: one model per stream, as before.
    let mut owner: Vec<usize> = (0..n_streams).collect();
    if breccia_vk::ctx::shared_device() {
        for s in 0..n_streams {
            let dev = &devs[s % devs.len()];
            if let Some(f) = (0..s).find(|&f| &devs[f % devs.len()] == dev) {
                if stream_specs[f] != stream_specs[s] {
                    anyhow::bail!(
                        "BRECCIA_SHARED_DEVICE=1: streams {f} and {s} share {dev} but ask for \
                         different --gpu-experts"
                    );
                }
                owner[s] = f;
            }
        }
        // Two sessions on one device must run on separate threads: the lockstep
        // decode and the token-by-token prompt pass interleave them on one thread,
        // and the second `dispatch_batch::begin` on that device then fails.
        if owner.iter().enumerate().any(|(s, &o)| o != s)
            && (std::env::var("BRECCIA_MULTI_DECODE_CONCURRENT").as_deref() != Ok("1")
                || a.prefill_chunk == 0)
        {
            anyhow::bail!(
                "BRECCIA_SHARED_DEVICE=1 with two streams on one card needs \
                 BRECCIA_MULTI_DECODE_CONCURRENT=1 and --prefill-chunk > 0"
            );
        }
    }
    // Helped streams load first so their helper-side load allocations land before
    // the helper card's own arena measures free VRAM.
    let mut loaded: Vec<Option<breccia_engine::Model>> = (0..n_streams).map(|_| None).collect();
    let order: Vec<usize> = (0..n_streams)
        .filter(|&s| helped(s))
        .chain((0..n_streams).filter(|&s| !helped(s)))
        .collect();
    for s in order {
        if owner[s] != s {
            continue;
        }
        let dev = devs[s % devs.len()].clone();
        let arena = stream_specs[s].clone();
        let (devices, blocks) = if helped(s) {
            (vec![dev.clone(), devs[0].clone()], Some(vec![usize::MAX, 0]))
        } else {
            (vec![dev.clone()], None)
        };
        let model = breccia_engine::Model::load(
            glob,
            &breccia_engine::LoadOptions {
                devices,
                ctx_capacity: a.ctx,
                blocks,
                head_device: breccia_engine::HeadSpec::Auto,
                profile: None,
                arena,
                host_expert_import: a.host_expert_import,
                page_hint_trim: a.page_hint_trim,
                verbose: s == 0,
            },
        )
        .with_context(|| format!("loading stream {s}'s model on {dev}"))?;
        if s == 0 {
            print!("{}", model.stats.banner(&model.cfg));
        }
        breccia_engine::graph::assert_head(&model)?;
        loaded[s] = Some(model);
    }
    let models: Vec<breccia_engine::Model> = loaded.into_iter().flatten().collect();
    // `models` holds the owners in stream order, so owner `o` sits at its rank.
    let rank = |o: usize| (0..o).filter(|&i| owner[i] == i).count();
    let refs: Vec<&breccia_engine::Model> = owner.iter().map(|&o| &models[rank(o)]).collect();
    let cfg = breccia_engine::SessionCfg {
        ctx_capacity: a.ctx,
        threads: a.threads.unwrap_or(0),
        timers: a.timers,
        batching: a.batching,
        inbatch_read: a.inbatch_read,
        prefill_chunk: a.prefill_chunk,
        prefill_merged: a.prefill_merged,
        probe: std::env::var_os("BRECCIA_GPU_TIME").is_some(),
        pool: None,
    };
    let mut multi = breccia_engine::Multi::with_chunks(&refs, &cfg, &stream_chunks)
        .context("building the streams")?;
    // `BRECCIA_BENCH_MIXED_PROMPTS=1`: stream 0 keeps the reference prompt, the others
    // get `synthetic_prefix` of the same length (seeded by stream index), so routing
    // differs per stream the way real agents' does; identical prompts route
    // identically and the later streams re-read experts from L3.
    let mixed = std::env::var_os("BRECCIA_BENCH_MIXED_PROMPTS").is_some_and(|v| v == "1");
    let vocab = models[0].cfg.vocab as u32;
    let prompts: Vec<Vec<u32>> = (0..n_streams)
        .map(|s| {
            if s == 0 || !mixed {
                prompt.to_vec()
            } else {
                synthetic_prefix(prompt, prompt.len(), 0x9e37_79b9 ^ ((s as u32) * 8191), vocab)
            }
        })
        .collect();
    for s in multi.streams.iter_mut() {
        s.sampling = breccia_engine::Sampling {
            temperature: a.temperature,
            top_k: 0,
            seed: 0x9e37_79b9_7f4a_7c15,
        };
    }
    let names: Vec<(String, String)> = multi
        .streams
        .iter()
        .map(|s| s.device_names().into_iter().next().unwrap_or_default())
        .collect();
    println!(
        "session  {} streams · {} shared expert threads ({}) · {} MiB state + scratch per stream",
        multi.len(),
        multi.streams[0].threads(),
        breccia_kernels::cpu::path().name(),
        multi.streams[0].session_bytes() >> 20
    );
    for (i, (name, pci)) in names.iter().enumerate() {
        let what = match &stream_specs[i] {
            breccia_engine::ArenaSpec::List(v) => format!("{} expert block(s) {:?}", v.len(), v),
            breccia_engine::ArenaSpec::Auto(_) => {
                "expert arena sized by measured free VRAM".to_string()
            }
            breccia_engine::ArenaSpec::Off => "0 expert block(s)".to_string(),
        };
        println!("  stream {i}  {name:<40} {pci}  {what}");
    }

    // ---- warm passes: `--warm N` full generations before the window ----------
    // One is the steady-state protocol: the first generation faults this run's
    // experts in from NVMe (tens of GB), the following one measures what a served
    // agent actually feels.
    let warm_runs = a.warm;
    for k in 0..warm_runs.max(1) {
        let r0 = io_read_mib();
        let w = multi
            .generate_each(&prompts, a.gen.min(8))
            .with_context(|| format!("warm pass {k}"))?;
        let r1 = io_read_mib();
        println!(
            "warm {k}  {:.2} tok/s aggregate ({:.2} tok/s per stream) · {:.2} GiB storage read",
            w.aggregate_tokens_per_s(),
            w.stream_tokens_per_s(),
            (r1 - r0) / 1024.0
        );
    }
    for s in multi.streams.iter_mut() {
        s.reset_counters()?;
    }

    // ---- measured ---------------------------------------------------------
    let mut runs = Vec::with_capacity(a.runs.max(1));
    for run in 0..a.runs.max(1) {
        let g = multi
            .generate_each(&prompts, a.gen)
            .with_context(|| format!("the two-stream run {run}"))?;
        println!(
            "run {run}  {:>6.2} tok/s aggregate  {:>6.2} tok/s/stream  {:>6.2} ms/stream-token  \
             {} steps  lane {:.1} ms/step  fence {:.1} ms/step  overlap {:.0}%",
            g.aggregate_tokens_per_s(),
            g.stream_tokens_per_s(),
            g.stream_ms_per_token(),
            g.steps,
            g.lane_ms_per_step(),
            g.fence_ms_per_step(),
            g.overlap_efficiency() * 100.0
        );
        println!(
            "  prefill  {:.3} s ({:.0} tok/s/stream; {} path)",
            g.prefill_seconds,
            prompt.len() as f64 / g.prefill_seconds.max(1e-9),
            if a.prefill_chunk > 0 {
                "chunked"
            } else {
                "token by token"
            }
        );
        for (i, st) in g.streams.iter().enumerate() {
            println!(
                "  stream {i}  {} tokens  lane {:.1} ms/step ({:.0}% under the other stream's GPU)  \
                 fence {:.1} ms/step  {:?}",
                st.tokens.len(),
                st.lane_s * 1e3 / g.steps.max(1) as f64,
                if st.lane_s > 0.0 {
                    st.lane_covered_s / st.lane_s * 100.0
                } else {
                    0.0
                },
                st.fence_s * 1e3 / g.steps.max(1) as f64,
                &st.tokens[..st.tokens.len().min(8)]
            );
            println!(
                "           prefill {:.3} s on {}",
                st.prefill_seconds, names[i].0
            );
        }
        // The probe measures GPU busy time per *device*, so utilisation is per
        // card: with two streams on one card the two sessions' busy times add up.
        let mut per_card: Vec<(String, String, Vec<usize>, f64)> = Vec::new();
        let mut busy_total = 0.0;
        for (i, sess) in multi.streams.iter_mut().enumerate() {
            let busy = sess.gpu_busy_s()?;
            let b = busy.first().copied().unwrap_or(0.0);
            busy_total += b;
            sess.reset_gpu_busy()?;
            let pci = names[i].1.clone();
            match per_card.iter_mut().find(|c| c.1 == pci) {
                Some(c) => {
                    c.2.push(i);
                    c.3 += b;
                }
                None => per_card.push((names[i].0.clone(), pci, vec![i], b)),
            }
        }
        for (name, _pci, streams_on_it, b) in &per_card {
            println!(
                "  card      {name:<40} streams {streams_on_it:?}  GPU busy {}",
                busy_summary(gpu_busy_measured(), *b, g.decode_seconds)
            );
        }
        println!(
            "  host {:.1} ms/step in finish_block (wait+read+lane), {:.1} ms/step in \
             submit_block (record+submit)",
            multi.t_finish * 1e3 / g.steps.max(1) as f64,
            multi.t_submit * 1e3 / g.steps.max(1) as f64
        );
        println!(
            "  record {:.1} ms/step of host recording, {:.1} ms/step of ring waits \
             (all streams)",
            breccia_kernels::batch_cmd::record_seconds() * 1e3 / g.steps.max(1) as f64,
            breccia_kernels::batch_cmd::wait_seconds() * 1e3 / g.steps.max(1) as f64
        );
        println!(
            "  aggregate GPU busy {} across {} card(s)",
            busy_per_step(gpu_busy_measured(), busy_total, g.steps as u64),
            per_card.len()
        );
        runs.push(g);
    }

    let runs_n = runs.len().max(1) as f64;
    let agg: f64 = runs.iter().map(|g| g.aggregate_tokens_per_s()).sum::<f64>() / runs_n;
    let per: f64 = runs.iter().map(|g| g.stream_tokens_per_s()).sum::<f64>() / runs_n;
    let lat: f64 = runs.iter().map(|g| g.stream_ms_per_token()).sum::<f64>() / runs_n;
    let ovl: f64 = runs.iter().map(|g| g.overlap_efficiency()).sum::<f64>() / runs_n;
    println!(
        "\ndecode  {agg:.2} tok/s aggregate · {per:.2} tok/s per stream · {lat:.2} ms per \
         stream-token · overlap {:.0}% of the host lane hidden",
        ovl * 100.0
    );
    if a.timers {
        for (i, sess) in multi.streams.iter().enumerate() {
            println!(
                "\nstream {i} per-stage (ms per stream-token; *wall* attribution — a stage\
                 also contains the other stream's work, so use the numbers above)"
            );
            let table = sess
                .timers
                .table(runs.last().map(|g| g.steps).unwrap_or(1).max(1));
            for (name, ms) in table {
                if ms >= 0.05 {
                    println!("  {name:<20} {ms:>8.3}");
                }
            }
        }
    }
    let same = runs
        .last()
        .map(|g| g.streams.windows(2).all(|w| w[0].tokens == w[1].tokens))
        .unwrap_or(false);
    if mixed {
        // Mixed prompts differ by design: check each stream repeats itself run to run.
        let steady = runs.windows(2).all(|w| {
            w[0].streams.iter().zip(&w[1].streams).all(|(x, y)| x.tokens == y.tokens)
        });
        println!("parity   mixed prompts; every stream repeated its tokens in every run: {steady}");
    } else {
        println!(
            "parity   every stream generated the same {} tokens: {same}",
            runs.last().map(|g| g.streams[0].tokens.len()).unwrap_or(0)
        );
    }

    if let Some(path) = &a.json {
        let value = serde_json::json!({
            "streams": n_streams,
            "prompt_ids": prompt,
            "mixed_prompts": mixed,
            "aggregate_tokens_per_s": agg,
            "stream_tokens_per_s": per,
            "stream_ms_per_token": lat,
            "overlap_efficiency": ovl,
            "devices": names.iter().map(|(n, p)| serde_json::json!({"name": n, "pci": p})).collect::<Vec<_>>(),
            "runs": runs.iter().map(|g| serde_json::json!({
                "aggregate_tokens_per_s": g.aggregate_tokens_per_s(),
                "stream_tokens_per_s": g.stream_tokens_per_s(),
                "stream_ms_per_token": g.stream_ms_per_token(),
                "steps": g.steps,
                "decode_seconds": g.decode_seconds,
                "prefill_seconds": g.prefill_seconds,
                "prefill_tokens_per_s_per_stream": prompt.len() as f64 / g.prefill_seconds.max(1e-9),
                "lane_ms_per_step": g.lane_ms_per_step(),
                "fence_ms_per_step": g.fence_ms_per_step(),
                "overlap_efficiency": g.overlap_efficiency(),
                "streams": g.streams.iter().enumerate().map(|(i, s)| {
                    let mut v = serde_json::json!({
                        "device": names[i].0,
                        "tokens": s.tokens,
                        "lane_s": s.lane_s,
                        "lane_covered_s": s.lane_covered_s,
                        "fence_s": s.fence_s,
                        "expert_bytes": s.expert_bytes,
                        "prefill_seconds": s.prefill_seconds,
                    });
                    if let Some(ls) = &s.lane_stats {
                        v["lane_stats"] = serde_json::json!(ls);
                    }
                    v
                }).collect::<Vec<_>>(),
            })).collect::<Vec<_>>(),
        });
        if let Some(dir) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        std::fs::write(path, serde_json::to_string_pretty(&value)? + "\n")
            .with_context(|| format!("writing {}", path.display()))?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

/// A deterministic synthetic prompt for an agent stream.
///
/// Two thirds of the ids come from a seeded LCG over the vocabulary and one third
/// cycle through the reference prompt's ids, so the expert routing varies between
/// agents the way different work would. No tokenizer is available offline, so the
/// *content* is synthetic — but the workload (positions, QSA indexer growth, MoE
/// routing over 512 experts, 300-token history) is exactly the real one.
fn synthetic_prefix(reference: &[u32], n: usize, seed: u32, vocab: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity(n);
    let mut x = seed | 1;
    for i in 0..n {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        if i % 3 == 0 && !reference.is_empty() {
            out.push(reference[(i / 3) % reference.len()]);
        } else {
            out.push(x % vocab.max(1));
        }
    }
    out
}

/// `breccia bench agents`: N agent-style streams, long prefixes, greedy decode.
fn bench_agents(a: AgentArgs) -> Result<()> {
    let devs: Vec<String> = a
        .devices
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if devs.is_empty() {
        anyhow::bail!("--devices needs at least one PCI address");
    }
    let n = a.streams.max(1);
    let reference = parse_reference_prompt();
    let glob = DEFAULT_MODEL_GLOB.to_string();
    println!(
        "breccia bench agents: {n} stream(s) over {:?}, {} prefixes (agent 0 = the parity \
         reference prompt of {} ids), {} greedy tokens each",
        devs,
        a.prefix,
        reference.len(),
        a.gen
    );
    // Same strict parse and per-stream selection as bench_token_multi.
    let opts = breccia_engine::ArenaOpts {
        display_reserve: a.arena_display_reserve_mib << 20,
        headroom: a.arena_headroom_mib << 20,
    };
    let groups: Vec<breccia_engine::ArenaSpec> = match a.gpu_experts.as_deref() {
        Some(spec) => multi_arena_specs(spec, n, opts)?,
        None if n == 2 && devs.len() >= 2 => vec![breccia_engine::ArenaSpec::List(vec![
            0, 1, 2, 4, 8, 18, 20, 22, 23,
        ])],
        None => vec![breccia_engine::ArenaSpec::Off],
    };
    let mut models = Vec::with_capacity(n);
    let mut stream_chunks = Vec::with_capacity(n);
    for s in 0..n {
        let dev = devs[s % devs.len()].clone();
        let (chunk, arena) = stream_chunk_arena(
            groups.get(s).or_else(|| groups.last()).cloned().unwrap_or(breccia_engine::ArenaSpec::Off),
            &a.prefill_chunks,
            s,
            n.div_ceil(devs.len()),
            512,
        );
        stream_chunks.push(chunk);
        let model = breccia_engine::Model::load(
            &glob,
            &breccia_engine::LoadOptions {
                devices: vec![dev.clone()],
                ctx_capacity: 512,
                blocks: None,
                head_device: breccia_engine::HeadSpec::Auto,
                profile: None,
                arena,
                host_expert_import: false,
                page_hint_trim: false,
                verbose: s == 0,
            },
        )
        .with_context(|| format!("loading agent {s}'s model on {dev}"))?;
        if s == 0 {
            print!("{}", model.stats.banner(&model.cfg));
        }
        breccia_engine::graph::assert_head(&model)?;
        models.push(model);
    }
    let vocab = models[0].cfg.vocab as u32;
    let refs: Vec<&breccia_engine::Model> = models.iter().collect();
    let cfg = breccia_engine::SessionCfg {
        ctx_capacity: 512,
        threads: a.threads.unwrap_or(0),
        timers: true,
        batching: true,
        inbatch_read: a.inbatch_read,
        prefill_chunk: a.prefill_chunk,
        prefill_merged: a.prefill_merged,
        probe: true,
        pool: None,
    };
    let mut multi = breccia_engine::Multi::with_chunks(&refs, &cfg, &stream_chunks)
        .context("building the agent streams")?;
    for s in multi.streams.iter_mut() {
        s.sampling = breccia_engine::Sampling {
            temperature: 0.0,
            top_k: 0,
            seed: 0x9e37_79b9_7f4a_7c15,
        };
    }
    let prompts: Vec<Vec<u32>> = (0..n)
        .map(|s| {
            if s == 0 || a.same_prompt {
                reference.clone()
            } else {
                synthetic_prefix(
                    &reference,
                    a.prefix,
                    0x9e37_79b9 ^ ((s as u32) * 8191),
                    vocab,
                )
            }
        })
        .collect();
    let names: Vec<(String, String)> = multi
        .streams
        .iter()
        .map(|s| s.device_names().into_iter().next().unwrap_or_default())
        .collect();
    println!(
        "session  {} streams · {} shared expert threads ({}) · prompts {:?} ids",
        multi.len(),
        multi.streams[0].threads(),
        breccia_kernels::cpu::path().name(),
        prompts.iter().map(|p| p.len()).collect::<Vec<_>>()
    );

    // ---- warm passes: `--warm N` generations before the window --------------
    // Cold, this box re-reads tens of GB of experts from NVMe inside the measured
    // window (41 GB of experts + 27 GB of PLE against 58 GB of RAM), so the cold
    // rate is an I/O number and `--warm 1` is the steady-state one.
    for k in 0..a.warm.max(1) {
        let r0 = io_read_mib();
        let w = multi
            .generate(&prompts[0], a.gen.min(8))
            .with_context(|| format!("warm pass {k}"))?;
        println!(
            "warm {k}  {:.2} tok/s aggregate ({:.2} tok/s per stream) · {:.2} GiB storage read",
            w.aggregate_tokens_per_s(),
            w.stream_tokens_per_s(),
            (io_read_mib() - r0) / 1024.0
        );
    }
    for s in multi.streams.iter_mut() {
        s.reset_counters()?;
    }
    breccia_kernels::batch_cmd::reset_record_seconds();
    multi.t_finish = 0.0;
    multi.t_submit = 0.0;
    let read_before = io_read_mib();

    // ---- the measured run: one generate() per agent is not possible (they must
    // interleave), so drive token_all directly over all agents' prompts --------
    let mut logits: Vec<Vec<f32>> = Vec::with_capacity(n);
    for sess in multi.streams.iter_mut() {
        sess.reset()?;
    }
    // All agents advance in lockstep, so a shorter prefix is extended by
    // repeating its last id (a real scheduler holding shorter work would do the
    // same: the streams must be at the same position to interleave).
    let common = if a.same_prompt {
        reference.len()
    } else {
        prompts.iter().map(|p| p.len()).max().unwrap_or(0)
    };
    let padded: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| {
            let mut v = p.clone();
            if let Some(&last) = p.last() {
                v.resize(common, last);
            }
            v
        })
        .collect();
    let t_prompt = std::time::Instant::now();
    if a.prefill_chunk > 0 {
        logits = vec![Vec::new(); n];
        for (s, sess) in multi.streams.iter_mut().enumerate() {
            let mut pos = 0usize;
            for len in breccia_engine::prefill_plan(padded[s].len(), stream_chunks[s]) {
                logits[s] = sess.prefill(&padded[s][pos..pos + len], pos)?;
                pos += len;
            }
        }
    } else {
        for pos in 0..common {
            let toks: Vec<u32> = padded.iter().map(|p| p[pos]).collect();
            logits = multi.token_all(&toks, pos)?;
        }
    }
    let prompt_seconds = t_prompt.elapsed().as_secs_f64();
    let prompt_tokens = common * n;
    // The merged-MoE budget of the whole prompt, captured before the counters are
    // zeroed for the decode window (§8u).
    let merged_budget = multi
        .streams
        .iter()
        .fold((0usize, 0u64, 0usize), |acc, st| {
            let (d, b, t) = st.merged_prefill_budget();
            (acc.0 + d, acc.1 + b, acc.2 + t)
        });
    // The merged pass's own breakdown has to be read here too: the counters below
    // are zeroed for the decode window.
    let merged_detail =
        multi
            .streams
            .iter()
            .fold((0.0, 0.0, 0u64, 0usize, 0u64, 0usize), |acc, st| {
                let (g, f, d, gr, b, sl) = st.merged_prefill_detail();
                (
                    acc.0 + g,
                    acc.1 + f,
                    acc.2 + d,
                    acc.3 + gr,
                    acc.4 + b,
                    acc.5 + sl,
                )
            });
    // The prompt is not part of the decode measurement (counters, probe, timers).
    for s in multi.streams.iter_mut() {
        s.reset_counters()?;
    }
    breccia_kernels::batch_cmd::reset_record_seconds();
    multi.t_finish = 0.0;
    multi.t_submit = 0.0;
    let mut tokens: Vec<Vec<u32>> = vec![Vec::with_capacity(a.gen); n];
    let mut steps_s: Vec<f64> = Vec::with_capacity(a.gen);
    let t0 = std::time::Instant::now();
    let mut pos = prompts.iter().map(|p| p.len()).max().unwrap_or(0);
    #[allow(clippy::explicit_counter_loop)]
    for i in 0..a.gen {
        let next: Vec<u32> = (0..n)
            .map(|s| multi.streams[s].sample(&logits[s]).unwrap_or(0))
            .collect();
        for (s, &t) in next.iter().enumerate() {
            tokens[s].push(t);
        }
        if i + 1 == a.gen {
            break;
        }
        let t_step = std::time::Instant::now();
        logits = multi.token_all(&next, pos)?;
        steps_s.push(t_step.elapsed().as_secs_f64());
        pos += 1;
    }
    let decode_seconds = t0.elapsed().as_secs_f64();
    let steps = steps_s.len();

    // ---- report ----------------------------------------------------------
    let mut sorted = steps_s.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pct = |q: f64| -> f64 {
        if sorted.is_empty() {
            return 0.0;
        }
        let i = ((sorted.len() - 1) as f64 * q).round() as usize;
        sorted[i] * 1e3
    };
    let agg = if decode_seconds > 0.0 {
        steps as f64 * n as f64 / decode_seconds
    } else {
        0.0
    };
    let per = if decode_seconds > 0.0 {
        steps as f64 / decode_seconds
    } else {
        0.0
    };
    let lane_total: f64 = multi.streams.iter().map(|s| s.lane_s).sum();
    let lane_covered: f64 = multi.streams.iter().map(|s| s.lane_covered_s).sum();
    let mut per_card: Vec<(String, Vec<usize>, f64)> = Vec::new();
    let mut busy_total = 0.0;
    for (i, sess) in multi.streams.iter_mut().enumerate() {
        let b = sess.gpu_busy_s()?.first().copied().unwrap_or(0.0);
        busy_total += b;
        let pci = names[i].1.clone();
        match per_card.iter_mut().find(|c| c.0 == pci) {
            Some(c) => {
                c.1.push(i);
                c.2 += b;
            }
            None => per_card.push((pci, vec![i], b)),
        }
    }
    let (distinct, merged_bytes, pf_tokens) = merged_budget;
    if pf_tokens > 0 {
        println!(
            "merged   {} tokens: {} distinct expert uses (summed over 48 blocks), \
             {:.1} GiB of expert bytes if each is touched once ({:.1} MiB/token, \
             vs {:.1} MiB/token token-by-token)",
            pf_tokens,
            distinct,
            merged_bytes as f64 / (1u64 << 30) as f64,
            merged_bytes as f64 / pf_tokens as f64 / (1u64 << 20) as f64,
            799.0
        );
    }
    {
        let (gemm_s, fanin_s, dispatches, groups, bytes, slots) = merged_detail;
        if groups > 0 {
            println!(
                "merged   {groups} expert GEMM groups over {slots} slots: {:.1} ms of GEMM,                  {:.1} ms of fan-in, {dispatches} dispatches ({:.1} GiB read once per expert,                  {:.0} us/group)",
                gemm_s * 1e3,
                fanin_s * 1e3,
                bytes as f64 / (1u64 << 30) as f64,
                gemm_s * 1e6 / groups as f64
            );
            let (sb, sf, sw) = multi.streams.iter().fold((0u64, 0.0, 0.0), |acc, st| {
                let (b, f, w) = st.merged_stream_detail();
                (acc.0 + b, acc.1 + f, acc.2 + w)
            });
            if sb > 0 {
                println!(
                    "stream   {:.1} GiB of expert blobs copied out of the mmap in {:.1} ms \
                     ({:.1} GiB/s), {:.1} ms waiting on the staging ring ({} experts per submit)",
                    sb as f64 / (1u64 << 30) as f64,
                    sf * 1e3,
                    sb as f64 / (1u64 << 30) as f64 / sf.max(1e-9),
                    sw * 1e3,
                    breccia_engine::graph::STREAM_SLOTS
                );
            }
        }
    }
    println!(
        "\nprefill  {prompt_tokens} prompt tokens ({common} x {n}) in {prompt_seconds:.2} s = \
         {:.2} tok/s per agent (chunk {:?} = {}, grouped lane {})",
        prompt_tokens as f64 / prompt_seconds.max(1e-9) / n as f64,
        stream_chunks,
        if a.prefill_chunk > 0 {
            "chunked"
        } else {
            "token by token"
        },
        if a.prefill_chunk > 0 { "on" } else { "n/a" }
    );
    println!(
        "cache    {:.2} GiB read from storage during the measured window ({:.1} MiB per prompt \
         token) · --warm {}",
        (io_read_mib() - read_before) / 1024.0,
        (io_read_mib() - read_before) / prompt_tokens.max(1) as f64,
        a.warm
    );
    println!(
        "agents   {n} concurrent · {steps} measured steps · {:.2} s of decode",
        decode_seconds
    );
    println!(
        "decode   {agg:.2} tok/s aggregate · {per:.2} tok/s per agent · {:.1} ms per step \
         (p50 {:.1} / p95 {:.1} / max {:.1} ms)",
        decode_seconds * 1e3 / steps.max(1) as f64,
        pct(0.50),
        pct(0.95),
        sorted.last().map(|x| x * 1e3).unwrap_or(0.0)
    );
    println!(
        "host     expert lane {:.1} ms/step ({:.0}% of the step, {:.0}% of it under another \
         agent's GPU work) · {:.1} ms/step of ring waits",
        lane_total * 1e3 / steps.max(1) as f64,
        lane_total * 1e3 / (decode_seconds * 1e3).max(1e-9) * 100.0,
        if lane_total > 0.0 {
            lane_covered / lane_total * 100.0
        } else {
            0.0
        },
        breccia_kernels::batch_cmd::wait_seconds() * 1e3 / steps.max(1) as f64
    );
    for (pci, streams_on_it, b) in &per_card {
        let name = names
            .iter()
            .find(|x| &x.1 == pci)
            .map(|x| x.0.clone())
            .unwrap_or_default();
        println!(
            "card     {name:<40} agents {streams_on_it:?}  GPU busy {}",
            busy_summary(gpu_busy_measured(), *b, decode_seconds)
        );
    }
    println!(
        "         aggregate GPU busy {} over {} card(s)",
        busy_per_step(gpu_busy_measured(), busy_total, steps as u64),
        per_card.len()
    );
    let parity = if a.same_prompt {
        format!(
            "all agents got the reference prompt; agent 0 vs the server: {}",
            reference_parity(&tokens[0])
        )
    } else {
        "mixed prompts (run with --same-prompt for server parity)".to_string()
    };
    println!("parity   {parity}");
    let dup = (1..n)
        .map(|s| (s, tokens[s].iter().take(8).copied().collect::<Vec<_>>()))
        .collect::<Vec<_>>();
    for (s, t) in dup {
        println!("         agent {s} first tokens {t:?}");
    }

    if let Some(path) = &a.json {
        let value = serde_json::json!({
            "agents": n,
            "steps": steps,
            "decode_seconds": decode_seconds,
            "prefill_tokens_per_s_per_agent": prompt_tokens as f64 / prompt_seconds.max(1e-9) / n as f64,
            "prefill_seconds": prompt_seconds,
            "aggregate_tokens_per_s": agg,
            "per_agent_tokens_per_s": per,
            "latency_ms": {"p50": pct(0.50), "p95": pct(0.95), "max": sorted.last().map(|x| x * 1e3)},
            "lane_ms_per_step": lane_total * 1e3 / steps.max(1) as f64,
            "lane_covered_fraction": if lane_total > 0.0 { lane_covered / lane_total } else { 0.0 },
            "ring_wait_ms_per_step": breccia_kernels::batch_cmd::wait_seconds() * 1e3 / steps.max(1) as f64,
            "prompt_lengths": prompts.iter().map(|p| p.len()).collect::<Vec<_>>(),
            "tokens": tokens,
            "cards": per_card.iter().map(|(pci, s, b)| serde_json::json!({
                "pci": pci, "agents": s, "gpu_utilisation": b / decode_seconds.max(1e-9),
            })).collect::<Vec<_>>(),
            "reference_parity": if a.same_prompt { reference_parity(&tokens[0]) } else { "n/a (mixed prompts)".to_string() },
        });
        if let Some(dir) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        std::fs::write(path, serde_json::to_string_pretty(&value)? + "\n")
            .with_context(|| format!("writing {}", path.display()))?;
        println!("wrote {}", path.display());
    }
    Ok(())
}

/// The reference prompt's ids (the ones `parity.py` captured), or a fallback.
/// Bytes this process has pulled **from storage** (page faults included), from
/// `/proc/self/io`. The measurement box cannot hold the model — 41 GB of experts +
/// 27 GB of PLE against 58 GB of RAM — so a cold run re-reads tens of GB and the
/// host expert lane stalls on NVMe. Printing this next to the rate is what makes a
/// "cold" and a "steady-state" number distinguishable instead of guessed at.
fn io_read_mib() -> f64 {
    let Ok(text) = std::fs::read_to_string("/proc/self/io") else {
        return 0.0;
    };
    for line in text.lines() {
        if let Some(v) = line.strip_prefix("read_bytes:") {
            if let Ok(b) = v.trim().parse::<u64>() {
                return b as f64 / (1u64 << 20) as f64;
            }
        }
    }
    0.0
}

fn parse_reference_prompt() -> Vec<u32> {
    // Parity-reference dir is host-specific; supplied via BRECCIA_PARITY_DIR.
    let dir = match std::env::var("BRECCIA_PARITY_DIR") {
        Ok(d) if !d.is_empty() => d,
        _ => return Vec::new(),
    };
    let path = format!("{dir}/parity-server.json.ids");
    std::fs::read_to_string(path)
        .ok()
        .map(|t| {
            t.split(|c: char| c == ',' || c.is_whitespace())
                .filter_map(|x| x.trim().parse::<u32>().ok())
                .collect()
        })
        .filter(|v: &Vec<u32>| !v.is_empty())
        .unwrap_or_else(|| vec![760, 13017, 1433, 364, 279, 5447])
}

/// `n/n` of agent 0's generated ids against the server's recorded sequence.
fn reference_parity(ours: &[u32]) -> String {
    // Parity-reference dir is host-specific; supplied via BRECCIA_PARITY_DIR.
    let Ok(dir) = std::env::var("BRECCIA_PARITY_DIR") else {
        return "no reference file".to_string();
    };
    if dir.is_empty() {
        return "no reference file".to_string();
    }
    let path = format!("{dir}/parity-server.json");
    let Ok(text) = std::fs::read_to_string(path) else {
        return "no reference file".to_string();
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else {
        return "unreadable reference".to_string();
    };
    let Some(ref_ids) = v.get("generated").and_then(|g| g.as_array()) else {
        return "no `generated` in the reference".to_string();
    };
    let ref_ids: Vec<u32> = ref_ids
        .iter()
        .filter_map(|x| x.as_u64().map(|y| y as u32))
        .collect();
    let n = ours.len().min(ref_ids.len());
    let ok = (0..n).filter(|&i| ours[i] == ref_ids[i]).count();
    format!("{ok}/{n}")
}

/// `BRECCIA_PREFILL_ONLY=1`: one warm prompt pass, then `--runs` measured passes;
/// prints prefill seconds, the GPU stage table per prompt token (with
/// `BRECCIA_GPU_TIME=1`) and, with `--dump-logits PATH`, writes the last pass's
/// full last-token logits as raw little-endian f32 (the parity input).
fn prefill_only(
    a: &TokenArgs,
    sess: &mut breccia_engine::Session<'_>,
    prompt: &[u32],
) -> Result<()> {
    feed_prompt(sess, prompt, a.prefill_chunk).context("the warm prompt pass")?;
    let mut secs = Vec::new();
    let mut logits = Vec::new();
    for run in 0..a.runs.max(1) {
        sess.reset_counters()?;
        let t0 = std::time::Instant::now();
        logits = feed_prompt(sess, prompt, a.prefill_chunk)
            .with_context(|| format!("prefill run {run}"))?;
        let s = t0.elapsed().as_secs_f64();
        println!(
            "prefill run {run}: {:.3} s · {:.2} ms/token · {:.1} tok/s ({} tokens)",
            s,
            s * 1e3 / prompt.len() as f64,
            prompt.len() as f64 / s,
            prompt.len()
        );
        secs.push(s);
    }
    // The stage table divides by `forwards`; a prefill counts its prompt tokens.
    sess.forwards = prompt.len();
    if let Some(rows) = sess.gpu_time_table() {
        println!("\ngpu time (us/prompt-token, last run):");
        let mut sum = 0.0;
        for row in &rows {
            println!("  {:<14} {:>9.1}", row.stage, row.total_us);
            sum += row.total_us;
        }
        println!("  {:<14} {:>9.1}", "sum", sum);
    }
    // Host recording vs ring waits (last run): which side bounds the dense chain.
    let (rec, n) = (
        breccia_kernels::batch_cmd::record_seconds(),
        breccia_kernels::batch_cmd::recorded_dispatches().max(1),
    );
    println!(
        "host record {:.2} s ({:.0} ns/dispatch, {n} dispatches) · ring wait {:.2} s",
        rec,
        rec * 1e9 / n as f64,
        breccia_kernels::batch_cmd::wait_seconds()
    );
    let s1 = breccia_engine::LogitSummary::of_top(&logits, 10);
    println!("top-1 {} ({:.4}) · top-10 {:?}", s1.top1, s1.top1_logit, s1.top10);
    if let Some(path) = &a.dump_logits {
        let bytes: Vec<u8> = logits.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))?;
    }
    if let Some(path) = &a.json {
        let best = secs.iter().copied().fold(f64::MAX, f64::min);
        let j = serde_json::json!({
            "prompt_tokens": prompt.len(),
            "prefill_chunk": a.prefill_chunk,
            "prefill_s": secs,
            "best_ms_per_token": best * 1e3 / prompt.len() as f64,
            "top1": s1.top1,
        });
        std::fs::write(path, serde_json::to_string_pretty(&j)?)
            .with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}

fn bench_token(a: TokenArgs) -> Result<()> {
    let prompt = parse_prompt_ids(&a)?;
    let glob = a
        .gguf
        .clone()
        .unwrap_or_else(|| DEFAULT_MODEL_GLOB.to_string());
    if a.streams > 1 {
        return bench_token_multi(&a, &prompt, &glob);
    }
    println!(
        "breccia bench token: {} prompt ids, {} to generate, {} run(s)",
        prompt.len(),
        a.gen,
        a.runs
    );

    // ---- load -------------------------------------------------------------
    // `--gpu-experts` is `auto`, an explicit block list, or absent (no arena).
    let arena = match a.gpu_experts.as_deref() {
        None => breccia_engine::ArenaSpec::Off,
        Some(list) => {
            let mut arena = breccia_engine::ArenaSpec::parse(list)
                .map_err(anyhow::Error::msg)
                .with_context(|| format!("parsing --gpu-experts {list:?}"))?;
            if let breccia_engine::ArenaSpec::Auto(_) = arena {
                arena = breccia_engine::ArenaSpec::Auto(breccia_engine::ArenaOpts {
                    display_reserve: a.arena_display_reserve_mib << 20,
                    headroom: a.arena_headroom_mib << 20,
                }
                .with_prefill_chunk(a.prefill_chunk, a.ctx));
            }
            arena
        }
    };
    let model = breccia_engine::Model::load(
        &glob,
        &breccia_engine::LoadOptions {
            devices: a.devices.clone(),
            ctx_capacity: a.ctx,
            blocks: a.split_blocks.as_ref().map(|s| {
                s.split(',')
                    .filter_map(|t| t.trim().parse::<usize>().ok())
                    .collect()
            }),
            head_device: match a.head_device.as_deref() {
                None => breccia_engine::HeadSpec::Auto,
                Some(v) => breccia_engine::HeadSpec::parse(v)?,
            },
            profile: None,
            arena,
            host_expert_import: a.host_expert_import,
            page_hint_trim: a.page_hint_trim,
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
        inbatch_read: a.inbatch_read,
        prefill_chunk: a.prefill_chunk,
        prefill_merged: a.prefill_merged,
        probe: false,
        pool: None,
    };
    let mut sess = breccia_engine::Session::new(&model, &cfg).context("building the session")?;
    sess.sampling = breccia_engine::Sampling {
        temperature: a.temperature,
        top_k: 0,
        seed: 0x9e37_79b9_7f4a_7c15,
    };
    if std::env::var_os("BRECCIA_PREFILL_ONLY").is_some() {
        return prefill_only(&a, &mut sess, &prompt);
    }
    println!(
        "session  {} expert threads ({}) · {} MiB state + scratch · {} compiled pipelines",
        sess.threads(),
        breccia_kernels::cpu::path().name(),
        sess.session_bytes() >> 20,
        model.keys.len()
    );

    // ---- A. one token, twice: finite logits and a stable top-1 -------------
    let t_first = std::time::Instant::now();
    let l1 = feed_prompt(&mut sess, &prompt, a.prefill_chunk).context("the first forward pass")?;
    let first_seconds = t_first.elapsed().as_secs_f64();
    let l2 = feed_prompt(&mut sess, &prompt, a.prefill_chunk).context("the repeat forward pass")?;
    if let Some(t) = std::env::var("BRECCIA_SPEC_VERIFY_BENCH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        if !(2..=5).contains(&t) {
            anyhow::bail!("BRECCIA_SPEC_VERIFY_BENCH={t}: T must be 2..=5");
        }
        spec_verify_bench(&mut sess, &l2, prompt.len(), t).context("the spec-verify bench")?;
    }
    if let Some(n) = std::env::var("BRECCIA_MTP_DRAFT_BENCH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        mtp_draft_bench(&model, prompt.len(), n).context("the MTP draft bench")?;
    }
    if let Ok(arms) = std::env::var("BRECCIA_MTP_SPEC_BENCH") {
        mtp_spec_bench(&model, &mut sess, &prompt, a.prefill_chunk, &arms).context("the MTP spec bench")?;
    }
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

    // ---- warm generations (`--warm N`): steady-state protocol ---------------
    for k in 0..a.warm {
        let r0 = io_read_mib();
        let w = sess
            .generate(&prompt, a.gen, None)
            .with_context(|| format!("warm generation {k}"))?;
        println!(
            "warm {k}  {:.2} tok/s ({:.2} ms/token) · {:.2} GiB storage read",
            w.tokens_per_s(),
            w.ms_per_token(),
            (io_read_mib() - r0) / 1024.0
        );
    }

    // ---- B/C. measured generations ----------------------------------------
    let read_before = io_read_mib();
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
            lane_stats: sess.lane_report(),
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
    // ---- session reuse: a second prompt in the *same* process ---------------
    // `scripts/session-reset-test.sh` compares this line against a fresh process's
    // output. Anything in `Session::reset` that is missing shows up here as a
    // different sequence, deterministically (see docs/phase-4/01-forward-pass.md §8az).
    let mut then_tokens: Vec<u32> = Vec::new();
    if let Some(list) = &a.then_prompt_ids {
        let then: Vec<u32> = list
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter_map(|t| t.trim().parse::<u32>().ok())
            .collect();
        if then.is_empty() {
            anyhow::bail!("--then-prompt-ids is empty");
        }
        let gen = sess
            .generate(&then, a.gen, None)
            .context("the second prompt (session reuse)")?;
        then_tokens = gen.tokens.clone();
        println!(
            "\nthen     {} ids -> {} tokens at {:.2} tok/s: {:?}",
            then.len(),
            then_tokens.len(),
            gen.tokens_per_s(),
            &then_tokens[..then_tokens.len().min(8)]
        );
    }

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
        then_generated: then_tokens,
    };
    println!(
        "cache    {:.2} GiB read from storage during the measured window ({:.1} MiB per prompt \
         token) · --warm {}",
        (io_read_mib() - read_before) / 1024.0,
        (io_read_mib() - read_before) / prompt.len().max(1) as f64,
        a.warm
    );
    println!(
        "\nexperts  {} block(s) on device ({:.0} MiB arena), {} on the host lane",
        sess.placed_blocks,
        model
            .placed
            .values()
            .map(|p| p.gate.buf.size + p.up.buf.size + p.down.buf.size)
            .sum::<u64>() as f64
            / (1 << 20) as f64,
        sess.host_blocks
    );
    println!(
        "\nrecord   {:.2} ms/token of host recording ({} dispatches, {:.0} per submit)",
        breccia_kernels::batch_cmd::record_seconds() * 1e3 / runs_n,
        sess.dispatches,
        sess.dispatches as f64 / sess.submits.max(1) as f64
    );
    // **Submit accounting** (§8bg): how many queue submissions the token costs and,
    // of those, how many the host actually waited for. A wait is what costs 33 us
    // (research 08), not a submit, so this is the number a pipelining change moves.
    {
        let st = breccia_vk::ctx::submit_stats_window();
        let per = forwards.max(1) as f64;
        println!(
            "submits  {:.1}/token · {} total · waited {} ({:.1}/token) · queued {} ({:.1}/token) \
             · fence waits {} ({:.1}/token, {:.2} ms/token in the wait) · deferred {} · \
             unfenced readbacks {}",
            st.submits as f64 / per,
            st.submits,
            st.waited,
            st.waited as f64 / per,
            st.queued,
            st.queued as f64 / per,
            st.waits,
            st.waits as f64 / per,
            st.wait_s * 1e3 / per,
            sess.deferred_waits,
            sess.readbacks_unfenced,
        );
    }
    // The per-dispatch price from **one** window: `RECORD_NS` and its own dispatch
    // counter are both reset after the prompt, so their ratio is a real number
    // (`docs/research/11` §A6; dividing by `sess.dispatches`, a whole-session count,
    // is what produced the historical 3-11 us figures).
    {
        let n = breccia_kernels::batch_cmd::recorded_dispatches().max(1) as f64;
        println!(
            "record   {:.0} ns/dispatch over the measured window ({} dispatches, {:.1} us each \
             incl. submit share)",
            breccia_kernels::batch_cmd::record_seconds() * 1e9 / n,
            n as u64,
            mspt * 1e3 / n
        );
    }
    // **The GPU-time table** (`BRECCIA_GPU_TIME=1`): per stage, µs of GPU time per
    // token, and the residual against the host's fence wait. Every row is a GPU
    // interval between two stage boundaries read after the slot's fence, so `sum` is
    // how much of `wait-gpu` was execution and the residual is launch/serialisation
    // (`docs/research/07` §A6).
    if let Some(rows) = sess.gpu_time_table() {
        let names: Vec<String> = sess
            .device_names()
            .iter()
            .enumerate()
            .map(|(i, (n, pci))| format!("d{i}={n} [{pci}]"))
            .collect();
        println!("\ngpu time (us/token of GPU execution, decode window):");
        println!(
            "  tokens in window {} · forwards {}",
            sess.forwards, sess.forwards
        );
        let mut sum = 0.0;
        for row in &rows {
            let cols: Vec<String> = row
                .per_device
                .iter()
                .map(|(us, n)| format!("{us:>8.1} (n={n:>4})"))
                .collect();
            println!(
                "  {:<14} {}   Σ {:>8.1}",
                row.stage,
                cols.join(""),
                row.total_us
            );
            sum += row.total_us;
        }
        let wait = sess.wait_gpu_us_per_token();
        println!("  {:<14} {}", "devices", names.join(" · "));
        println!(
            "  {:<14} Σ {sum:>8.1} · wait-gpu {wait:>8.1} · residual {:+.1} us/token ({:+.1} % \
             of wait-gpu) = submit round trips, CB prologue and GPU idle waiting for the host",
            "sum",
            wait - sum,
            if wait > 0.0 {
                (wait - sum) / wait * 100.0
            } else {
                0.0
            }
        );
    }
    // Per-call record profile (docs/phase-4 §8aq): where the ~7 us/dispatch goes.
    {
        let p = breccia_kernels::batch_cmd::record_profile();
        let names = [
            "key+cache",
            "set write",
            "bind_pipeline",
            "bind_set",
            "push_const",
            "dispatch",
            "barrier",
            "tracker",
        ];
        let n = sess.dispatches.max(1) as f64;
        let rows: Vec<String> = names
            .iter()
            .zip(p.iter())
            .map(|(name, ns)| format!("{name} {:.0} ns", *ns as f64 / n))
            .collect();
        let sum: u64 = p.iter().sum();
        println!(
            "profile  {} ns/dispatch (summed {:.2} ms; measured record {:.2} ms/token)",
            rows.join(" · "),
            sum as f64 / 1e6,
            breccia_kernels::batch_cmd::record_seconds() * 1e3 / runs_n
        );
    }
    // Barrier accounting: `barriers + pruned` is the always-barrier shape, so the
    // pair is the per-resource tracker's own delta (docs/phase-4 §8al).
    println!(
        "barrier  {} emitted, {} pruned by the per-resource tracker ({:.2} vs {:.2} per token, \
         {:.2} barriers/dispatch)",
        breccia_kernels::batch_cmd::barriers(),
        breccia_kernels::batch_cmd::barriers_pruned(),
        breccia_kernels::batch_cmd::barriers() as f64 / runs_n,
        (breccia_kernels::batch_cmd::barriers() + breccia_kernels::batch_cmd::barriers_pruned())
            as f64
            / runs_n,
        breccia_kernels::batch_cmd::barriers() as f64 / sess.dispatches.max(1) as f64
    );
    // The head path's traffic is asserted, not just printed: a per-token weight
    // re-upload turns a 53 ms token into a 700 ms one, and a number in a log is
    // exactly what nobody reads (`docs/phase-4/01-forward-pass.md` §8aw).
    match sess.head_traffic_ok(forwards) {
        Ok((up, down)) => println!(
            "\nhead     device {} of {} · path traffic {:.1} KiB up + {:.1} KiB down per token \
             (residual {:.1} KiB, logits {:.0} KiB)",
            sess.head_device(),
            model.devices.len(),
            up / 1024.0,
            down / 1024.0,
            (model.cfg.hc_dim() * 4) as f64 / 1024.0,
            (model.cfg.vocab * 4) as f64 / 1024.0,
        ),
        Err(e) => return Err(e).context("the head path moved far more than a residual per token"),
    }
    if std::env::var_os("BRECCIA_DUMP_LAYOUT").is_some() {
        println!("\nlayout (BRECCIA_DUMP_LAYOUT):");
        for line in sess.layout_dump() {
            println!("  {line}");
        }
    }
    if breccia_engine::model::weight_check_bytes() > 0 {
        println!(
            "weights  {:.1} MiB of tensor uploads read back and compared byte-for-byte \
             (BRECCIA_VERIFY_WEIGHTS) — 0 divergences",
            breccia_engine::model::weight_check_bytes() as f64 / (1 << 20) as f64
        );
    }
    println!(
        "\nreadback {} in-batch readback(s) of which {} were read without a waited \
         submission (must be 0; docs/phase-4 §8ba)",
        sess.forwards * (48 - sess.placed_blocks) + sess.placed_blocks,
        sess.readbacks_unfenced
    );
    println!("\nper device (per generated token):");
    for line in sess.per_device(forwards / runs_n.max(1.0).floor() as usize) {
        println!("  {line}");
    }
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
