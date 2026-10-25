//! vk-probe — Phase 0.2 raw Vulkan micro-benchmark harness (headless, RADV).
//!
//! Usage:
//!   cargo run --release -- [--device N] [--json PATH] [--quick] [--all-devices]

mod bench;
mod ctx;
mod report;

use ctx::{api_version_string, Dev, Gpu, Host};
use report::{DeviceReport, Environment, Report};
use std::process::Command;

struct Args {
    device: Option<u32>,
    json: Option<String>,
    quick: bool,
    all_devices: bool,
}

fn usage() -> ! {
    eprintln!(
        "vk-probe — raw Vulkan micro-benchmarks (VRAM bandwidth, FMA, PCIe, overlap)\n\
         \n\
         Usage: vk-probe [--device N] [--json PATH] [--quick] [--all-devices]\n\
         \n\
         --device N       run only Vulkan device index N (as listed on startup)\n\
         --json PATH      write the full JSON report to PATH\n\
         --quick          smaller buffers / shorter timing windows\n\
         --all-devices    include the integrated GPU (default: discrete GPUs only)"
    );
    std::process::exit(2);
}

fn parse_args() -> Args {
    let mut args = Args {
        device: None,
        json: None,
        quick: false,
        all_devices: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--device" | "-d" => {
                let v = it.next().unwrap_or_else(|| {
                    eprintln!("--device requires a value");
                    std::process::exit(2);
                });
                args.device = Some(v.parse().unwrap_or_else(|_| {
                    eprintln!("--device expects an integer, got {v}");
                    std::process::exit(2);
                }));
            }
            "--json" | "-j" => {
                args.json = Some(it.next().unwrap_or_else(|| {
                    eprintln!("--json requires a path");
                    std::process::exit(2);
                }));
            }
            "--quick" | "-q" => args.quick = true,
            "--all-devices" | "-a" => args.all_devices = true,
            "--help" | "-h" => usage(),
            other => {
                eprintln!("unknown argument: {other}");
                usage();
            }
        }
    }
    args
}

fn host_cmd_line(args: &[String]) -> String {
    args.join(" ")
}

fn capture(program: &str, argv: &[&str]) -> String {
    match Command::new(program).args(argv).output() {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        _ => String::new(),
    }
}

fn competing_processes() -> Vec<String> {
    let out = Command::new("pgrep")
        .args(["-af", "llama-server|llama-bench|strata-iq|comfyui"])
        .output();
    match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect(),
        _ => Vec::new(),
    }
}

fn fmt_gib(b: u64) -> String {
    format!("{:.2} GiB", b as f64 / (1u64 << 30) as f64)
}

fn print_summary(rep: &DeviceReport) {
    let d = &rep.device;
    let p = &rep.properties;
    println!();
    println!(
        "=== Vulkan device {}: {} ===",
        d.index, d.name
    );
    println!(
        "    vendor {} device {} | type {} | api {} | driver {} ({}, {})",
        d.vendor_id, d.device_id, d.device_type, d.api_version, d.driver_version, d.driver_name, d.driver_info
    );
    println!("    memory heaps:");
    for h in &p.memory_heaps {
        println!(
            "      heap {}: {} {}",
            h.index,
            fmt_gib(h.size_bytes),
            if h.device_local { "device-local" } else { "host" }
        );
    }
    println!("    queue families:");
    for f in &p.queue_families {
        println!(
            "      family {}: {} | queues={} | timestampValidBits={}",
            f.index,
            f.flags.join("|"),
            f.queue_count,
            f.timestamp_valid_bits
        );
    }
    let s = &p.subgroup;
    println!(
        "    subgroup: size={} | stages=[{}] | ops=[{}] | quad_all_stages={}",
        s.subgroup_size,
        s.supported_stages.join(","),
        s.supported_operations.join(","),
        yn(s.quad_operations_in_all_stages)
    );
    if s.min_subgroup_size > 0 && s.max_subgroup_size > 0 && s.min_subgroup_size != s.max_subgroup_size {
        println!(
            "    subgroup size control: min={} max={} max_compute_workgroup_subgroups={}",
            s.min_subgroup_size, s.max_subgroup_size, s.max_compute_workgroup_subgroups
        );
    }
    let e = &p.extensions;
    println!(
        "    extensions: timeline_sema={} external_memory_host={} cooperative_matrix={} \
         subgroup_size_control={} shader_atomic_int64={} ({} total)",
        yn(e.khr_timeline_semaphore),
        yn(e.ext_external_memory_host),
        yn(e.khr_cooperative_matrix),
        yn(e.ext_subgroup_size_control),
        yn(e.khr_shader_atomic_int64),
        e.total_device_extensions
    );
    let f = &p.features;
    println!(
        "    features: float16={} int8={} atomic64(buf/shared)={}/{} timeline={} \
         subgroup_size_control={} compute_full_subgroups={} cooperative_matrix={}",
        yn(f.shader_float16),
        yn(f.shader_int8),
        yn(f.shader_atomic_int64_buffer),
        yn(f.shader_atomic_int64_shared),
        yn(f.timeline_semaphore),
        yn(f.subgroup_size_control),
        yn(f.compute_full_subgroups),
        yn(f.cooperative_matrix)
    );
    println!("    timestampPeriod = {:.1} ns/tick", p.timestamp_period_ns);

    if let Some(b) = &rep.vram_bandwidth_gbps {
        println!(
            "    VRAM bandwidth: copy {:.1} GB/s | read {:.1} GB/s | write {:.1} GB/s \
             (2 x {} MiB, {} windows of {:.3}s)",
            b.copy_gbps,
            b.read_gbps,
            b.write_gbps,
            b.buffer_bytes >> 20,
            b.iterations,
            b.seconds
        );
    }
    if let Some(b) = &rep.vram_bandwidth_cache_gbps {
        println!(
            "    VRAM bandwidth (cache-resident window): copy {:.1} GB/s \
             (2 x {} MiB, {} windows of {:.3}s)",
            b.copy_gbps,
            b.buffer_bytes >> 20,
            b.iterations,
            b.seconds
        );
    }
    if let Some(c) = &rep.compute_gflops {
        println!(
            "    compute: FP32 {:.0} GFLOPS | FP16 {} | {} threads x {} rounds x {} chains",
            c.fp32_gflops,
            c.fp16_gflops
                .map(|v| format!("{v:.0} GFLOPS"))
                .unwrap_or_else(|| "skipped".into()),
            c.threads,
            c.fma_rounds_per_dispatch,
            c.fma_chains
        );
        for n in &c.notes {
            println!("        note: {n}");
        }
    }
    if let Some(pc) = &rep.pcie {
        println!(
            "    PCIe (queue family {} index {}, staging HOST_VISIBLE|HOST_COHERENT):",
            pc.queue_family, pc.queue_index
        );
        for s in &pc.sizes {
            println!(
                "      {:>5} MiB: H2D {:.2} GB/s  D2H {:.2} GB/s",
                s.size_mb, s.h2d_gbps, s.d2h_gbps
            );
        }
    }
    if let Some(o) = &rep.overlap {
        println!(
            "    overlap: compute {:.3}s, transfer {:.3}s, both {:.3}s -> efficiency {:.1}% ({})",
            o.compute_seconds, o.transfer_seconds, o.both_seconds, o.efficiency_percent, o.queue_mode
        );
        println!("        {}", o.formula);
    }
    for n in &rep.notes {
        println!("    note: {n}");
    }
    if let Some(e) = &rep.error {
        println!("    ERROR: {e}");
    }
}

fn yn(b: bool) -> &'static str {
    if b {
        "yes"
    } else {
        "no"
    }
}

fn main() {
    let args = parse_args();
    let argv: Vec<String> = std::env::args().collect();
    let raw_args: Vec<String> = argv[1..].to_vec();

    let host = match Host::new() {
        Ok(h) => h,
        Err(e) => {
            eprintln!("vk-probe: {e}");
            std::process::exit(1);
        }
    };
    let devices = match host.devices() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("vk-probe: {e}");
            std::process::exit(1);
        }
    };
    if devices.is_empty() {
        eprintln!("vk-probe: no Vulkan physical devices found");
        std::process::exit(1);
    }

    println!(
        "vk-probe: instance API {} (MESA_VK_DEVICE_SELECT cleared if set)",
        api_version_string(host.api_version)
    );
    println!("discovered {} Vulkan device(s):", devices.len());
    for d in &devices {
        println!(
            "  [{}] {} — {} {:04x}, api {}, driver {}",
            d.index,
            d.name,
            hex_vendor(d.vendor_id),
            d.device_id,
            api_version_string(d.api_version),
            ctx::driver_version_string(d.driver_version)
        );
    }

    // Device selection
    let selected: Vec<u32> = if let Some(n) = args.device {
        if n as usize >= devices.len() {
            eprintln!("--device {n} out of range (0..{})", devices.len() - 1);
            std::process::exit(2);
        }
        vec![n]
    } else if args.all_devices {
        (0..devices.len() as u32).collect()
    } else {
        let discrete: Vec<u32> = devices
            .iter()
            .filter(|d| d.device_type == ash::vk::PhysicalDeviceType::DISCRETE_GPU)
            .map(|d| d.index)
            .collect();
        if discrete.is_empty() {
            devices
                .iter()
                .filter(|d| d.device_type != ash::vk::PhysicalDeviceType::INTEGRATED_GPU)
                .map(|d| d.index)
                .collect()
        } else {
            discrete
        }
    };
    if selected.is_empty() {
        eprintln!("vk-probe: no devices selected (try --all-devices)");
        std::process::exit(2);
    }
    println!(
        "running on device(s): {:?}{}{}",
        selected,
        if args.quick { " [quick mode]" } else { "" },
        if args.all_devices { " [--all-devices]" } else { "" }
    );

    let env = Environment {
        timestamp: capture("date", &["-Iseconds"]),
        hostname: capture("hostname", &[]),
        command: host_cmd_line(&raw_args),
        instance_api_version: api_version_string(host.api_version),
        quick_mode: args.quick,
        all_devices: args.all_devices,
        competing_gpu_processes: competing_processes(),
    };
    if !env.competing_gpu_processes.is_empty() {
        println!("processes alive at start (verify GPUs are idle):");
        for p in &env.competing_gpu_processes {
            println!("    {p}");
        }
    }

    let mut results: Vec<DeviceReport> = Vec::new();
    for &idx in &selected {
        let dev: &Dev = &devices[idx as usize];
        let gpu = match Gpu::create(&host, dev, idx) {
            Ok(g) => g,
            Err(e) => {
                let mut rep = report::DeviceReport {
                    device: bench::device_id(dev),
                    properties: bench::properties(dev),
                    notes: Vec::new(),
                    ..Default::default()
                };
                rep.error = Some(e);
                print_summary(&rep);
                results.push(rep);
                continue;
            }
        };
        let rep = bench::run_device(dev, &gpu, args.quick);
        print_summary(&rep);
        results.push(rep);
        gpu.destroy();
    }

    let report = Report {
        schema_version: 1,
        environment: env,
        results,
    };

    if let Some(path) = &args.json {
        match std::fs::File::create(path) {
            Ok(f) => match serde_json::to_writer_pretty(f, &report) {
                Ok(()) => println!("wrote {path}"),
                Err(e) => eprintln!("failed to write {path}: {e}"),
            },
            Err(e) => eprintln!("failed to create {path}: {e}"),
        }
    }

    host.destroy();

    let any_ok = report
        .results
        .iter()
        .any(|r| r.vram_bandwidth_gbps.is_some() || r.compute_gflops.is_some() || r.pcie.is_some());
    if !any_ok {
        eprintln!("vk-probe: no benchmark completed successfully");
        std::process::exit(1);
    }
}

fn hex_vendor(v: u32) -> String {
    format!("0x{v:04x}")
}
