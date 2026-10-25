# Project setup

Requirements, frameworks, and commands for building and measuring Breccia. Machine: 7950X (32T), 58 GB RAM, RX 6950 XT (gfx1030) + MI50 (gfx906), Mesa RADV 26.2.4, ROCm 7.2, 121 GB free NVMe.

## Toolchain (verified present on this machine)

| Tool | Version | Needed for |
|---|---|---|
| rustc / cargo | 1.95.0 | engine + probe harnesses |
| glslc, glslangValidator, spirv-val | (shaderc, Vulkan SDK) | SPIR-V shader compile + validation in `build.rs` |
| vulkan headers | 1.4.363 | `ash` bindings, probe harness |
| cmake / ninja | system | llama.cpp reference builds, C harnesses |
| hipcc (ROCm) | 7.2.53211 | HIP probes, baseline llama.cpp comparisons |
| clang++ / g++ | 22 / 16 | small C harnesses |
| liburing | 2.15 | Tier-3 NVMe direct I/O (io_uring) |
| git | system | reference clones |

Missing-but-likely-needed (install on first use): `rustup component clippy rustfmt` (if absent), `pkg-config`, `vulkan-validationlayers` (already present as a layer).

## Framework choices (locked)

Per the plan's language analysis (C++20/23 vs Go vs Rust):

| Concern | Crate / tool | Rationale |
|---|---|---|
| Vulkan bindings | `ash` | explicit 1:1 API control; timeline semaphores, descriptor sets without abstraction tax |
| Buffer memory | `gpu-allocator` | proven sub-allocation over VMA-equivalent pools |
| Async runtime | `tokio` | agent loops, scheduler, server |
| NVMe direct I/O | `tokio-uring` / raw `io_uring` (liburing 2.15) | Tier-3 KV/expert paging without page-cache thrash |
| Shader pipeline | GLSL → SPIR-V via `glslc` in `build.rs`, validated with `spirv-val`; specialization constants for Wave32/Wave64 | matches existing system tooling; Slang evaluation deferred |
| GGUF parsing | own zero-copy `breccia-format` (mmap) | split-residency loading of dense / expert / PLE tensors |
| HTTP server | `axum` | OpenAI-compatible endpoints, OpenTelemetry-ish `/metrics` |
| Fingerprints | `blake3` | hardware profile invalidation |

Deliberately **not** using: heavy engine frameworks (we are the engine), or `vulkano` (too much wrapper for kernel-level work).

## Build & test commands

```bash
# Rust workspace (from repo root) — Phase 1+
cargo build --workspace
cargo test  --workspace
cargo fmt --check && cargo clippy --workspace -- -D warnings

# Shader compile + validation (harnesses and crates share this)
glslc -O -fauto-bind-reflect --target-env=vulkan1.3 shaders/foo.comp -o foo.spv
spirv-val foo.spv

# Phase-0 probes (harnesses/ — build instructions in each harness's README)
# see docs/phase-0/ for the exact benchmark command lines used
```

## Machine-specific hazards (learned the hard way — see docs/phase-0/)

| Hazard | Effect | Workaround (encoded in `harnesses/bench_attrib.sh`) |
|---|---|---|
| `MESA_VK_DEVICE_SELECT=1002:164e!` in session env | Hides both discrete GPUs from **every** Vulkan process → silent iGPU-only runs (Vulkan 8× slower, llama-bench still reports success) | `env -u MESA_VK_DEVICE_SELECT` for any Vulkan command; `vulkaninfo` sanity check first |
| korphaus not in `gpu-compute` group (`/dev/kfd` is root:gpu-compute) | HIP/HSA init silently fails → CPU fallback **mislabelled `ROCm`** by llama-bench | run HIP tools via `sudo -n` (passwordless), or through systemd units (they add `SupplementaryGroups=gpu-compute`) |
| Setting `CUDA_VISIBLE_DEVICES`+`HIP_VISIBLE_DEVICES`+`ROCR_VISIBLE_DEVICES` together | Double-masking → HIP init fails outright (index 1 of a 1-device set is empty) | set **only** `CUDA_VISIBLE_DEVICES` |
| `llama-bench -d N` does not isolate a device | Layer-split stays active across all visible GPUs; results identical for every `-d` | pin with visibility env, confirm with busy% sampling |
| `ulimit -l` = 8 MiB | Any pinned (Tier-2) buffer pool fails at alloc time | units need `LimitMEMLOCK=infinity` (Strata precedent) |
| gpu-manager monitor runs even when its unit is inactive | Stops idle GPU services when `MemAvailable < 4 GiB` — benchmark servers get evicted mid-window | keep requests in flight while measuring; restore service state after windows |
| GPU DPM clock states | Same run at boosted vs idle clocks differs up to ~1.6× (273 vs 174 tok/s) | warm up / pin clocks for comparable numbers |

## Reference builds (baselines — do not rebuild without reason)

| Build | Path | Backend |
|---|---|---|
| llama.cpp qwen4exp HIP | `/home/korphaus/llama.cpp-qwen4exp-mmap/build-hip-q5kv/bin/llama-server` | ROCm, used by `qwen-dual.service` |
| llama.cpp qwen4exp Vulkan | `/home/korphaus/llama.cpp-qwen4exp-mmap/build-vulkan/bin/llama-server` | RADV — the 2.56 tok/s data point |
| Strata engine | `/home/korphaus/projects/strata` (`strata-iq3_xxs.json`) | HIP |
| distro llama.cpp | `/usr/bin/llama-server` (libggml-hip) | ROCm (init currently fails — tracked in phase-0) |

## Conventions

- New engine code lives in `crates/<name>/`; harnesses stay out of the workspace in `harnesses/` when they are one-off.
- All measurement code writes machine-readable results (JSON) plus a short human summary in `docs/phase-*/`.
- Read-only inputs: `/srv/strata-data/**` and `Reference Projects/**`. No writes, no moves, no copies of weight files (disk is 85% full).
