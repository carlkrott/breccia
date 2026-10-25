# vk-probe — Phase 0.2 raw Vulkan micro-benchmark harness

Headless (compute-only) Vulkan probe used by Breccia's Phase 0.2 backend experiment.
Measures, per physical device:

1. **Device discovery** — IDs, API/driver version, memory heaps, queue families,
   `VkPhysicalDeviceSubgroupProperties` (+ subgroup-size-control range), and the
   presence of `VK_KHR_timeline_semaphore`, `VK_EXT_external_memory_host`,
   `VK_KHR_cooperative_matrix`, `VK_EXT_subgroup_size_control`,
   `VK_KHR_shader_atomic_int64`.
2. **VRAM bandwidth** — strided vec4 read+write kernel over two device-local
   buffers, timed with queue timestamps (`vkCmdWriteTimestamp` +
   `vkGetQueryPoolResults(TYPE_64|WAIT)`).
3. **FMA throughput** — 8 independent dependent `fma()` chains per thread,
   fp32 always, fp16 when `shaderFloat16` + `storageBuffer16BitAccess` are
   supported.
4. **PCIe H2D/D2H** — `vkCmdCopyBuffer` between a `HOST_VISIBLE|HOST_COHERENT`
   staging buffer and a device-local buffer at 16/64/256/1024 MiB.
5. **Compute/transfer overlap** — sustained FMA on the compute queue and a
   back-to-back H2D+D2H loop on a (separate) transfer queue, submitted
   concurrently; `eff = 1 - (T_both - max(Tc,Tt)) / min(Tc,Tt)`.

## Build

```bash
cd harnesses/vk-probe
cargo build --release
```

`build.rs` compiles `shaders/*.comp` with `glslc --target-env=vulkan1.3` into
`OUT_DIR` and validates each module with `spirv-val`; the build fails if either
tool rejects a shader.

## Run

```bash
# default: all discrete GPUs (skips the integrated iGPU)
cargo run --release -- --json results.json

# options
cargo run --release -- --device 1        # a single Vulkan device index
cargo run --release -- --quick           # smaller buffers, shorter timing windows
cargo run --release -- --all-devices     # include the integrated GPU
```

Machine note: this environment exports `MESA_VK_DEVICE_SELECT=1002:164e!`, which
pins enumeration to the iGPU only. The harness unsets it at startup and prints a
warning; run with `env -u MESA_VK_DEVICE_SELECT` if invoking `vulkaninfo` etc.

## Validation

```bash
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation cargo run --release -- --quick
```

Latest run: **0 validation errors** (both discrete GPUs).

## Output

Human summary on stdout; `--json PATH` writes:

```json
{
  "schema_version": 1,
  "environment": { "timestamp": ..., "competing_gpu_processes": [...] },
  "results": [
    {
      "device": { "index": 1, "name": ..., "vendor_id": "0x1002", "device_id": "0x73a5", ... },
      "properties": { "memory_heaps": [...], "queue_families": [...],
                      "subgroup": {...}, "extensions": {...}, "features": {...} },
      "vram_bandwidth_gbps": { "copy_gbps": ..., "read_gbps": ..., "write_gbps": ... },
      "compute_gflops":     { "fp32_gflops": ..., "fp16_gflops": ... },
      "pcie":               { "sizes": [ { "size_mb": 16, "h2d_gbps": ..., "d2h_gbps": ... } ] },
      "overlap":            { "compute_seconds": ..., "transfer_seconds": ...,
                              "both_seconds": ..., "efficiency_percent": ... },
      "notes": [...], "error": null
    }
  ]
}
```

A sub-test that cannot run on a device is `null` and carries a reason in `notes`.

## Measurement caveats

- Bandwidth `copy_gbps` = (bytes read + bytes written) / time, i.e. 2 x buffer
  size per pass; `read_gbps`/`write_gbps` are the per-direction halves.
- FMA counts each `fma()` as 2 flops. The chains are dependency-chained, so the
  number is "achievable sustained FMA with ILP = 8", not a theoretical peak.
  fp16 reports binary16 ops as issued (pair/packed math counts 2 ops).
- PCIe staging is `vkAllocateMemory`'d host memory: the driver page-locks
  (pins) it while device-visible — the Vulkan analogue of registered host memory,
  not ordinary pageable `malloc`.
- Overlap timing is wall-clock around two fence waits; the two legs run on
  separate `VkQueue`s when the device exposes a second transfer-capable family.
