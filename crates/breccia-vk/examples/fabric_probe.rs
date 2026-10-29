//! Single-card fabric probe: sparse binding, VK_KHR_copy_memory_indirect and the
//! texture-unit format features, measured on one device (default the RX).
//!
//! `cargo run --release -p breccia-vk --example fabric_probe -- [pci]`
//! Run under `timeout 90` with the validation layer; it never touches a second card.

use ash::vk;
use breccia_vk::ctx::{find_mem_type, vk as chk, Host, R};
use std::ffi::c_void;
use std::time::Instant;

// VK_KHR_copy_memory_indirect is newer than ash 0.38: hand-written from vulkan_core.h.
const ST_INDIRECT_FEATS: i32 = 1000549000;
const ST_INDIRECT_PROPS: i32 = 1000426001;
const ST_INDIRECT_INFO: i32 = 1000549002;
const ADDRESS_COPY_DEVICE_LOCAL: u32 = 1;

#[repr(C)]
struct IndirectFeats {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    indirect_memory_copy: vk::Bool32,
    indirect_memory_to_image_copy: vk::Bool32,
}
#[repr(C)]
struct IndirectProps {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    supported_queues: vk::QueueFlags,
}
#[repr(C)]
struct StridedRange {
    address: u64,
    size: u64,
    stride: u64,
}
#[repr(C)]
struct IndirectInfo {
    s_type: vk::StructureType,
    p_next: *const c_void,
    src_copy_flags: u32,
    dst_copy_flags: u32,
    copy_count: u32,
    copy_address_range: StridedRange,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct CopyCmd {
    src: u64,
    dst: u64,
    size: u64,
}
type CmdCopyIndirect = unsafe extern "system" fn(vk::CommandBuffer, *const IndirectInfo);

/// Page-aligned host allocation, zeroed.
struct HostPages {
    ptr: *mut u8,
    layout: std::alloc::Layout,
}
impl HostPages {
    fn new(bytes: u64, align: u64) -> HostPages {
        let layout = std::alloc::Layout::from_size_align(bytes as usize, align as usize).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "host alloc of {bytes} bytes");
        HostPages { ptr, layout }
    }
    fn words(&self) -> &mut [u32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr as *mut u32, self.layout.size() / 4) }
    }
}
impl Drop for HostPages {
    fn drop(&mut self) {
        unsafe { std::alloc::dealloc(self.ptr, self.layout) }
    }
}

struct Ctx {
    host: Host,
    dev: ash::Device,
    mem: vk::PhysicalDeviceMemoryProperties,
    queue: vk::Queue,
    family: u32,
    pool: vk::CommandPool,
    qpool: vk::QueryPool,
    ts_ns: f64,
    hostptr: ash::ext::external_memory_host::Device,
}

impl Ctx {
    fn buffer(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        flags: vk::BufferCreateFlags,
        external: bool,
    ) -> R<vk::Buffer> {
        let mut ext = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
        let mut bci = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .flags(flags)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        if external {
            bci = bci.push_next(&mut ext);
        }
        unsafe { chk(self.dev.create_buffer(&bci, None), "vkCreateBuffer") }
    }

    fn alloc(&self, size: u64, ty: u32, bda: bool, import: Option<*mut u8>) -> R<vk::DeviceMemory> {
        let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut imp = vk::ImportMemoryHostPointerInfoEXT::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
        let mut mai = vk::MemoryAllocateInfo::default().allocation_size(size).memory_type_index(ty);
        if bda {
            mai = mai.push_next(&mut flags);
        }
        if let Some(p) = import {
            imp = imp.host_pointer(p as *mut c_void);
            mai = mai.push_next(&mut imp);
        }
        unsafe { chk(self.dev.allocate_memory(&mai, None), "vkAllocateMemory") }
    }

    /// Plain buffer + dedicated memory of the first type in `want`.
    fn plain(&self, size: u64, usage: vk::BufferUsageFlags, want: vk::MemoryPropertyFlags) -> R<(vk::Buffer, vk::DeviceMemory)> {
        let bda = usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
        let b = self.buffer(size, usage, vk::BufferCreateFlags::empty(), false)?;
        let reqs = unsafe { self.dev.get_buffer_memory_requirements(b) };
        let ty = find_mem_type(&self.mem, reqs.memory_type_bits, want, want).ok_or("no memory type")?;
        let m = self.alloc(reqs.size, ty, bda, None)?;
        unsafe { chk(self.dev.bind_buffer_memory(b, m, 0), "vkBindBufferMemory")? };
        Ok((b, m))
    }

    /// Host range imported as a buffer (what the D3 arena does).
    fn imported(&self, p: &HostPages, usage: vk::BufferUsageFlags) -> R<(vk::Buffer, vk::DeviceMemory)> {
        let size = p.layout.size() as u64;
        let bits = self.host_bits(p.ptr)?;
        let b = self.buffer(size, usage, vk::BufferCreateFlags::empty(), true)?;
        let reqs = unsafe { self.dev.get_buffer_memory_requirements(b) };
        let ty = find_mem_type(&self.mem, bits & reqs.memory_type_bits, vk::MemoryPropertyFlags::empty(), vk::MemoryPropertyFlags::HOST_VISIBLE)
            .ok_or("no common import type")?;
        let m = self.alloc(size, ty, usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS), Some(p.ptr))?;
        unsafe { chk(self.dev.bind_buffer_memory(b, m, 0), "vkBindBufferMemory(import)")? };
        Ok((b, m))
    }

    fn host_bits(&self, p: *mut u8) -> R<u32> {
        let mut props = vk::MemoryHostPointerPropertiesEXT::default();
        unsafe {
            (self.hostptr.fp().get_memory_host_pointer_properties_ext)(
                self.dev.handle(),
                vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT,
                p as *const c_void,
                &mut props,
            )
            .result()
            .map_err(|e| format!("host pointer props: {e}"))?;
        }
        Ok(props.memory_type_bits)
    }

    fn addr(&self, b: vk::Buffer) -> u64 {
        unsafe { self.dev.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(b)) }
    }

    /// Record, submit, wait. Returns GPU ns between the two timestamps around `f`.
    fn once(&self, f: impl FnOnce(vk::CommandBuffer)) -> R<f64> {
        unsafe {
            let cb = chk(
                self.dev.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default().command_pool(self.pool).command_buffer_count(1),
                ),
                "vkAllocateCommandBuffers",
            )?[0];
            chk(self.dev.begin_command_buffer(cb, &vk::CommandBufferBeginInfo::default()), "begin")?;
            self.dev.cmd_reset_query_pool(cb, self.qpool, 0, 2);
            self.dev.cmd_write_timestamp(cb, vk::PipelineStageFlags::ALL_COMMANDS, self.qpool, 0);
            f(cb);
            self.dev.cmd_write_timestamp(cb, vk::PipelineStageFlags::ALL_COMMANDS, self.qpool, 1);
            chk(self.dev.end_command_buffer(cb), "end")?;
            let fence = chk(self.dev.create_fence(&vk::FenceCreateInfo::default(), None), "fence")?;
            let cbs = [cb];
            chk(self.dev.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cbs)], fence), "submit")?;
            chk(self.dev.wait_for_fences(&[fence], true, 10_000_000_000), "wait")?;
            self.dev.destroy_fence(fence, None);
            self.dev.free_command_buffers(self.pool, &cbs);
            let mut ts = [0u64; 2];
            chk(
                self.dev.get_query_pool_results(self.qpool, 0, &mut ts, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT),
                "query",
            )?;
            Ok(ts[1].wrapping_sub(ts[0]) as f64 * self.ts_ns)
        }
    }

    fn barrier(&self, cb: vk::CommandBuffer) {
        let mb = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        unsafe {
            self.dev.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[mb],
                &[],
                &[],
            )
        }
    }

    /// vkQueueBindSparse + fence wait, wall-clock µs (call return, signalled).
    fn bind(&self, buf: vk::Buffer, binds: &[vk::SparseMemoryBind]) -> R<(f64, f64)> {
        unsafe {
            let fence = chk(self.dev.create_fence(&vk::FenceCreateInfo::default(), None), "fence")?;
            let bb = [vk::SparseBufferMemoryBindInfo::default().buffer(buf).binds(binds)];
            let info = vk::BindSparseInfo::default().buffer_binds(&bb);
            let t0 = Instant::now();
            chk(self.dev.queue_bind_sparse(self.queue, &[info], fence), "vkQueueBindSparse")?;
            let t1 = Instant::now();
            chk(self.dev.wait_for_fences(&[fence], true, 10_000_000_000), "wait(bind)")?;
            let t2 = Instant::now();
            self.dev.destroy_fence(fence, None);
            Ok(((t1 - t0).as_secs_f64() * 1e6, (t2 - t0).as_secs_f64() * 1e6))
        }
    }

    fn read_words(&self, src: vk::Buffer, off: u64, words: usize) -> R<Vec<u32>> {
        let bytes = (words * 4) as u64;
        let (rb, rm) = self.plain(
            bytes,
            vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        self.once(|cb| unsafe {
            self.dev.cmd_copy_buffer(cb, src, rb, &[vk::BufferCopy { src_offset: off, dst_offset: 0, size: bytes }])
        })?;
        let out = unsafe {
            let p = chk(self.dev.map_memory(rm, 0, bytes, vk::MemoryMapFlags::empty()), "map")? as *const u32;
            let v = std::slice::from_raw_parts(p, words).to_vec();
            self.dev.unmap_memory(rm);
            self.dev.destroy_buffer(rb, None);
            self.dev.free_memory(rm, None);
            v
        };
        Ok(out)
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let pci = std::env::args().nth(1).unwrap_or_else(|| "0000:03:00.0".into());
    if let Err(e) = run(&pci) {
        eprintln!("fabric_probe: {e}");
        std::process::exit(1);
    }
}

fn run(pci: &str) -> R<()> {
    let host = Host::new()?;
    let d = host.devices()?.into_iter().find(|d| d.pci_address == pci).ok_or(format!("no device {pci}"))?;
    println!("device {} ({}) {}", d.name, d.pci_address, d.driver_info);
    for (i, f) in d.families.iter().enumerate() {
        println!("  family {i}: {:?} x{} ts_bits {}", f.queue_flags, f.queue_count, f.timestamp_valid_bits);
    }

    // ---- capabilities -------------------------------------------------------
    let mut ifeat = IndirectFeats {
        s_type: vk::StructureType::from_raw(ST_INDIRECT_FEATS),
        p_next: std::ptr::null_mut(),
        indirect_memory_copy: 0,
        indirect_memory_to_image_copy: 0,
    };
    let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
    v12.p_next = &mut ifeat as *mut _ as *mut c_void;
    let mut f2 = vk::PhysicalDeviceFeatures2::default();
    f2.p_next = &mut v12 as *mut _ as *mut c_void;
    unsafe { host.instance.get_physical_device_features2(d.pdev, &mut f2) };
    let has_indirect = d.has_ext("VK_KHR_copy_memory_indirect") && ifeat.indirect_memory_copy != 0;

    let mut iprops = IndirectProps {
        s_type: vk::StructureType::from_raw(ST_INDIRECT_PROPS),
        p_next: std::ptr::null_mut(),
        supported_queues: vk::QueueFlags::empty(),
    };
    let mut minmax = vk::PhysicalDeviceSamplerFilterMinmaxProperties::default();
    minmax.p_next = &mut iprops as *mut _ as *mut c_void;
    let mut p2 = vk::PhysicalDeviceProperties2::default();
    p2.p_next = &mut minmax as *mut _ as *mut c_void;
    unsafe { host.instance.get_physical_device_properties2(d.pdev, &mut p2) };
    let sp = p2.properties.sparse_properties;
    println!(
        "sparse: binding {} residencyBuffer {} aliased {} | residencyStandard2DBlockShape {} nonResidentStrict {} alignedMipSize {}",
        f2.features.sparse_binding, f2.features.sparse_residency_buffer, f2.features.sparse_residency_aliased,
        sp.residency_standard2_d_block_shape, sp.residency_non_resident_strict, sp.residency_aligned_mip_size
    );
    println!(
        "copy_memory_indirect: ext {} indirectMemoryCopy {} toImage {} supportedQueues {:?}",
        d.has_ext("VK_KHR_copy_memory_indirect"), ifeat.indirect_memory_copy, ifeat.indirect_memory_to_image_copy, iprops.supported_queues
    );
    println!(
        "minmax sampler: singleComponent {} imageComponentMapping {} | maxSamplerAnisotropy {} | maxImageDimension2D {} maxTexelBufferElements {}",
        minmax.filter_minmax_single_component_formats, minmax.filter_minmax_image_component_mapping,
        p2.properties.limits.max_sampler_anisotropy, p2.properties.limits.max_image_dimension2_d,
        p2.properties.limits.max_texel_buffer_elements
    );
    format_table(&host, &d);

    // External-host import into a SPARSE buffer: allowed at all?
    let mut ebp = vk::ExternalBufferProperties::default();
    unsafe {
        host.instance.get_physical_device_external_buffer_properties(
            d.pdev,
            &vk::PhysicalDeviceExternalBufferInfo::default()
                .flags(vk::BufferCreateFlags::SPARSE_BINDING)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER)
                .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT),
            &mut ebp,
        )
    };
    let sparse_import_ok = ebp.external_memory_properties.external_memory_features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE);
    println!("sparse buffer + HOST_ALLOCATION import: features {:?}", ebp.external_memory_properties.external_memory_features);

    // ---- device: one compute queue that also binds sparse ------------------
    let family = d
        .families
        .iter()
        .position(|f| {
            f.queue_flags.contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::SPARSE_BINDING)
                && !f.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && f.timestamp_valid_bits > 0
        })
        .ok_or("no compute+sparse family")? as u32;
    let prio = [1.0f32];
    let qci = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&prio)];
    let mut exts = vec![c"VK_EXT_external_memory_host".as_ptr()];
    if has_indirect {
        exts.push(c"VK_KHR_copy_memory_indirect".as_ptr());
    }
    let mut en_ind = IndirectFeats {
        s_type: vk::StructureType::from_raw(ST_INDIRECT_FEATS),
        p_next: std::ptr::null_mut(),
        indirect_memory_copy: has_indirect as u32,
        indirect_memory_to_image_copy: 0,
    };
    let mut en12 = vk::PhysicalDeviceVulkan12Features::default()
        .buffer_device_address(true)
        .sampler_filter_minmax(v12.sampler_filter_minmax != 0);
    if has_indirect {
        en12.p_next = &mut en_ind as *mut _ as *mut c_void;
    }
    let mut en2 = vk::PhysicalDeviceFeatures2::default()
        .features(vk::PhysicalDeviceFeatures::default().sparse_binding(f2.features.sparse_binding != 0));
    en2.p_next = &mut en12 as *mut _ as *mut c_void;
    let mut dci = vk::DeviceCreateInfo::default().queue_create_infos(&qci).enabled_extension_names(&exts);
    dci.p_next = &mut en2 as *mut _ as *mut c_void;
    let dev = unsafe { chk(host.instance.create_device(d.pdev, &dci, None), "vkCreateDevice")? };
    let queue = unsafe { dev.get_device_queue(family, 0) };
    let pool = unsafe {
        chk(
            dev.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(family).flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            ),
            "pool",
        )?
    };
    let qpool = unsafe {
        chk(dev.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2), None), "qpool")?
    };
    let hostptr = ash::ext::external_memory_host::Device::new(&host.instance, &dev);
    let c = Ctx { mem: d.mem, queue, family, pool, qpool, ts_ns: d.timestamp_period_ns as f64, hostptr, dev, host };
    println!("using queue family {}", c.family);

    if f2.features.sparse_binding != 0 {
        sparse(&c, sparse_import_ok)?;
    }
    if has_indirect && iprops.supported_queues.contains(vk::QueueFlags::COMPUTE) {
        indirect(&c)?;
    } else {
        println!("indirect copy: skipped (not on a compute queue here)");
    }
    if v12.sampler_filter_minmax != 0 {
        tmu(&c)?;
    }
    submit_cost(&c)?;
    unsafe {
        c.dev.device_wait_idle().ok();
        c.dev.destroy_query_pool(c.qpool, None);
        c.dev.destroy_command_pool(c.pool, None);
        c.dev.destroy_device(None);
    }
    Ok(())
}

fn format_table(host: &Host, d: &breccia_vk::ctx::Dev) {
    use vk::Format as F;
    use vk::FormatFeatureFlags as X;
    let fmts = [
        F::R8_UNORM, F::R8_SNORM, F::R8_UINT, F::R16_UNORM, F::R16_SNORM, F::R16_SFLOAT, F::R32_SFLOAT, F::R32_UINT,
        F::R8G8B8A8_UNORM, F::R8G8B8A8_SNORM, F::R16G16B16A16_SFLOAT, F::R32G32B32A32_SFLOAT, F::A2B10G10R10_UNORM_PACK32,
        F::B10G11R11_UFLOAT_PACK32, F::E5B9G9R9_UFLOAT_PACK32, F::BC4_UNORM_BLOCK, F::BC4_SNORM_BLOCK, F::BC5_UNORM_BLOCK,
        F::R64_UINT,
    ];
    let cols: [(&str, X); 6] = [
        ("sampled", X::SAMPLED_IMAGE),
        ("linear", X::SAMPLED_IMAGE_FILTER_LINEAR),
        ("minmax", X::SAMPLED_IMAGE_FILTER_MINMAX),
        ("storage", X::STORAGE_IMAGE),
        ("st-atomic", X::STORAGE_IMAGE_ATOMIC),
        ("blit-dst", X::BLIT_DST),
    ];
    println!("format features (optimal tiling | buffer: uniform-texel storage-texel):");
    for f in fmts {
        let p = unsafe { host.instance.get_physical_device_format_properties(d.pdev, f) };
        let o: Vec<&str> = cols.iter().filter(|(_, b)| p.optimal_tiling_features.contains(*b)).map(|(n, _)| *n).collect();
        println!(
            "  {:<28} {:<48} | utb {} stb {}",
            format!("{f:?}"),
            o.join(","),
            p.buffer_features.contains(X::UNIFORM_TEXEL_BUFFER) as u8,
            p.buffer_features.contains(X::STORAGE_TEXEL_BUFFER) as u8
        );
    }
}

fn sparse(c: &Ctx, import_ok: bool) -> R<()> {
    const N: u64 = 256;
    let usage = vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER;
    // Probe the page size with a 1-page sparse buffer first.
    let probe = c.buffer(1 << 16, usage, vk::BufferCreateFlags::SPARSE_BINDING, import_ok)?;
    let page = unsafe { c.dev.get_buffer_memory_requirements(probe).alignment };
    unsafe { c.dev.destroy_buffer(probe, None) };
    let size = N * page;
    let sb = c.buffer(size, usage, vk::BufferCreateFlags::SPARSE_BINDING, import_ok)?;
    let reqs = unsafe { c.dev.get_buffer_memory_requirements(sb) };
    println!("sparse buffer: page {} KiB, {} pages, memoryTypeBits {:#x}", page >> 10, N, reqs.memory_type_bits);
    let ty = find_mem_type(&c.mem, reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .ok_or("no VRAM type for sparse")?;
    let vram = c.alloc(reqs.size, ty, false, None)?;
    let binds = |n: u64, mem: vk::DeviceMemory| -> Vec<vk::SparseMemoryBind> {
        (0..n)
            .map(|i| vk::SparseMemoryBind { resource_offset: i * page, size: page, memory: mem, memory_offset: i * page, flags: vk::SparseMemoryBindFlags::empty() })
            .collect()
    };
    for n in [1u64, 16, 256] {
        let (mut call, mut done) = (vec![], vec![]);
        for _ in 0..20 {
            let (a, b) = c.bind(sb, &binds(n, vram))?;
            call.push(a);
            done.push(b);
        }
        println!("  bind {n:>3} pages: call {:.1} us, signalled {:.1} us (median of 20)", median(call), median(done));
    }
    // Unbind cost: memory = null.
    let unb: Vec<f64> = (0..20).map(|_| c.bind(sb, &binds(1, vk::DeviceMemory::null())).map(|x| x.1)).collect::<R<_>>()?;
    println!("  unbind   1 page : signalled {:.1} us", median(unb));
    c.bind(sb, &binds(N, vram))?;

    const A: u32 = 0xAAAA_AAAA;
    c.once(|cb| unsafe { c.dev.cmd_fill_buffer(cb, sb, 0, size, A) })?;
    let w = c.read_words(sb, 0, (3 * page / 4) as usize)?;
    assert!(w.iter().all(|&x| x == A), "VRAM-backed sparse fill readback");
    println!("  VRAM-backed fill + readback: OK");

    if !import_ok {
        println!("  host page remap: skipped (driver does not report HOST_ALLOCATION importable for sparse buffers)");
        return free_sparse(c, sb, vram);
    }
    let hp = HostPages::new(page, page);
    let bits = c.host_bits(hp.ptr)?;
    let common = bits & reqs.memory_type_bits;
    println!("  host-pointer type bits {bits:#x} & sparse bits {:#x} = {common:#x}", reqs.memory_type_bits);
    let Some(hty) = find_mem_type(&c.mem, common, vk::MemoryPropertyFlags::empty(), vk::MemoryPropertyFlags::HOST_VISIBLE) else {
        println!("  host page remap: skipped (no common memory type)");
        return free_sparse(c, sb, vram);
    };
    for (i, x) in hp.words().iter_mut().enumerate() {
        *x = (i as u32) ^ 0x00C0_FFEE;
    }
    let hmem = c.alloc(page, hty, false, Some(hp.ptr))?;
    let one = |m: vk::DeviceMemory, off: u64| [vk::SparseMemoryBind { resource_offset: page, size: page, memory: m, memory_offset: off, flags: vk::SparseMemoryBindFlags::empty() }];
    let remap: Vec<f64> = (0..20).map(|_| c.bind(sb, &one(hmem, 0)).map(|x| x.1)).collect::<R<_>>()?;
    println!("  remap 1 page VRAM->host: signalled {:.1} us (median of 20)", median(remap));
    let pw = (page / 4) as usize;
    let w = c.read_words(sb, 0, 3 * pw)?;
    let mid_ok = w[pw..2 * pw].iter().enumerate().all(|(i, &x)| x == (i as u32) ^ 0x00C0_FFEE);
    let edges_ok = w[..pw].iter().chain(&w[2 * pw..]).all(|&x| x == A);
    println!("  GPU reads host page through sparse VA: middle {} edges {}", ok(mid_ok), ok(edges_ok));
    c.once(|cb| unsafe { c.dev.cmd_fill_buffer(cb, sb, page, page, 0x1111_1111) })?;
    let host_sees = hp.words().iter().all(|&x| x == 0x1111_1111);
    println!("  GPU write via sparse VA lands in host page: {}", ok(host_sees));
    c.bind(sb, &one(vram, page))?;
    let w = c.read_words(sb, page, pw)?;
    println!("  rebind to VRAM restores old VRAM page: {}", ok(w.iter().all(|&x| x == A)));
    unsafe { c.dev.free_memory(hmem, None) };
    free_sparse(c, sb, vram)
}

fn free_sparse(c: &Ctx, sb: vk::Buffer, vram: vk::DeviceMemory) -> R<()> {
    unsafe {
        c.dev.queue_wait_idle(c.queue).ok();
        c.dev.destroy_buffer(sb, None);
        c.dev.free_memory(vram, None);
    }
    Ok(())
}

fn ok(b: bool) -> &'static str {
    if b { "OK" } else { "FAIL" }
}

fn indirect(c: &Ctx) -> R<()> {
    // Gather 10 scattered 1.75 MiB "experts" out of a 40-slot host arena into VRAM.
    const CHUNK: u64 = 1792 << 10;
    const SLOTS: u64 = 40;
    const K: usize = 10;
    let pick: [u64; K] = [37, 2, 19, 11, 30, 5, 24, 8, 33, 16];
    let arena = HostPages::new(CHUNK * SLOTS, 1 << 16);
    for (i, x) in arena.words().iter_mut().enumerate() {
        *x = (i as u32).wrapping_mul(2_654_435_761);
    }
    let bda = vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
    let (src, src_m) = c.imported(&arena, vk::BufferUsageFlags::TRANSFER_SRC | bda)?;
    let (dst, dst_m) = c.plain(
        CHUNK * K as u64,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC | bda,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let cmd_bytes = (K * std::mem::size_of::<CopyCmd>()) as u64;
    let (cmds, cmds_m) = c.plain(
        cmd_bytes,
        vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | bda,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    let (sa, da, ca) = (c.addr(src), c.addr(dst), c.addr(cmds));
    let list: Vec<CopyCmd> = pick.iter().enumerate().map(|(j, &s)| CopyCmd { src: sa + s * CHUNK, dst: da + j as u64 * CHUNK, size: CHUNK }).collect();
    unsafe {
        let p = chk(c.dev.map_memory(cmds_m, 0, cmd_bytes, vk::MemoryMapFlags::empty()), "map(cmds)")? as *mut CopyCmd;
        std::ptr::copy_nonoverlapping(list.as_ptr(), p, K);
        c.dev.unmap_memory(cmds_m);
    }
    let name = c"vkCmdCopyMemoryIndirectKHR";
    let fp = unsafe { (c.host.instance.fp_v1_0().get_device_proc_addr)(c.dev.handle(), name.as_ptr()) }.ok_or("no vkCmdCopyMemoryIndirectKHR")?;
    let copy_ind: CmdCopyIndirect = unsafe { std::mem::transmute(fp) };
    let info = IndirectInfo {
        s_type: vk::StructureType::from_raw(ST_INDIRECT_INFO),
        p_next: std::ptr::null(),
        src_copy_flags: 0,
        dst_copy_flags: ADDRESS_COPY_DEVICE_LOCAL,
        copy_count: K as u32,
        copy_address_range: StridedRange { address: ca, size: cmd_bytes, stride: std::mem::size_of::<CopyCmd>() as u64 },
    };
    let regions: Vec<vk::BufferCopy> = pick.iter().enumerate().map(|(j, &s)| vk::BufferCopy { src_offset: s * CHUNK, dst_offset: j as u64 * CHUNK, size: CHUNK }).collect();
    let verify = || -> R<bool> {
        let got = c.read_words(dst, 0, (CHUNK * K as u64 / 4) as usize)?;
        let w = arena.words();
        let cw = (CHUNK / 4) as usize;
        Ok(pick.iter().enumerate().all(|(j, &s)| got[j * cw..(j + 1) * cw] == w[s as usize * cw..(s as usize + 1) * cw]))
    };
    let mb = (CHUNK * K as u64) as f64 / 1e6;
    for round in 0..2 {
        let clear = |c: &Ctx| c.once(|cb| unsafe { c.dev.cmd_fill_buffer(cb, dst, 0, vk::WHOLE_SIZE, 0) });
        let mut ti = vec![];
        let mut tc = vec![];
        for _ in 0..10 {
            clear(c)?;
            ti.push(c.once(|cb| unsafe { copy_ind(cb, &info) })?);
        }
        let ind_ok = verify()?;
        for _ in 0..10 {
            clear(c)?;
            tc.push(c.once(|cb| unsafe { c.dev.cmd_copy_buffer(cb, src, dst, &regions) })?);
        }
        let cpy_ok = verify()?;
        let (mi, mc) = (median(ti), median(tc));
        println!(
            "indirect copy r{round}: {K} x {} KiB host->VRAM gather | indirect {:.0} us ({:.1} GB/s) {} | vkCmdCopyBuffer {:.0} us ({:.1} GB/s) {}",
            CHUNK >> 10, mi / 1e3, mb / (mi / 1e9) / 1e3, ok(ind_ok), mc / 1e3, mb / (mc / 1e9) / 1e3, ok(cpy_ok)
        );
    }
    // One more: the command list rewritten by the GPU itself (fill swaps slot 0's src),
    // i.e. a kernel-chosen expert list consumed without a host round trip.
    let new_src = sa + 3 * CHUNK;
    c.once(|cb| unsafe {
        c.dev.cmd_fill_buffer(cb, dst, 0, vk::WHOLE_SIZE, 0);
        c.dev.cmd_update_buffer(cb, cmds, 0, &new_src.to_le_bytes());
        c.barrier(cb);
        copy_ind(cb, &info);
    })?;
    let got = c.read_words(dst, 0, (CHUNK / 4) as usize)?;
    let cw = (CHUNK / 4) as usize;
    println!("indirect copy from a GPU-written list (same submit): {}", ok(got[..] == arena.words()[3 * cw..4 * cw]));
    unsafe {
        for (b, m) in [(src, src_m), (dst, dst_m), (cmds, cmds_m)] {
            c.dev.destroy_buffer(b, None);
            c.dev.free_memory(m, None);
        }
    }
    Ok(())
}

const SPV_TMU: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/tmu_minmax.spv"));

/// MAX-reduction sampler as a 4:1 max pyramid step, vs the shader doing it, plus
/// bilinear weight precision. 2048^2 R32_SFLOAT (16 MiB) stays cache resident.
fn tmu(c: &Ctx) -> R<()> {
    const W: u32 = 2048;
    const OUT: usize = (W as usize / 2) * (W as usize / 2);
    let n = (W * W) as usize;
    let mut rng = 0x9E37_79B9u32;
    let data: Vec<f32> = (0..n)
        .map(|_| {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            (rng >> 8) as f32 / (1u32 << 24) as f32
        })
        .collect();
    let bytes = (n * 4) as u64;
    let hv = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    let (stage, stage_m) = c.plain(bytes, vk::BufferUsageFlags::TRANSFER_SRC, hv)?;
    let (src, src_m) = c.plain(bytes, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
    let (dst, dst_m) = c.plain((OUT * 4) as u64, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
    unsafe {
        let p = chk(c.dev.map_memory(stage_m, 0, bytes, vk::MemoryMapFlags::empty()), "map")? as *mut f32;
        std::ptr::copy_nonoverlapping(data.as_ptr(), p, n);
        c.dev.unmap_memory(stage_m);
    }
    let d = &c.dev;
    let fmt = vk::Format::R32_SFLOAT;
    let img = unsafe {
        chk(
            d.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(fmt)
                    .extent(vk::Extent3D { width: W, height: W, depth: 1 })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                    .initial_layout(vk::ImageLayout::UNDEFINED),
                None,
            ),
            "vkCreateImage",
        )?
    };
    let ireq = unsafe { d.get_image_memory_requirements(img) };
    let ity = find_mem_type(&c.mem, ireq.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL, vk::MemoryPropertyFlags::DEVICE_LOCAL).ok_or("image mem")?;
    let img_m = c.alloc(ireq.size, ity, false, None)?;
    unsafe { chk(d.bind_image_memory(img, img_m, 0), "bind image")? };
    let range = vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 };
    c.once(|cb| unsafe {
        let to_dst = vk::ImageMemoryBarrier::default()
            .image(img)
            .subresource_range(range)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::TRANSFER, vk::DependencyFlags::empty(), &[], &[], &[to_dst]);
        d.cmd_copy_buffer_to_image(
            cb,
            stage,
            img,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::BufferImageCopy {
                image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
                image_extent: vk::Extent3D { width: W, height: W, depth: 1 },
                ..Default::default()
            }],
        );
        d.cmd_copy_buffer(cb, stage, src, &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: bytes }]);
        let to_read = vk::ImageMemoryBarrier::default()
            .image(img)
            .subresource_range(range)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        let mb = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
        d.cmd_pipeline_barrier(cb, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[mb], &[], &[to_read]);
    })?;
    let view = unsafe {
        chk(
            d.create_image_view(
                &vk::ImageViewCreateInfo::default().image(img).view_type(vk::ImageViewType::TYPE_2D).format(fmt).subresource_range(range),
                None,
            ),
            "view",
        )?
    };
    let sampler = |red: Option<vk::SamplerReductionMode>| -> R<vk::Sampler> {
        let mut rci = vk::SamplerReductionModeCreateInfo::default();
        let mut sci = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(0.0);
        if let Some(m) = red {
            rci = rci.reduction_mode(m);
            sci = sci.push_next(&mut rci);
        }
        unsafe { chk(d.create_sampler(&sci, None), "sampler") }
    };
    let smax = sampler(Some(vk::SamplerReductionMode::MAX))?;
    let slin = sampler(None)?;
    let b = |i: u32, t: vk::DescriptorType| vk::DescriptorSetLayoutBinding::default().binding(i).descriptor_type(t).descriptor_count(1).stage_flags(vk::ShaderStageFlags::COMPUTE);
    let binds = [
        b(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
        b(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
        b(2, vk::DescriptorType::STORAGE_BUFFER),
        b(3, vk::DescriptorType::STORAGE_BUFFER),
    ];
    let dsl = unsafe { chk(d.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&binds), None), "dsl")? };
    let pcr = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: 16 }];
    let dsls = [dsl];
    let pl = unsafe { chk(d.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&dsls).push_constant_ranges(&pcr), None), "pl")? };
    let code: Vec<u32> = SPV_TMU.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect();
    let module = unsafe { chk(d.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None), "module")? };
    let stage_ci = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
    let pipe = unsafe {
        d.create_compute_pipelines(vk::PipelineCache::null(), &[vk::ComputePipelineCreateInfo::default().stage(stage_ci).layout(pl)], None)
            .map_err(|(_, e)| format!("pipeline: {e}"))?[0]
    };
    let sizes = [
        vk::DescriptorPoolSize { ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER, descriptor_count: 2 },
        vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: 2 },
    ];
    let dp = unsafe { chk(d.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&sizes), None), "dpool")? };
    let set = unsafe { chk(d.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(dp).set_layouts(&dsls)), "dset")?[0] };
    let ii = |s| [vk::DescriptorImageInfo { sampler: s, image_view: view, image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL }];
    let (i0, i1) = (ii(smax), ii(slin));
    let bi = |buf| [vk::DescriptorBufferInfo { buffer: buf, offset: 0, range: vk::WHOLE_SIZE }];
    let (b2, b3) = (bi(src), bi(dst));
    let w = |i: u32, t| vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i).descriptor_type(t);
    unsafe {
        d.update_descriptor_sets(
            &[
                w(0, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&i0),
                w(1, vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&i1),
                w(2, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b2),
                w(3, vk::DescriptorType::STORAGE_BUFFER).buffer_info(&b3),
            ],
            &[],
        )
    };
    let run = |mode: u32, reps: u32| -> R<f64> {
        let pc: Vec<u8> = [W, W, mode, reps].iter().flat_map(|x| x.to_le_bytes()).collect();
        c.once(|cb| unsafe {
            d.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, pipe);
            d.cmd_bind_descriptor_sets(cb, vk::PipelineBindPoint::COMPUTE, pl, 0, &[set], &[]);
            d.cmd_push_constants(cb, pl, vk::ShaderStageFlags::COMPUTE, 0, &pc);
            d.cmd_dispatch(cb, W / 32, W / 32, 1);
        })
    };
    let out_f32 = |c: &Ctx| -> R<Vec<f32>> { Ok(c.read_words(dst, 0, OUT)?.into_iter().map(f32::from_bits).collect()) };
    let h = W as usize / 2;
    let at = |x: usize, y: usize| data[y * W as usize + x];
    // Correctness, reps = 1.
    run(0, 1)?;
    let tm = out_f32(c)?;
    run(1, 1)?;
    let sh = out_f32(c)?;
    let exact = |i: usize| {
        let (x, y) = ((i % h) * 2, (i / h) * 2);
        at(x, y).max(at(x + 1, y)).max(at(x, y + 1)).max(at(x + 1, y + 1))
    };
    let tm_bad = (0..OUT).filter(|&i| tm[i].to_bits() != exact(i).to_bits()).count();
    let sh_bad = (0..OUT).filter(|&i| sh[i].to_bits() != exact(i).to_bits()).count();
    println!("TMU max-of-2x2 via MAX sampler: {} of {OUT} bit-exact | shader 4-load max: {} bit-exact", OUT - tm_bad, OUT - sh_bad);
    run(2, 1)?;
    let bl = out_f32(c)?;
    let mut worst = 0f64;
    for i in 0..OUT {
        let (x, y) = ((i % h) * 2, (i / h) * 2);
        let (fx, fy) = (0.3f64, 0.6f64);
        let top = at(x, y) as f64 * (1.0 - fx) + at(x + 1, y) as f64 * fx;
        let bot = at(x, y + 1) as f64 * (1.0 - fx) + at(x + 1, y + 1) as f64 * fx;
        worst = worst.max((bl[i] as f64 - (top * (1.0 - fy) + bot * fy)).abs());
    }
    println!("bilinear weights: worst abs error {worst:.2e} on [0,1) data (2^-8 = {:.2e}, 2^-24 = {:.2e})", 1.0 / 256.0, 1.0 / (1u64 << 24) as f64);
    // Throughput: 64 passes over the cache-resident image per dispatch.
    const REPS: u32 = 64;
    let fetches = (OUT as f64) * REPS as f64;
    for (mode, name) in [(0u32, "TMU MAX sampler"), (1, "shader 4 loads+max"), (2, "TMU bilinear")] {
        run(mode, REPS)?;
        let t = median((0..7).map(|_| run(mode, REPS)).collect::<R<Vec<_>>>()?);
        println!("  {name:<20} {:.2} ms, {:.1} G 2x2-blocks/s ({:.0} Gtexel/s)", t / 1e6, fetches / t, 4.0 * fetches / t);
    }
    unsafe {
        d.destroy_pipeline(pipe, None);
        d.destroy_shader_module(module, None);
        d.destroy_descriptor_pool(dp, None);
        d.destroy_pipeline_layout(pl, None);
        d.destroy_descriptor_set_layout(dsl, None);
        d.destroy_sampler(smax, None);
        d.destroy_sampler(slin, None);
        d.destroy_image_view(view, None);
        d.destroy_image(img, None);
        d.free_memory(img_m, None);
        for (b, m) in [(stage, stage_m), (src, src_m), (dst, dst_m)] {
            d.destroy_buffer(b, None);
            d.free_memory(m, None);
        }
    }
    Ok(())
}

/// CPU wall time of an empty submit+wait while a large allocation is merely alive
/// (unreferenced): userptr (host-pointer import) vs driver-allocated GTT. D3 showed
/// every stage that submits slowing in proportion to the imported bytes.
fn submit_cost(c: &Ctx) -> R<()> {
    const GIB: u64 = 1 << 30;
    let size = 3 * GIB / 2;
    let per_submit = |label: &str| -> R<()> {
        c.once(|_| {})?;
        let t = std::time::Instant::now();
        for _ in 0..50 {
            c.once(|_| {})?;
        }
        println!("  empty submit+wait, {label:<34} {:>8.1} us", t.elapsed().as_secs_f64() * 1e6 / 50.0);
        Ok(())
    };
    println!("submit cost vs live allocations ({} MiB each):", size >> 20);
    per_submit("nothing extra")?;
    let pages = HostPages::new(size, 4096);
    pages.words().iter_mut().step_by(1024).for_each(|w| *w = 1); // fault every page in, like the filled arena
    let (ub, um) = c.imported(&pages, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)?;
    per_submit("+ host-pointer import (userptr)")?;
    unsafe {
        c.dev.destroy_buffer(ub, None);
        c.dev.free_memory(um, None);
    }
    drop(pages);
    per_submit("after freeing the import")?;
    let gtt = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_CACHED;
    let (gb, gm) = c.plain(size, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS, gtt)?;
    per_submit("+ driver GTT (HOST_VISIBLE|CACHED)")?;
    unsafe {
        c.dev.destroy_buffer(gb, None);
        c.dev.free_memory(gm, None);
    }
    Ok(())
}
