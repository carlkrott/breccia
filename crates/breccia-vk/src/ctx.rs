//! Vulkan plumbing for the breccia probe: instance/device setup, discovery, buffers,
//! kernels, and timestamp-timed submissions. Everything headless (no surface, no swapchain).
//!
//! Ported from `harnesses/vk-probe/src/ctx.rs` (Phase-0.2, validation-clean) with two
//! additions: `deviceUUID` extraction (fingerprint) and PCIe address discovery.

use ash::vk;
use std::os::raw::c_char;

pub type R<T> = Result<T, String>;

/// Unwrap a `vk::Result`-returning call with a clear, contextual error message.
/// (ash 0.38 returns `Result<T, vk::Result>` where `vk::Result` is the error value.)
pub fn vk<T>(r: std::result::Result<T, vk::Result>, what: &str) -> R<T> {
    r.map_err(|e| format!("{what}: {e}"))
}

fn cstr_arr(a: &[c_char]) -> String {
    let bytes: Vec<u8> = a
        .iter()
        .take_while(|&&c| c != 0)
        .map(|&c| c as u8)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

// ---------------------------------------------------------------------------
// Physical device discovery (pre device-creation)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Default)]
pub struct Feats {
    pub shader_float16: bool,
    pub shader_int8: bool,
    pub atomic64_buffer: bool,
    pub atomic64_shared: bool,
    pub timeline_semaphore: bool,
    pub subgroup_size_control: bool,
    pub compute_full_subgroups: bool,
    pub cooperative_matrix: bool,
    /// Needed by vkResetQueryPool (Vulkan 1.2 core feature).
    pub host_query_reset: bool,
    /// Needed by the fp16 kernel's float16_t storage buffer (Vulkan 1.1 core feature).
    pub storage_buffer16_bit_access: bool,
}

pub struct Dev {
    pub index: u32,
    pub pdev: vk::PhysicalDevice,
    pub name: String,
    pub device_type: vk::PhysicalDeviceType,
    pub vendor_id: u32,
    pub device_id: u32,
    pub api_version: u32,
    pub driver_version: u32,
    pub driver_name: String,
    pub driver_info: String,
    /// `VkPhysicalDeviceIDProperties::deviceUUID` as a hyphenated 8-4-4-4-12 string.
    pub uuid: String,
    /// PCIe address (`0000:03:00.0`) — from `VK_EXT_pci_bus_info` or sysfs id match.
    /// Empty when neither source is available.
    pub pci_address: String,
    pub families: Vec<vk::QueueFamilyProperties>,
    pub mem: vk::PhysicalDeviceMemoryProperties,
    pub exts: Vec<String>,
    pub timestamp_period_ns: f32,
    pub subgroup_size: u32,
    pub subgroup_stages: vk::ShaderStageFlags,
    pub subgroup_ops: vk::SubgroupFeatureFlags,
    pub quad_all_stages: bool,
    pub min_subgroup_size: u32,
    pub max_subgroup_size: u32,
    pub max_compute_workgroup_subgroups: u32,
    pub feats: Feats,
}

impl Dev {
    pub fn has_ext(&self, name: &str) -> bool {
        self.exts.iter().any(|e| e == name)
    }
}

pub struct Host {
    /// Kept alive for as long as the instance exists (loader lifetime).
    #[allow(dead_code)]
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub api_version: u32,
    /// True when `MESA_VK_DEVICE_SELECT` was present in the environment and had to
    /// be cleared before instance creation (the iGPU-pinning trap, docs/PROJECT-SETUP).
    pub device_select_was_set: bool,
    /// Value of the variable before it was cleared (for the warning text).
    pub device_select_value: Option<String>,
}

impl Host {
    pub fn new() -> R<Host> {
        // The MESA device-select layer would otherwise pin enumeration to a single
        // device (it is set in this machine's environment for the desktop session).
        let was_set = std::env::var("MESA_VK_DEVICE_SELECT").ok();
        if let Some(v) = &was_set {
            eprintln!(
                "breccia-vk: WARNING: MESA_VK_DEVICE_SELECT={v} was set — unsetting it so all \
                 Vulkan devices enumerate (it would otherwise hide both discrete GPUs)"
            );
            // Safety: edition 2021. Must happen before vkCreateInstance.
            std::env::remove_var("MESA_VK_DEVICE_SELECT");
        }

        let entry =
            unsafe { ash::Entry::load() }.map_err(|e| format!("failed to load libvulkan: {e}"))?;
        let app = vk::ApplicationInfo::default()
            .application_name(c"breccia-vk")
            .application_version(1)
            .engine_name(c"breccia")
            .engine_version(1)
            .api_version(vk::API_VERSION_1_3);
        let ici = vk::InstanceCreateInfo::default().application_info(&app);
        let instance = unsafe { vk(entry.create_instance(&ici, None), "vkCreateInstance")? };
        let api_version = unsafe { entry.try_enumerate_instance_version() }
            .ok()
            .flatten()
            .unwrap_or(vk::API_VERSION_1_0);
        Ok(Host {
            entry,
            instance,
            api_version,
            device_select_was_set: was_set.is_some(),
            device_select_value: was_set,
        })
    }

    /// Enumerate every physical device with its properties, queues, memory and extensions.
    pub fn devices(&self) -> R<Vec<Dev>> {
        let pdevs = unsafe {
            vk(
                self.instance.enumerate_physical_devices(),
                "vkEnumeratePhysicalDevices",
            )?
        };
        let mut out = Vec::new();
        for (index, pdev) in pdevs.into_iter().enumerate() {
            let props = unsafe { self.instance.get_physical_device_properties(pdev) };

            // Extensions first: some property structs are only chained when the
            // device advertises the extension (VK_EXT_pci_bus_info).
            let exts = unsafe {
                self.instance
                    .enumerate_device_extension_properties(pdev)
                    .map_err(|e| format!("vkEnumerateDeviceExtensionProperties: {e}"))?
            };
            let ext_names: Vec<String> = exts.iter().map(|e| cstr_arr(&e.extension_name)).collect();

            let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
            let mut ssc = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
            let mut driver = vk::PhysicalDeviceDriverProperties::default();
            let mut idp = vk::PhysicalDeviceIDProperties::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut subgroup)
                .push_next(&mut ssc)
                .push_next(&mut driver)
                .push_next(&mut idp);
            let has_pci_bus_info = ext_names.iter().any(|e| e == "VK_EXT_pci_bus_info");
            let mut pci = vk::PhysicalDevicePCIBusInfoPropertiesEXT::default();
            if has_pci_bus_info {
                p2 = p2.push_next(&mut pci);
            }
            unsafe { self.instance.get_physical_device_properties2(pdev, &mut p2) };

            // Feature query (chained structs are filled by the driver when supported).
            let mut f16 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
            let mut f64 = vk::PhysicalDeviceShaderAtomicInt64Features::default();
            let mut ftl = vk::PhysicalDeviceTimelineSemaphoreFeatures::default();
            let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
            let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut fcoop = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
            let mut f2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut f16)
                .push_next(&mut f64)
                .push_next(&mut ftl)
                .push_next(&mut f11)
                .push_next(&mut f12)
                .push_next(&mut f13);
            // Chaining KHR_cooperative_matrix is only meaningful if the device lists it;
            // drivers ignore unknown pNext structs, but stay tidy anyway.
            let has_coop = ext_names.iter().any(|e| e == "VK_KHR_cooperative_matrix");
            if has_coop {
                f2 = f2.push_next(&mut fcoop);
            }
            unsafe {
                self.instance.get_physical_device_features2(pdev, &mut f2);
            }

            let feats = Feats {
                shader_float16: f16.shader_float16 != 0,
                shader_int8: f16.shader_int8 != 0,
                atomic64_buffer: f64.shader_buffer_int64_atomics != 0,
                atomic64_shared: f64.shader_shared_int64_atomics != 0,
                timeline_semaphore: ftl.timeline_semaphore != 0,
                subgroup_size_control: f13.subgroup_size_control != 0,
                compute_full_subgroups: f13.compute_full_subgroups != 0,
                cooperative_matrix: fcoop.cooperative_matrix != 0,
                host_query_reset: f12.host_query_reset != 0,
                storage_buffer16_bit_access: f11.storage_buffer16_bit_access != 0,
            };

            let families = unsafe {
                self.instance
                    .get_physical_device_queue_family_properties(pdev)
            };
            let mem = unsafe { self.instance.get_physical_device_memory_properties(pdev) };

            let uuid = format_uuid(&idp.device_uuid);
            let pci_address = if has_pci_bus_info {
                crate::pci::address_from_bus_info(pci.pci_bus, pci.pci_device, pci.pci_function)
            } else {
                crate::pci::find_address_by_ids(props.vendor_id, props.device_id)
                    .unwrap_or_default()
            };

            out.push(Dev {
                index: index as u32,
                pdev,
                name: cstr_arr(&props.device_name),
                device_type: props.device_type,
                vendor_id: props.vendor_id,
                device_id: props.device_id,
                api_version: props.api_version,
                driver_version: props.driver_version,
                driver_name: cstr_arr(&driver.driver_name),
                driver_info: cstr_arr(&driver.driver_info),
                uuid,
                pci_address,
                families,
                mem,
                exts: ext_names,
                timestamp_period_ns: props.limits.timestamp_period,
                subgroup_size: subgroup.subgroup_size,
                subgroup_stages: subgroup.supported_stages,
                subgroup_ops: subgroup.supported_operations,
                quad_all_stages: subgroup.quad_operations_in_all_stages != 0,
                min_subgroup_size: ssc.min_subgroup_size,
                max_subgroup_size: ssc.max_subgroup_size,
                max_compute_workgroup_subgroups: ssc.max_compute_workgroup_subgroups,
                feats,
            });
        }
        Ok(out)
    }

    pub fn destroy(self) {
        unsafe { self.instance.destroy_instance(None) };
    }
}

fn format_uuid(bytes: &[u8; 16]) -> String {
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

pub fn api_version_string(v: u32) -> String {
    format!(
        "{}.{}.{}",
        vk::api_version_major(v),
        vk::api_version_minor(v),
        vk::api_version_patch(v)
    )
}

/// RADV/AMD driver versions are encoded with 7/10/12-bit fields in Vulkan 1.3+
/// (matches `vulkaninfo`'s decode on this machine, e.g. 26.2.4).
pub fn driver_version_string(v: u32) -> String {
    format!("{}.{}.{}", (v >> 22) & 0x7f, (v >> 12) & 0x3ff, v & 0xfff)
}

// ---------------------------------------------------------------------------
// Device (logical) + queues + command/timing resources
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Qc {
    /// Compute (or universal) queue.
    C,
    /// Transfer queue (may be the same queue object as C).
    X,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum XferMode {
    DistinctFamily,
    SameFamilySecondQueue,
    SameQueue,
}

impl XferMode {
    pub fn as_str(self) -> &'static str {
        match self {
            XferMode::DistinctFamily => "distinct_transfer_queue_family",
            XferMode::SameFamilySecondQueue => "second_queue_of_compute_family",
            XferMode::SameQueue => "same_queue_as_compute",
        }
    }
}

pub struct Buf {
    pub buf: vk::Buffer,
    pub mem: vk::DeviceMemory,
    pub size: u64,
}

pub struct Kernel {
    pub pipe: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub dsl: vk::DescriptorSetLayout,
    pub dpool: vk::DescriptorPool,
    pub set: vk::DescriptorSet,
}

pub struct Timing {
    pub iters: u32,
    pub total_s: f64,
    pub per_iter_s: f64,
    pub wall_s: f64,
    pub used_timestamps: bool,
}

/// Command buffers the compute pool owns: index 0 is the **synchronous one-shot**
/// buffer (`Gpu::cmd_c`, used by `record`/`wall_run`), the rest are the
/// pipelining ring that `breccia_kernels::dispatch_batch` records a block's
/// segments into (six: two streams keep one block's batch open each, plus the
/// segments still in flight).
///
/// They must be separate objects: a buffer that is pending execution cannot be
/// reset, and the whole point of the ring is to have one segment in flight while
/// the next is being recorded.
pub const BATCH_CMDS: usize = 6;

/// A timeline semaphore, used as a **mid-batch sync point**: a submission can
/// signal a value part-way through a block's work and the host can wait for it
/// while the rest of that work is still executing.
///
/// This is what lets the host expert lane overlap the GPU: a block is submitted
/// as two submissions — everything up to the router (which signals) and the
/// shared expert (submitted behind it) — so the host learns the router's ids one
/// shared-expert-worth of GPU time before the block ends instead of after a
/// fence (`docs/phase-4/01-forward-pass.md` §8m/§8n).
///
/// The wait is **bounded**: [`Timeline::wait_host`] takes a timeout and the
/// caller falls back to the fence. Nothing here can hang the box.
pub struct Timeline {
    pub sem: vk::Semaphore,
    /// Highest value handed to a submission, so a caller cannot signal
    /// backwards (VUID-VkSubmitInfo-pSignalSemaphores-03242 is a validation
    /// error *and* a hard `vkQueueSubmit` failure).
    value: std::cell::Cell<u64>,
}

/// Outcome of [`Timeline::wait_host`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Signalled {
    /// The GPU reached the marker before the timeout.
    Yes,
    /// The timeout expired; the caller must fall back to the fence.
    Timeout,
}

impl Timeline {
    /// The next value a submission should signal (strictly increasing).
    pub fn next(&self) -> u64 {
        self.value.get() + 1
    }

    /// Record that `value` was handed to a submission (call after the submit
    /// returned `Ok`).
    pub fn advanced(&self, value: u64) {
        self.value.set(value);
    }

    /// Wait (host side, no fence for the submission) until the GPU signals at
    /// least `value`. `timeout_ns` bounds it.
    pub fn wait_host(&self, gpu: &Gpu, value: u64, timeout_ns: u64) -> R<Signalled> {
        let sems = [self.sem];
        let vals = [value];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&sems)
            .values(&vals);
        match unsafe { gpu.device.wait_semaphores(&info, timeout_ns) } {
            Ok(()) => Ok(Signalled::Yes),
            Err(vk::Result::TIMEOUT) => Ok(Signalled::Timeout),
            Err(e) => Err(format!("vkWaitSemaphores({value}): {e}")),
        }
    }

    /// The counter the GPU has actually reached.
    pub fn counter(&self, gpu: &Gpu) -> R<u64> {
        vk(
            unsafe { gpu.device.get_semaphore_counter_value(self.sem) },
            "vkGetSemaphoreCounterValue",
        )
    }

    pub fn release(&self, gpu: &Gpu) {
        unsafe { gpu.device.destroy_semaphore(self.sem, None) };
    }
}

pub struct Gpu {
    pub index: u32,
    pub device: ash::Device,
    /// Handle copy of the owning instance (extension function-pointer loading).
    /// The [`Host`] still owns instance destruction.
    pub instance: ash::Instance,
    /// `VK_EXT_external_memory_host` was advertised **and** enabled on this device.
    pub external_memory_host: bool,
    pub mem: vk::PhysicalDeviceMemoryProperties,
    pub timestamp_period_ns: f64,
    pub feats: Feats,

    pub queue_c: vk::Queue,
    pub family_c: u32,
    pub queue_x: vk::Queue,
    pub family_x: u32,
    pub queue_x_index: u32,
    pub xfer_mode: XferMode,

    pub pool_c: vk::CommandPool,
    pub cmd_c: vk::CommandBuffer,
    /// `[cmd_c, ring[0], ring[1], …]` — see [`BATCH_CMDS`]. Allocated from
    /// `pool_c`, so destroying the pool frees them.
    pub cmds_c: Vec<vk::CommandBuffer>,
    pub pool_x: vk::CommandPool,
    pub cmd_x: vk::CommandBuffer,
    pub timer_c: vk::QueryPool,
    pub timer_x: vk::QueryPool,
    pub ts_ok_c: bool,
    pub ts_ok_x: bool,
    /// `vkQueueSubmit2` (VK_KHR_synchronization2, core in Vulkan 1.3): the only
    /// submit path that can carry a timeline-semaphore value.
    ///
    /// Loaded by its **core** name: ash 0.38's `synchronization2` wrapper loads
    /// `vkQueueSubmit2KHR`, which RADV does not export (it advertises sync2 only
    /// through the promoted 1.3 entry point), so the wrapper panics at submit.
    pub queue_submit2: Option<QueueSubmit2Fn>,
    /// `vkCmdPipelineBarrier2` by its **core** 1.3 name (same loader caveat as
    /// `queue_submit2`). Used for the per-dispatch compute→compute barrier, whose
    /// `ALL_COMMANDS → ALL_COMMANDS` form costs 4.4 µs (MI50) / 5.2 µs (RX) per
    /// dispatch against 2.7 / 1.1 µs for a compute-scoped sync2 barrier
    /// (`docs/research/01-vulkan-and-mlc.md`, re-measured in
    /// `docs/research/02-vulkan-followups-and-powerinfer.md` §A1).
    pub pipeline_barrier2: Option<PipelineBarrier2Fn>,
    /// Scope the compute→compute dispatch barrier uses: see [`BarrierScope`].
    pub barrier_scope: BarrierScope,
}

/// `vkCmdPipelineBarrier2` with the lifetime erased.
pub type PipelineBarrier2Fn = unsafe extern "system" fn(
    command_buffer: vk::CommandBuffer,
    dependency_info: *const vk::DependencyInfo<'static>,
);

/// How [`Gpu::barrier_compute`] fences one recorded dispatch against the next.
///
/// Decided by measurement, not taste: the full `ALL_COMMANDS → ALL_COMMANDS`
/// memory barrier is correct but forces a launch serialisation that costs more
/// than the dispatch it protects; `Sync2` narrows the scope to the compute stage
/// with shader-storage access masks, which is the strictest scope that still
/// covers "the next dispatch reads what this one wrote". `None` exists **only**
/// for the three-arm bench and is never a shipped default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BarrierScope {
    /// `ALL_COMMANDS → ALL_COMMANDS` memory barrier (`Gpu::barrier_all`).
    Full,
    /// `COMPUTE → COMPUTE` with shader read/write access masks (sync2).
    #[default]
    Sync2,
    /// No barrier at all — bench arm only; produces wrong results.
    None,
}

/// Wait for `fence` with [`Gpu::WAIT_TIMEOUT_NS`]; `false` on timeout or device loss.
///
/// The one place a fence wait happens, so no call site can reintroduce `u64::MAX`
/// (AGENTS.md: an unbounded wait on the display GPU rebooted the box).
fn wait_fence_bounded(d: &ash::Device, fence: vk::Fence, what: &str) -> bool {
    let r = unsafe { d.wait_for_fences(&[fence], true, Gpu::WAIT_TIMEOUT_NS) };
    match r {
        Ok(()) => true,
        Err(e) => {
            eprintln!("breccia: waiting on {what}: {e}");
            false
        }
    }
}

/// `vkQueueSubmit2` with the lifetime erased (it is an ABI-level function
/// pointer; the structs it points at are borrowed only for the call).
pub type QueueSubmit2Fn = unsafe extern "system" fn(
    queue: vk::Queue,
    submit_count: u32,
    submits: *const vk::SubmitInfo2<'static>,
    fence: vk::Fence,
) -> vk::Result;

/// Pick the first memory type in `bits` that satisfies `req`, preferring `pref`.
pub fn find_mem_type(
    mem: &vk::PhysicalDeviceMemoryProperties,
    bits: u32,
    req: vk::MemoryPropertyFlags,
    pref: vk::MemoryPropertyFlags,
) -> Option<u32> {
    let mut fallback = None;
    for i in 0..mem.memory_type_count {
        if bits & (1u32 << i) == 0 {
            continue;
        }
        let ty = mem.memory_types[i as usize].property_flags;
        if ty.contains(req) {
            if ty.contains(pref) {
                return Some(i);
            }
            if fallback.is_none() {
                fallback = Some(i);
            }
        }
    }
    fallback
}

impl Gpu {
    pub fn create(host: &Host, dev: &Dev, index: u32) -> R<Gpu> {
        let instance = &host.instance;

        // --- queue selection -------------------------------------------------
        let fam_c =
            dev.families
                .iter()
                .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .ok_or_else(|| "no compute-capable queue family".to_string())? as u32;
        let queue_c_count = dev.families[fam_c as usize].queue_count;

        let dedicated_x = dev
            .families
            .iter()
            .position(|f| {
                f.queue_flags.contains(vk::QueueFlags::TRANSFER)
                    && !f.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    && !f.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            })
            .map(|i| i as u32);
        // Otherwise prefer a *different* family that can still transfer (async
        // compute/transfer queue); RADV exposes one on both cards.
        let other_x = dev
            .families
            .iter()
            .enumerate()
            .find(|(i, f)| *i as u32 != fam_c && f.queue_flags.contains(vk::QueueFlags::TRANSFER))
            .map(|(i, _)| i as u32);
        let any_x = dev
            .families
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::TRANSFER))
            .map(|i| i as u32);
        let fam_x = dedicated_x.or(other_x).or(any_x);

        let (family_x, xfer_mode, queue_x_index) = match fam_x {
            Some(fx) if fx != fam_c => (fx, XferMode::DistinctFamily, 0),
            Some(_) if queue_c_count >= 2 => (fam_c, XferMode::SameFamilySecondQueue, 1),
            Some(_) => (fam_c, XferMode::SameQueue, 0),
            None => (fam_c, XferMode::SameQueue, 0),
        };

        // --- device queue create infos ---------------------------------------
        // One entry per family; the transfer queue needs a second queue slot on
        // the compute family when there is no dedicated transfer family.
        let x_wants_two = family_x == fam_c && queue_x_index == 1;
        let mut fam_counts: Vec<(u32, u32)> = vec![(fam_c, 1)];
        if family_x == fam_c {
            if x_wants_two {
                fam_counts[0].1 = 2;
            }
        } else {
            fam_counts.push((family_x, 1));
        }
        let prios = [0.0f32; 2];
        let qcis: Vec<vk::DeviceQueueCreateInfo> = fam_counts
            .iter()
            .map(|&(fam, cnt)| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(fam)
                    .queue_priorities(&prios[..cnt as usize])
            })
            .collect();

        // --- optional extensions + features ----------------------------------
        let mut enabled_exts: Vec<String> = Vec::new();
        if dev.feats.cooperative_matrix && dev.has_ext("VK_KHR_cooperative_matrix") {
            enabled_exts.push("VK_KHR_cooperative_matrix".to_string());
        }
        // Phase 2: needed by the zero-copy pinned-host probe in `transfer`.
        let external_memory_host = dev.has_ext("VK_EXT_external_memory_host");
        if external_memory_host {
            enabled_exts.push("VK_EXT_external_memory_host".to_string());
        }

        // NOTE: Vulkan12Features already carries shaderFloat16/shaderInt8/atomic64/
        // timeline/hostQueryReset — validation forbids also chaining the promoted
        // per-feature structs alongside it (VUID-VkDeviceCreateInfo-pNext-02830).
        let mut en_f12 = vk::PhysicalDeviceVulkan12Features {
            shader_float16: dev.feats.shader_float16 as u32,
            shader_int8: dev.feats.shader_int8 as u32,
            shader_buffer_int64_atomics: dev.feats.atomic64_buffer as u32,
            shader_shared_int64_atomics: dev.feats.atomic64_shared as u32,
            timeline_semaphore: dev.feats.timeline_semaphore as u32,
            host_query_reset: dev.feats.host_query_reset as u32,
            ..Default::default()
        };
        let mut en_f11 = vk::PhysicalDeviceVulkan11Features {
            storage_buffer16_bit_access: dev.feats.storage_buffer16_bit_access as u32,
            ..Default::default()
        };
        let mut en_f13 = vk::PhysicalDeviceVulkan13Features {
            subgroup_size_control: dev.feats.subgroup_size_control as u32,
            compute_full_subgroups: dev.feats.compute_full_subgroups as u32,
            // vkQueueSubmit2 is how a submission carries a timeline-semaphore
            // value (the mid-batch marker); the feature gates the command.
            synchronization2: 1,
            ..Default::default()
        };
        let mut en_coop = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR {
            cooperative_matrix: dev.feats.cooperative_matrix as u32,
            ..Default::default()
        };

        let mut feats2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut en_f11)
            .push_next(&mut en_f12)
            .push_next(&mut en_f13);
        if enabled_exts
            .iter()
            .any(|e| e == "VK_KHR_cooperative_matrix")
        {
            feats2 = feats2.push_next(&mut en_coop);
        }

        // Keep NUL-terminated extension name copies alive across vkCreateDevice.
        let ext_owned: Vec<std::ffi::CString> = enabled_exts
            .iter()
            .map(|s| std::ffi::CString::new(s.as_str()).unwrap())
            .collect();
        let ext_ptrs: Vec<*const c_char> = ext_owned.iter().map(|c| c.as_ptr()).collect();

        let dci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qcis)
            .enabled_extension_names(&ext_ptrs)
            .push_next(&mut feats2);

        let device = unsafe {
            vk(
                instance.create_device(dev.pdev, &dci, None),
                &format!("vkCreateDevice({})", dev.name),
            )?
        };

        let queue_c = unsafe { device.get_device_queue(fam_c, 0) };
        let queue_x = unsafe { device.get_device_queue(family_x, queue_x_index) };

        // --- command pools / buffers / query pools ---------------------------
        // Always separate command buffers (and pools) per queue: the overlap test
        // submits both at once, and they must not alias when the "transfer" queue
        // happens to be the same queue object as the compute queue.
        let (pool_c, cmds_c) = make_cmd_pool_n(&device, fam_c, 1 + BATCH_CMDS)?;
        let cmd_c = cmds_c[0];
        let (pool_x, cmd_x) = make_cmd_pool(&device, family_x)?;
        let timer_c = make_timer(&device)?;
        let timer_x = make_timer(&device)?;
        let fam_c_props = dev.families[fam_c as usize];
        let fam_x_props = dev.families[family_x as usize];

        println!(
            "  [{}] logical device '{}' created: compute=queue family {fam_c}, transfer=family \
             {family_x} index {queue_x_index} ({}), enabled optional extensions=[{}]",
            index,
            dev.name,
            xfer_mode.as_str(),
            enabled_exts.join(", ")
        );

        // Load vkQueueSubmit2 by its core name (see the field's docs).
        let queue_submit2: Option<QueueSubmit2Fn> = {
            let name = c"vkQueueSubmit2";
            let fp = unsafe { instance.get_device_proc_addr(device.handle(), name.as_ptr()) };
            if fp.is_none() {
                None
            } else {
                // SAFETY: the loader returned a non-null pointer for the core
                // 1.3 name of vkQueueSubmit2; the signature is Vulkan's own.
                Some(unsafe { std::mem::transmute::<vk::PFN_vkVoidFunction, QueueSubmit2Fn>(fp) })
            }
        };
        // Load vkCmdPipelineBarrier2 by its core name (same caveat as submit2).
        let pipeline_barrier2: Option<PipelineBarrier2Fn> = {
            let name = c"vkCmdPipelineBarrier2";
            let fp = unsafe { instance.get_device_proc_addr(device.handle(), name.as_ptr()) };
            if fp.is_none() {
                None
            } else {
                // SAFETY: the loader returned a non-null pointer for the core 1.3
                // name; the signature is Vulkan's own.
                Some(unsafe {
                    std::mem::transmute::<vk::PFN_vkVoidFunction, PipelineBarrier2Fn>(fp)
                })
            }
        };
        // `BRECCIA_BARRIER=full|sync2|none` — the three-arm bench of §8w. `none`
        // is a measurement arm only (it produces wrong results).
        let barrier_scope = match std::env::var("BRECCIA_BARRIER").as_deref() {
            Ok("full") => BarrierScope::Full,
            Ok("none") => BarrierScope::None,
            _ => BarrierScope::Sync2,
        };
        Ok(Gpu {
            index,
            device,
            instance: instance.clone(),
            external_memory_host,
            mem: dev.mem,
            timestamp_period_ns: dev.timestamp_period_ns as f64,
            feats: dev.feats,
            queue_c,
            family_c: fam_c,
            queue_x,
            family_x,
            queue_x_index,
            xfer_mode,
            pool_c,
            cmd_c,
            cmds_c,
            pool_x,
            cmd_x,
            timer_c,
            timer_x,
            ts_ok_c: fam_c_props.timestamp_valid_bits > 0,
            ts_ok_x: fam_x_props.timestamp_valid_bits > 0,
            queue_submit2,
            pipeline_barrier2,
            barrier_scope,
        })
    }

    fn parts(&self, q: Qc) -> (vk::Queue, vk::CommandBuffer, vk::QueryPool, bool) {
        match q {
            Qc::C => (self.queue_c, self.cmd_c, self.timer_c, self.ts_ok_c),
            Qc::X => (self.queue_x, self.cmd_x, self.timer_x, self.ts_ok_x),
        }
    }

    // --- buffers -------------------------------------------------------------

    pub fn buf(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        req: vk::MemoryPropertyFlags,
        pref: vk::MemoryPropertyFlags,
    ) -> R<Buf> {
        self.buf_with(size, usage, req, pref, None)
    }

    /// [`Gpu::buf`] with an explicit set of queue families.
    ///
    /// `families = Some(&[..])` creates a `VK_SHARING_MODE_CONCURRENT` buffer, which
    /// is what the Phase-2 transfer ring needs: slot buffers are written by the
    /// transfer queue family and read by the compute family, and concurrent sharing
    /// avoids queue-family ownership transfers on every reuse. A single (or empty)
    /// family degrades to the exclusive path.
    pub fn buf_with(
        &self,
        size: u64,
        usage: vk::BufferUsageFlags,
        req: vk::MemoryPropertyFlags,
        pref: vk::MemoryPropertyFlags,
        families: Option<&[u32]>,
    ) -> R<Buf> {
        unsafe {
            let mut fams: Vec<u32> = families.map(|f| f.to_vec()).unwrap_or_default();
            fams.sort_unstable();
            fams.dedup();
            let mut bci = vk::BufferCreateInfo::default().size(size).usage(usage);
            bci = if fams.len() > 1 {
                bci.sharing_mode(vk::SharingMode::CONCURRENT)
                    .queue_family_indices(&fams)
            } else {
                bci.sharing_mode(vk::SharingMode::EXCLUSIVE)
            };
            let buffer = vk(self.device.create_buffer(&bci, None), "vkCreateBuffer")?;
            let reqs = self.device.get_buffer_memory_requirements(buffer);
            let idx = match find_mem_type(&self.mem, reqs.memory_type_bits, req, pref) {
                Some(i) => i,
                None => {
                    self.device.destroy_buffer(buffer, None);
                    return Err(format!(
                        "no memory type satisfies {req:?} (preferred {pref:?}) for {size} bytes \
                         (type bits {:#x})",
                        reqs.memory_type_bits
                    ));
                }
            };
            let mai = vk::MemoryAllocateInfo::default()
                .allocation_size(reqs.size)
                .memory_type_index(idx);
            let mem = match self.device.allocate_memory(&mai, None) {
                Ok(m) => m,
                Err(e) => {
                    self.device.destroy_buffer(buffer, None);
                    return Err(format!(
                        "vkAllocateMemory({} bytes, type {idx}) failed: {e}",
                        reqs.size
                    ));
                }
            };
            if let Err(e) = self.device.bind_buffer_memory(buffer, mem, 0) {
                self.device.free_memory(mem, None);
                self.device.destroy_buffer(buffer, None);
                return Err(format!("vkBindBufferMemory failed: {e}"));
            }
            Ok(Buf {
                buf: buffer,
                mem,
                size,
            })
        }
    }

    /// Write bytes into a host-visible buffer (map -> copy -> unmap).
    pub fn write_host(&self, b: &Buf, data: &[u8]) -> R<()> {
        assert!(data.len() as u64 <= b.size, "write_host overflow");
        unsafe {
            let ptr = vk(
                self.device
                    .map_memory(b.mem, 0, data.len() as u64, vk::MemoryMapFlags::empty()),
                "vkMapMemory",
            )? as *mut u8;
            std::slice::from_raw_parts_mut(ptr, data.len()).copy_from_slice(data);
            self.device.unmap_memory(b.mem);
        }
        Ok(())
    }

    /// Read bytes back from a host-visible buffer (map -> copy -> unmap).
    pub fn read_host(&self, b: &Buf, out: &mut [u8]) -> R<()> {
        assert!(out.len() as u64 <= b.size, "read_host overflow");
        unsafe {
            let ptr = vk(
                self.device
                    .map_memory(b.mem, 0, out.len() as u64, vk::MemoryMapFlags::empty()),
                "vkMapMemory",
            )? as *const u8;
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), out.len());
            self.device.unmap_memory(b.mem);
        }
        Ok(())
    }

    pub fn destroy_buf(&self, b: &Buf) {
        unsafe {
            self.device.destroy_buffer(b.buf, None);
            self.device.free_memory(b.mem, None);
        }
    }

    // --- kernels -------------------------------------------------------------

    pub fn kernel(&self, spv: &[u8], n_storage: u32) -> R<Kernel> {
        unsafe {
            let words: Vec<u32> = spv
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().expect("4-byte SPIR-V chunks")))
                .collect();
            let smci = vk::ShaderModuleCreateInfo::default().code(&words);
            let module = vk(
                self.device.create_shader_module(&smci, None),
                "vkCreateShaderModule",
            )?;

            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..n_storage)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let dslci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            let dsl = vk(
                self.device.create_descriptor_set_layout(&dslci, None),
                "vkCreateDescriptorSetLayout",
            )?;

            // 32 bytes, not 16. The decode kernels' original four words
            // (rows/cols/bpr/splits) exactly filled 16 bytes, so there was no
            // room for a fifth word (accumulator selector, per-row weight base)
            // without a second pipeline layout per shader. Every device here
            // reports maxPushConstantsSize >= 128 (Vulkan's guaranteed minimum is
            // also 128), so 32 costs nothing. Shaders still declare only the
            // 16-byte prefix they read; a 32-byte range with a 16-byte block at
            // offset 0 is backwards compatible (`cmd_push_constants` may push a
            // prefix of the range), and a shader that needs a fifth word can now
            // declare 20/32 bytes without touching the layout
            // (docs/phase-3/03-gaps-closed.md).
            let pcr = vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(32);
            let layouts = [dsl];
            let pcrs = [pcr];
            let plci = vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&layouts)
                .push_constant_ranges(&pcrs);
            let layout = vk(
                self.device.create_pipeline_layout(&plci, None),
                "vkCreatePipelineLayout",
            )?;

            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(c"main");
            let cpci = vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(layout);
            let pipes =
                self.device
                    .create_compute_pipelines(vk::PipelineCache::null(), &[cpci], None);
            let pipe = match pipes {
                Ok(v) => v[0],
                Err((_, e)) => {
                    self.device.destroy_pipeline_layout(layout, None);
                    self.device.destroy_descriptor_set_layout(dsl, None);
                    self.device.destroy_shader_module(module, None);
                    return Err(format!("vkCreateComputePipelines: {e}"));
                }
            };
            self.device.destroy_shader_module(module, None);

            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(n_storage.max(1))];
            let dpci = vk::DescriptorPoolCreateInfo::default()
                .max_sets(1)
                .pool_sizes(&sizes);
            let dpool = vk(
                self.device.create_descriptor_pool(&dpci, None),
                "vkCreateDescriptorPool",
            )?;
            let sl = [dsl];
            let dai = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(dpool)
                .set_layouts(&sl);
            let set = vk(
                self.device.allocate_descriptor_sets(&dai),
                "vkAllocateDescriptorSets",
            )?[0];

            Ok(Kernel {
                pipe,
                layout,
                dsl,
                dpool,
                set,
            })
        }
    }

    pub fn bind_kernel(&self, k: &Kernel, bufs: &[vk::Buffer]) -> R<()> {
        unsafe {
            let infos: Vec<vk::DescriptorBufferInfo> = bufs
                .iter()
                .map(|b| vk::DescriptorBufferInfo {
                    buffer: *b,
                    offset: 0,
                    range: vk::WHOLE_SIZE,
                })
                .collect();
            let writes: Vec<vk::WriteDescriptorSet> = infos
                .iter()
                .enumerate()
                .map(|(i, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(k.set)
                        .dst_binding(i as u32)
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(std::slice::from_ref(info))
                })
                .collect();
            self.device.update_descriptor_sets(&writes, &[]);
        }
        Ok(())
    }

    pub fn destroy_kernel(&self, k: &Kernel) {
        unsafe {
            self.device.destroy_descriptor_pool(k.dpool, None);
            self.device.destroy_pipeline(k.pipe, None);
            self.device.destroy_pipeline_layout(k.layout, None);
            self.device.destroy_descriptor_set_layout(k.dsl, None);
        }
    }

    // --- recording / submission ---------------------------------------------

    /// Conservative full barrier (validation-friendly, harmless for a probe).
    ///
    /// # Safety
    /// `cmd` must be a command buffer in the recording state on `d`'s device.
    pub unsafe fn barrier_all(d: &ash::Device, cmd: vk::CommandBuffer) {
        let mb = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE);
        d.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[mb],
            &[],
            &[],
        );
    }

    /// How long any single GPU wait may block before the process aborts.
    ///
    /// An **unbounded** `vkWaitForFences(u64::MAX)` is what turned the 2026-10-26
    /// ring timeout into a machine reboot: the wait blocked forever, the kernel's
    /// GPU watchdog decided the card was wedged, and it issued a BACO reset on the
    /// **display** GPU (AGENTS.md incident log). A bounded wait leaves the device
    /// healthy — measured 81 ms to a usable device after an abort — so every fence
    /// and semaphore wait here gives up, prints what it was waiting for, and aborts
    /// the process instead of hanging.
    pub const WAIT_TIMEOUT_NS: u64 = 10_000_000_000; // 10 s

    /// Print a diagnostic and abort: used when a GPU wait expires.
    pub fn fault_abort(what: &str) -> ! {
        eprintln!(
            "breccia: GPU wait for {what} exceeded {} s — aborting instead of hanging the \
             device (AGENTS.md: a wedged wait on the display GPU costs the machine)",
            Self::WAIT_TIMEOUT_NS / 1_000_000_000
        );
        eprintln!("breccia: capture `journalctl -b -1 | grep -i amdgpu` before retrying");
        std::process::abort();
    }

    /// The per-dispatch barrier: fence a recorded compute dispatch against the one
    /// that follows it, at the narrowest scope that is still correct.
    ///
    /// A recorded graph dispatch always reads shader-storage data the previous
    /// dispatch wrote, so the dependency is `COMPUTE → COMPUTE` with
    /// `SHADER_WRITE → SHADER_READ|SHADER_WRITE`; that is *sufficient* for a
    /// compute-only command stream and it is what the measured numbers say to use
    /// (`ALL_COMMANDS → ALL_COMMANDS` costs 4.4 µs of launch serialisation on the
    /// MI50 for a dispatch whose arithmetic is 4-5 µs). `BRECCIA_BARRIER` selects
    /// the arm.
    ///
    /// # Safety
    ///
    /// `cmd` must be in the recording state on the thread that owns it, and every
    /// access the next recorded dispatch can make must be covered by a
    /// compute-stage fence with shader read/write masks — true for a compute-only
    /// dispatch stream, **not** true across a transfer (copies keep
    /// `Gpu::barrier_all`).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn barrier_compute_on(&self, cmd: vk::CommandBuffer) {
        match self.barrier_scope {
            BarrierScope::None => {}
            BarrierScope::Full => unsafe { Gpu::barrier_all(&self.device, cmd) },
            BarrierScope::Sync2 => {
                let Some(f) = self.pipeline_barrier2 else {
                    unsafe { Gpu::barrier_all(&self.device, cmd) };
                    return;
                };
                let mb = vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(
                        vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
                    );
                let dep = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mb));
                // SAFETY: `cmd` is recording on this thread and the dependency
                // covers every shader-visible access the next dispatch can make.
                unsafe { f(cmd, &dep as *const vk::DependencyInfo<'_> as *const _) };
            }
        }
    }

    /// Fence a **copy** against the dispatch stream, precisely: `pre` orders prior
    /// shader writes before the copy (COMPUTE -> COPY), `!pre` orders the copy's
    /// writes before the next dispatch (COPY -> COMPUTE).
    ///
    /// An `ALL_COMMANDS -> ALL_COMMANDS` barrier does both in one packet but forces
    /// a full launch serialisation, and the copy-dense passes (chunked prefill,
    /// merged experts) pay it 2-6 times per expert group — measured as a 10x
    /// regression in per-group GEMM time (§8x). The stage pair here is the same
    /// dependency without the global stall.
    ///
    /// # Safety
    ///
    /// `cmd` must be recording on the calling thread; `pre` must describe the
    /// caller's actual producer/consumer order.
    pub unsafe fn copy_dependency(&self, cmd: vk::CommandBuffer, pre: bool) {
        let (src_stage, src_access, dst_stage, dst_access) = if pre {
            (
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_WRITE,
            )
        } else {
            (
                vk::PipelineStageFlags2::COPY,
                vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_READ,
            )
        };
        let Some(f) = self.pipeline_barrier2 else {
            // No sync2: the coarse barrier is the only correct form available.
            unsafe { Gpu::barrier_all(&self.device, cmd) };
            return;
        };
        if self.barrier_scope == BarrierScope::None {
            return;
        }
        let mb = vk::MemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access);
        let dep = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&mb));
        // SAFETY: `cmd` is recording on this thread and the dependency is the
        // caller's declared producer/consumer pair.
        unsafe { f(cmd, &dep as *const vk::DependencyInfo<'_> as *const _) };
    }

    fn reset_record<F: FnOnce(&ash::Device, vk::CommandBuffer)>(
        &self,
        cmd: vk::CommandBuffer,
        f: F,
    ) -> R<()> {
        unsafe {
            vk(
                self.device
                    .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()),
                "vkResetCommandBuffer",
            )?;
            let bi = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            vk(
                self.device.begin_command_buffer(cmd, &bi),
                "vkBeginCommandBuffer",
            )?;
            f(&self.device, cmd);
            vk(self.device.end_command_buffer(cmd), "vkEndCommandBuffer")?;
        }
        Ok(())
    }

    /// Record `f` into the queue's command buffer and return it (already ended).
    pub fn record<F: FnOnce(&ash::Device, vk::CommandBuffer)>(&self, q: Qc, f: F) -> R<()> {
        let (_, cmd, _, _) = self.parts(q);
        self.reset_record(cmd, f)
    }

    /// Submit the queue's recorded command buffer; returns a fence to wait on.
    pub fn submit(&self, q: Qc) -> R<vk::Fence> {
        let (_, cmd, _, _) = self.parts(q);
        self.submit_cmd(q, cmd, None)
    }

    /// Submit `cmd` (which need not be the queue's own one-shot buffer), with an
    /// optional timeline-semaphore signal that is raised when **this
    /// submission** completes. Returns the fence for the submission.
    ///
    /// A signal attached to a submission is the mid-batch sync point: submit the
    /// work up to the marker with the signal, submit the rest immediately behind
    /// it on the same queue, and the host can wait for just the marker
    /// ([`Timeline::wait_host`]) while the second submission still runs.
    pub fn submit_cmd(
        &self,
        q: Qc,
        cmd: vk::CommandBuffer,
        signal: Option<(&Timeline, u64)>,
    ) -> R<vk::Fence> {
        let (queue, _, _, _) = self.parts(q);
        unsafe {
            let fence = vk(
                self.device
                    .create_fence(&vk::FenceCreateInfo::default(), None),
                "vkCreateFence",
            )?;
            let r = match signal {
                None => {
                    let cmds = [cmd];
                    let si = vk::SubmitInfo::default().command_buffers(&cmds);
                    self.device.queue_submit(queue, &[si], fence)
                }
                // A timeline value can only ride a submission through
                // VkSubmitInfo2 (VUID-VkSubmitInfo-pSignalSemaphores-03242):
                // VkSubmitInfo has no `pSignalSemaphoreValues`.
                Some((tl, value)) => {
                    let sig = [vk::SemaphoreSubmitInfo::default()
                        .semaphore(tl.sem)
                        .value(value)
                        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
                    let cbs = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];
                    let si = vk::SubmitInfo2::default()
                        .command_buffer_infos(&cbs)
                        .signal_semaphore_infos(&sig);
                    match self.queue_submit2 {
                        None => Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT),
                        Some(f) => {
                            let r = (f)(
                                queue,
                                1,
                                &si as *const vk::SubmitInfo2<'_> as *const _,
                                fence,
                            );
                            if r == vk::Result::SUCCESS {
                                Ok(())
                            } else {
                                Err(r)
                            }
                        }
                    }
                }
            };
            if let Err(e) = r {
                self.device.destroy_fence(fence, None);
                return Err(format!("vkQueueSubmit: {e}"));
            }
            if let Some((tl, value)) = signal {
                tl.advanced(value);
            }
            Ok(fence)
        }
    }

    /// A timeline semaphore at `initial`, for [`submit_cmd`]'s signal and
    /// [`Timeline::wait_host`].
    pub fn timeline(&self, initial: u64) -> R<Timeline> {
        if !self.feats.timeline_semaphore {
            return Err(format!(
                "device {} does not support VK_KHR_timeline_semaphore, which the \
                 mid-batch sync needs",
                self.index
            ));
        }
        let sem = vk(
            unsafe {
                self.device.create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(
                        &mut vk::SemaphoreTypeCreateInfo::default()
                            .semaphore_type(vk::SemaphoreType::TIMELINE)
                            .initial_value(initial),
                    ),
                    None,
                )
            },
            "vkCreateSemaphore(TIMELINE)",
        )?;
        Ok(Timeline {
            sem,
            value: std::cell::Cell::new(initial),
        })
    }

    pub fn wait_fence(&self, fence: vk::Fence) -> R<()> {
        let ok = wait_fence_bounded(&self.device, fence, "a submitted fence");
        unsafe { self.device.destroy_fence(fence, None) };
        if !ok {
            Self::fault_abort("a submitted fence");
        }
        Ok(())
    }

    /// Record + submit + wait, returning wall-clock seconds for the whole span.
    pub fn wall_run<F: FnOnce(&ash::Device, vk::CommandBuffer)>(&self, q: Qc, f: F) -> R<f64> {
        self.record(q, f)?;
        let t0 = std::time::Instant::now();
        let fence = self.submit(q)?;
        self.wait_fence(fence)?;
        Ok(t0.elapsed().as_secs_f64())
    }

    /// Timed run: brackets the recorded payload with queue timestamps
    /// (vkCmdWriteTimestamp at TOP/BOTTOM_OF_PIPE, read back with
    /// vkGetQueryPoolResults TYPE_64|WAIT). Runs a 1-iteration probe first, then
    /// scales the repeat count toward `target_s` so the measurement window is long
    /// enough to swamp submission overhead.
    pub fn run_timed<F: FnMut(&ash::Device, vk::CommandBuffer, u32)>(
        &self,
        q: Qc,
        target_s: f64,
        mut record: F,
    ) -> R<Timing> {
        let probe = self.time_once(q, &mut record, 1)?;
        let per = probe.per_iter_s.max(1e-9);
        let iters = ((target_s / per).ceil() as u64).clamp(1, 200_000) as u32;
        if iters == 1 {
            return Ok(probe);
        }
        self.time_once(q, &mut record, iters)
    }

    fn time_once<F: FnMut(&ash::Device, vk::CommandBuffer, u32)>(
        &self,
        q: Qc,
        record: &mut F,
        iters: u32,
    ) -> R<Timing> {
        let (_, cmd, timer, ts_ok) = self.parts(q);
        unsafe {
            self.device.reset_query_pool(timer, 0, 2);
        }
        self.reset_record(cmd, |d, cmd| unsafe {
            d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, timer, 0);
            record(d, cmd, iters);
            d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, timer, 1);
        })?;

        let t0 = std::time::Instant::now();
        let fence = self.submit(q)?;
        self.wait_fence(fence)?;
        let wall_s = t0.elapsed().as_secs_f64();

        let (gpu_s, used_ts) = if ts_ok && self.timestamp_period_ns > 0.0 {
            let mut ts = [0u64; 2];
            unsafe {
                vk(
                    self.device.get_query_pool_results(
                        timer,
                        0,
                        &mut ts,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    ),
                    "vkGetQueryPoolResults",
                )?;
            }
            let ticks = ts[1].saturating_sub(ts[0]) as f64;
            (ticks * self.timestamp_period_ns / 1e9, true)
        } else {
            (wall_s, false)
        };
        let per = gpu_s / f64::from(iters.max(1));
        Ok(Timing {
            iters,
            total_s: gpu_s,
            per_iter_s: per,
            wall_s,
            used_timestamps: used_ts,
        })
    }

    /// Copy `size` bytes src -> dst on the transfer-capable queue, once, and wait.
    pub fn copy_once(&self, src: &Buf, dst: &Buf, size: u64) -> R<()> {
        self.wall_run(Qc::X, |d, cmd| unsafe {
            Self::barrier_all(d, cmd);
            let region = vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size,
            };
            d.cmd_copy_buffer(cmd, src.buf, dst.buf, &[region]);
            Self::barrier_all(d, cmd);
        })?;
        Ok(())
    }

    pub fn wait_idle(&self) -> R<()> {
        unsafe { vk(self.device.device_wait_idle(), "vkDeviceWaitIdle") }
    }

    /// A `TIMESTAMP` query pool with `n` queries, for measuring a submission's
    /// own GPU span (as opposed to its fence wait, which also contains the host
    /// and driver's submission latency).
    pub fn query_pool(&self, n: u32) -> R<vk::QueryPool> {
        let qpci = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(n);
        vk(
            unsafe { self.device.create_query_pool(&qpci, None) },
            "vkCreateQueryPool(TIMESTAMP)",
        )
    }

    pub fn destroy_query_pool(&self, pool: vk::QueryPool) {
        unsafe { self.device.destroy_query_pool(pool, None) };
    }

    /// `2n` timestamps in **nanoseconds**: `[start_0, end_0, start_1, end_1, …]`.
    /// The caller must have waited for the submissions that wrote them.
    pub fn timestamps(&self, pool: vk::QueryPool, n: u32) -> R<Vec<f64>> {
        let mut ticks = vec![0u64; (2 * n) as usize];
        unsafe {
            vk(
                self.device.get_query_pool_results(
                    pool,
                    0,
                    &mut ticks,
                    vk::QueryResultFlags::TYPE_64,
                ),
                "vkGetQueryPoolResults",
            )?;
        }
        Ok(ticks
            .into_iter()
            .map(|t| t as f64 * self.timestamp_period_ns)
            .collect())
    }

    /// Reset the first `n` queries of `pool` (needs `hostQueryReset`, enabled on
    /// every device here).
    pub fn reset_queries(&self, pool: vk::QueryPool, n: u32) {
        unsafe { self.device.reset_query_pool(pool, 0, n) };
    }

    pub fn destroy(self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_query_pool(self.timer_x, None);
            self.device.destroy_query_pool(self.timer_c, None);
            self.device.destroy_command_pool(self.pool_x, None);
            self.device.destroy_command_pool(self.pool_c, None);
            self.device.destroy_device(None);
        }
    }
}

fn make_cmd_pool(device: &ash::Device, family: u32) -> R<(vk::CommandPool, vk::CommandBuffer)> {
    let (pool, cmds) = make_cmd_pool_n(device, family, 1)?;
    Ok((pool, cmds[0]))
}

/// A pool plus `n` primary command buffers allocated from it.
fn make_cmd_pool_n(
    device: &ash::Device,
    family: u32,
    n: usize,
) -> R<(vk::CommandPool, Vec<vk::CommandBuffer>)> {
    unsafe {
        let pci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = vk(
            device.create_command_pool(&pci, None),
            "vkCreateCommandPool",
        )?;
        let ai = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(n as u32);
        match device.allocate_command_buffers(&ai) {
            Ok(c) => Ok((pool, c)),
            Err(e) => {
                device.destroy_command_pool(pool, None);
                Err(format!("vkAllocateCommandBuffers: {e}"))
            }
        }
    }
}

fn make_timer(device: &ash::Device) -> R<vk::QueryPool> {
    unsafe {
        let qpci = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(4);
        vk(
            device.create_query_pool(&qpci, None),
            "vkCreateQueryPool(TIMESTAMP)",
        )
    }
}
