//! Serializable measurement types for the breccia Vulkan probe.
//!
//! Ported from `harnesses/vk-probe/src/report.rs` (schema documented in
//! `docs/phase-0/02-vulkan-raw-probe.md` §3.5), with `Deserialize` added so
//! `breccia-profile` can round-trip results and `uuid`/`pci_address` added to
//! [`DeviceId`] for the hardware fingerprint.

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Default, Clone, Debug, PartialEq)]
pub struct DeviceReport {
    pub device: DeviceId,
    pub properties: Properties,
    pub vram_bandwidth_gbps: Option<Bandwidth>,
    /// Second bandwidth pass with a small (cache-resident) working set.
    pub vram_bandwidth_cache_gbps: Option<Bandwidth>,
    pub compute_gflops: Option<Compute>,
    pub pcie: Option<Pcie>,
    pub overlap: Option<Overlap>,
    pub notes: Vec<String>,
    pub error: Option<String>,
}

#[derive(Serialize, Deserialize, Default, Clone, Debug, PartialEq)]
pub struct DeviceId {
    pub index: u32,
    pub name: String,
    pub vendor_id: String,
    pub device_id: String,
    /// `VkPhysicalDeviceIDProperties::deviceUUID` (stable per physical device).
    pub uuid: String,
    /// `dddd:bb:dd.f`, empty when neither `VK_EXT_pci_bus_info` nor sysfs matched.
    pub pci_address: String,
    pub device_type: String,
    pub api_version: String,
    pub driver_version: String,
    pub driver_name: String,
    pub driver_info: String,
}

#[derive(Serialize, Deserialize, Default, Clone, Debug, PartialEq)]
pub struct Properties {
    pub memory_heaps: Vec<Heap>,
    pub queue_families: Vec<QueueFamily>,
    pub subgroup: Subgroup,
    pub extensions: ExtPresence,
    pub features: FeatPresence,
    /// Nanoseconds per timestamp tick (vkPhysicalDeviceLimits::timestampPeriod).
    pub timestamp_period_ns: f32,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Heap {
    pub index: u32,
    pub size_bytes: u64,
    pub device_local: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct QueueFamily {
    pub index: u32,
    pub flags: Vec<String>,
    pub queue_count: u32,
    pub timestamp_valid_bits: u32,
    pub supports_compute: bool,
    pub supports_transfer: bool,
    pub supports_graphics: bool,
}

#[derive(Serialize, Deserialize, Default, Clone, Debug, PartialEq)]
pub struct Subgroup {
    pub subgroup_size: u32,
    pub supported_stages: Vec<String>,
    pub supported_operations: Vec<String>,
    pub quad_operations_in_all_stages: bool,
    /// Wave32..Wave64 size-control range: `32..64` = Wave32 selectable (gfx1030),
    /// `64..64` = Wave64 only (gfx906). 0 when VK_EXT_subgroup_size_control is absent.
    pub min_subgroup_size: u32,
    pub max_subgroup_size: u32,
    pub max_compute_workgroup_subgroups: u32,
}

#[derive(Serialize, Deserialize, Default, Clone, Debug, PartialEq)]
pub struct ExtPresence {
    pub khr_timeline_semaphore: bool,
    pub ext_external_memory_host: bool,
    pub khr_cooperative_matrix: bool,
    pub ext_subgroup_size_control: bool,
    pub khr_shader_atomic_int64: bool,
    pub total_device_extensions: u32,
}

#[derive(Serialize, Deserialize, Default, Clone, Debug, PartialEq)]
pub struct FeatPresence {
    pub shader_float16: bool,
    pub shader_int8: bool,
    pub shader_atomic_int64_buffer: bool,
    pub shader_atomic_int64_shared: bool,
    pub timeline_semaphore: bool,
    pub subgroup_size_control: bool,
    pub compute_full_subgroups: bool,
    pub cooperative_matrix: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Bandwidth {
    /// Bytes per buffer (there are two: src + dst).
    pub buffer_bytes: u64,
    pub iterations: u32,
    pub seconds: f64,
    /// Wall-clock seconds for the same window (cross-check on the timestamp result).
    pub wall_seconds: f64,
    pub read_gbps: f64,
    pub write_gbps: f64,
    pub copy_gbps: f64,
    pub notes: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Compute {
    pub fp32_gflops: f64,
    pub fp16_gflops: Option<f64>,
    pub fp16_supported: bool,
    pub threads: u32,
    pub fma_chains: u32,
    pub fma_rounds_per_dispatch: u32,
    pub seconds_per_dispatch: f64,
    pub notes: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Pcie {
    pub queue_family: u32,
    pub queue_index: u32,
    pub distinct_from_compute_queue: bool,
    pub staging_memory: String,
    pub device_memory: String,
    pub sizes: Vec<PcieSize>,
    pub notes: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct PcieSize {
    pub size_mb: u64,
    pub h2d_gbps: f64,
    pub d2h_gbps: f64,
    pub h2d_seconds_per_copy: f64,
    pub d2h_seconds_per_copy: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Overlap {
    pub compute_queue_family: u32,
    pub transfer_queue_family: u32,
    pub queue_mode: String,
    pub compute_seconds: f64,
    pub transfer_seconds: f64,
    pub both_seconds: f64,
    pub efficiency_percent: f64,
    pub formula: String,
    pub compute_work_seconds: f64,
    pub transfer_work_bytes: u64,
    pub notes: Vec<String>,
}
