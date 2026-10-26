//! The typed launch plan and its canonical re-rendering.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::device::{
    Backend, BackendHint, DeviceRef, DeviceResolution, LaunchWarning, ResolvedDevice, WarningCode,
};
use crate::error::FlagError;
use crate::overrides::{validate_overrides, CompiledOverride};
use crate::value::{KvCacheType, ReasoningMode, Switch};

/// A fully typed llama.cpp-compatible `llama-server` launch plan.
///
/// Every field corresponds to one flag from the production launch line (see
/// `docs/phase-1/02-flags-report.md` for the flag↔field table). `Option` means
/// "flag absent — use llama.cpp's own default"; no defaults are invented here so
/// that a plan can be diffed against the reference engine's behaviour.
///
/// Unknown flags are rejected at parse time, so this struct is the complete
/// surface Breccia honours. `-ot` entries keep their raw pattern text and are
/// compiled by [`LaunchPlan::validated_overrides`].
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct LaunchPlan {
    /// Engine backend (`--backend`, default [`Backend::Vulkan`]).
    ///
    /// Not a llama.cpp flag; the production line omits it and gets Vulkan.
    pub backend: Backend,

    /// `-D` — path to the server binary being described.
    ///
    /// Fork-specific: `qwen-dual-serve.sh` uses `-D` to name the binary itself.
    /// Breccia records it so a plan round-trips through the same line.
    pub binary: Option<PathBuf>,

    /// `-m` — primary GGUF (shard 1 of N).
    pub model: Option<PathBuf>,

    /// `--device` — devices the engine may use, in priority order.
    pub devices: Vec<DeviceRef>,

    /// `-ngl` — layers to offload to GPU (`99` = effectively all).
    pub gpu_layers: Option<i32>,

    /// `-ts` — per-device tensor split fractions, e.g. `9,15`.
    pub tensor_split: Option<Vec<f32>>,

    /// `-ot` — per-tensor device overrides as `(regex, device)` pairs.
    pub override_tensors: Vec<(String, DeviceRef)>,

    /// `-cmoe` — keep MoE expert tensors on the CPU unless overridden.
    pub cpu_moe: bool,

    /// `--moe-cache-mib` — MoE expert cache budget in MiB (`0` = disabled).
    pub moe_cache_mib: Option<u64>,

    /// `-lzm` — fork-specific load/layer switch, `on`/`off`.
    pub lzm: Option<Switch>,

    /// `-lm` — fork-specific mode selector; value passed through verbatim
    /// (production uses `none`). Kept as a string because its value set is
    /// owned by the reference fork and is not documented upstream.
    pub lm: Option<String>,

    /// `--no-host` — do not keep host (CPU) copies of offloaded weights.
    pub no_host: bool,

    /// `-c` — context size in tokens.
    pub ctx_size: Option<u32>,

    /// `-kvu` — unified (single) KV cache across slots.
    pub kv_unified: bool,

    /// `-np` — number of parallel slots.
    pub parallel: Option<u32>,

    /// `--kv-unified-per-slot` — per-slot context inside the unified KV cache.
    pub kv_unified_per_slot: Option<u32>,

    /// `-ctk` — K-cache element type.
    pub cache_type_k: Option<KvCacheType>,

    /// `-ctv` — V-cache element type.
    pub cache_type_v: Option<KvCacheType>,

    /// `-fa` — flash attention.
    pub flash_attn: Option<Switch>,

    /// `-cram` — fork-specific context-RAM budget (production uses `0`).
    pub cram: Option<u64>,

    /// `-ctxcp` — fork-specific context-copy switch (production uses `1`).
    pub ctxcp: Option<u32>,

    /// `--slot-save-path` — directory for persisted slot/KV state.
    pub slot_save_path: Option<PathBuf>,

    /// `--mmproj` — multimodal projector GGUF.
    pub mmproj: Option<PathBuf>,

    /// `--mmdev` — device that hosts the projector.
    pub mmproj_device: Option<DeviceRef>,

    /// `--image-min-tokens` — lower bound on image token count.
    pub image_min_tokens: Option<u32>,

    /// `--image-max-tokens` — upper bound on image token count.
    pub image_max_tokens: Option<u32>,

    /// `-lv` — log verbosity.
    pub log_verbosity: Option<u32>,

    /// `--reasoning` — reasoning-trace mode.
    pub reasoning: Option<ReasoningMode>,

    /// `-t` — generation threads.
    pub threads: Option<u32>,

    /// `-tb` — batch/prefill threads.
    pub threads_batch: Option<u32>,

    /// `--host` — bind address.
    pub host: Option<String>,

    /// `--port` — bind port.
    pub port: Option<u16>,

    /// `--jinja` — use the model's Jinja chat template.
    pub jinja: bool,

    /// `--metrics` — expose Prometheus metrics.
    pub metrics: bool,

    /// `--fit` — fork-specific memory-fitting switch.
    pub fit: Option<Switch>,

    /// `--no-op-offload` — do not offload the output projection.
    pub no_op_offload: bool,

    /// `-ub` — micro-batch (ubatch) size.
    pub ubatch_size: Option<u32>,
}

impl LaunchPlan {
    /// Resolve every device reference against the plan's [`Backend`].
    ///
    /// Convenience wrapper over [`LaunchPlan::resolve_devices_detailed`] for
    /// callers that do not need the warnings.
    pub fn resolve_devices(&self) -> Vec<ResolvedDevice> {
        self.resolve_devices_detailed().resolved
    }

    /// Resolve devices and collect the structured warnings that go with it.
    ///
    /// Rules (see the phase report for the table):
    ///
    /// | backend | alias | resolved |
    /// |---|---|---|
    /// | `vulkan` | `ROCmN`/`CUDAN`/`VulkanN`/`GPU.N` | Vulkan device `N` |
    /// | `vulkan` | `CPU` | CPU |
    /// | `hip` | `ROCmN`/`CUDAN`/`VulkanN`/`GPU.N` | HIP device `N` |
    /// | `hip` | `CPU` | CPU |
    /// | `auto` | `ROCmN`/`CUDAN` | HIP device `N` |
    /// | `auto` | `VulkanN`/`GPU.N` | Vulkan device `N` |
    /// | `auto` | `CPU` | CPU |
    ///
    /// Resolution is **ordinal-preserving**: index `N` always stays `N`. It never
    /// re-orders devices — that is the profile/partition layer's job, not the
    /// flag layer's. Cross-family aliases are warnings, not errors.
    pub fn resolve_devices_detailed(&self) -> DeviceResolution {
        let mut resolved = Vec::with_capacity(self.devices.len());
        let mut warnings = Vec::new();

        for requested in &self.devices {
            let (backend, code) = match (self.backend, requested.backend_hint) {
                (_, BackendHint::Cpu) => (BackendHint::Cpu, None),
                (Backend::Vulkan, BackendHint::Hip) => {
                    (BackendHint::Vulkan, Some(WarningCode::HipAliasOnVulkan))
                }
                (Backend::Vulkan, BackendHint::Vulkan) => (BackendHint::Vulkan, None),
                (Backend::Hip, BackendHint::Hip) => (BackendHint::Hip, None),
                (Backend::Hip, BackendHint::Vulkan) => {
                    (BackendHint::Hip, Some(WarningCode::VulkanAliasOnHip))
                }
                (Backend::Auto, hint) => (hint, None),
            };

            let device = ResolvedDevice {
                requested: *requested,
                backend,
                index: requested.index,
            };
            resolved.push(device);

            if let Some(code) = code {
                warnings.push(LaunchWarning {
                    code,
                    requested: *requested,
                    resolved_backend: backend,
                    resolved_index: requested.index,
                    message: alias_warning_message(code, requested, backend),
                });
            }
        }

        DeviceResolution { resolved, warnings }
    }

    /// Structured warnings implied by this plan's device aliases.
    pub fn warnings(&self) -> Vec<LaunchWarning> {
        self.resolve_devices_detailed().warnings
    }

    /// Compile the `-ot` patterns, reporting every failure with its index.
    pub fn validated_overrides(&self) -> Result<Vec<CompiledOverride>, Vec<FlagError>> {
        validate_overrides(&self.override_tensors)
    }

    /// Re-emit an equivalent llama.cpp argv (no program name).
    ///
    /// Output is canonical: one fixed flag order, `--long` spellings wherever the
    /// reference engine has one, fork-only short flags kept as-is, values
    /// space-separated (never `=`, so a value starting with `-` stays unambiguous),
    /// and absent flags omitted. Parse→render→parse is a fixed point, which is
    /// what the round-trip test asserts.
    pub fn render_args(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();

        if let Some(binary) = &self.binary {
            push_pair(&mut out, "-D", &path_str(binary));
        }
        if let Some(model) = &self.model {
            push_pair(&mut out, "-m", &path_str(model));
        }
        if self.backend != Backend::Vulkan {
            push_pair(&mut out, "--backend", &self.backend.to_string());
        }
        if !self.devices.is_empty() {
            let list = self
                .devices
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            push_pair(&mut out, "--device", &list);
        }
        if let Some(layers) = self.gpu_layers {
            push_pair(&mut out, "-ngl", &layers.to_string());
        }
        if let Some(split) = &self.tensor_split {
            let list = split
                .iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>()
                .join(",");
            push_pair(&mut out, "-ts", &list);
        }
        if !self.override_tensors.is_empty() {
            let list = self
                .override_tensors
                .iter()
                .map(|(pattern, device)| format!("{pattern}={device}"))
                .collect::<Vec<_>>()
                .join(",");
            push_pair(&mut out, "-ot", &list);
        }
        if self.cpu_moe {
            out.push("-cmoe".to_string());
        }
        if let Some(mib) = self.moe_cache_mib {
            push_pair(&mut out, "--moe-cache-mib", &mib.to_string());
        }
        if let Some(lzm) = self.lzm {
            push_pair(&mut out, "-lzm", &lzm.to_string());
        }
        if let Some(lm) = &self.lm {
            push_pair(&mut out, "-lm", lm);
        }
        if self.no_host {
            out.push("--no-host".to_string());
        }
        if let Some(ctx) = self.ctx_size {
            push_pair(&mut out, "-c", &ctx.to_string());
        }
        if self.kv_unified {
            out.push("-kvu".to_string());
        }
        if let Some(np) = self.parallel {
            push_pair(&mut out, "-np", &np.to_string());
        }
        if let Some(per_slot) = self.kv_unified_per_slot {
            push_pair(&mut out, "--kv-unified-per-slot", &per_slot.to_string());
        }
        if let Some(ctk) = self.cache_type_k {
            push_pair(&mut out, "-ctk", &ctk.to_string());
        }
        if let Some(ctv) = self.cache_type_v {
            push_pair(&mut out, "-ctv", &ctv.to_string());
        }
        if let Some(fa) = self.flash_attn {
            push_pair(&mut out, "--flash-attn", &fa.to_string());
        }
        if let Some(cram) = self.cram {
            push_pair(&mut out, "-cram", &cram.to_string());
        }
        if let Some(ctxcp) = self.ctxcp {
            push_pair(&mut out, "-ctxcp", &ctxcp.to_string());
        }
        if let Some(path) = &self.slot_save_path {
            push_pair(&mut out, "--slot-save-path", &path_str(path));
        }
        if let Some(mmproj) = &self.mmproj {
            push_pair(&mut out, "--mmproj", &path_str(mmproj));
        }
        if let Some(device) = self.mmproj_device {
            push_pair(&mut out, "-mmdev", &device.to_string());
        }
        if let Some(min) = self.image_min_tokens {
            push_pair(&mut out, "--image-min-tokens", &min.to_string());
        }
        if let Some(max) = self.image_max_tokens {
            push_pair(&mut out, "--image-max-tokens", &max.to_string());
        }
        if let Some(level) = self.log_verbosity {
            push_pair(&mut out, "-lv", &level.to_string());
        }
        if let Some(reasoning) = self.reasoning {
            push_pair(&mut out, "--reasoning", &reasoning.to_string());
        }
        if let Some(threads) = self.threads {
            push_pair(&mut out, "-t", &threads.to_string());
        }
        if let Some(threads) = self.threads_batch {
            push_pair(&mut out, "-tb", &threads.to_string());
        }
        if let Some(host) = &self.host {
            push_pair(&mut out, "--host", host);
        }
        if let Some(port) = self.port {
            push_pair(&mut out, "--port", &port.to_string());
        }
        if self.jinja {
            out.push("--jinja".to_string());
        }
        if self.metrics {
            out.push("--metrics".to_string());
        }
        if let Some(fit) = self.fit {
            push_pair(&mut out, "--fit", &fit.to_string());
        }
        if self.no_op_offload {
            out.push("--no-op-offload".to_string());
        }
        if let Some(ubatch) = self.ubatch_size {
            push_pair(&mut out, "-ub", &ubatch.to_string());
        }

        out
    }
}

fn push_pair(out: &mut Vec<String>, flag: &str, value: &str) {
    out.push(flag.to_string());
    out.push(value.to_string());
}

fn path_str(path: &std::path::Path) -> String {
    path.to_string_lossy().into_owned()
}

fn alias_warning_message(code: WarningCode, requested: &DeviceRef, backend: BackendHint) -> String {
    match code {
        WarningCode::HipAliasOnVulkan => format!(
            "device `{requested}` is a HIP/CUDA alias but the plan backend is vulkan; \
             resolving to Vulkan device {} (HIP launch flags are accepted, backend stays vulkan)",
            requested.index
        ),
        WarningCode::VulkanAliasOnHip => format!(
            "device `{requested}` is a Vulkan alias but the plan backend is hip; \
             resolving to {backend} device {}",
            requested.index
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::BackendHint;

    fn plan_with(backend: Backend, aliases: &[&str]) -> LaunchPlan {
        LaunchPlan {
            backend,
            devices: aliases
                .iter()
                .map(|a| DeviceRef::parse(a).unwrap())
                .collect(),
            ..LaunchPlan::default()
        }
    }

    /// Full backend × alias matrix. Every cell is asserted, and remap warnings are
    /// asserted in both directions.
    #[test]
    fn backend_alias_resolution_matrix() {
        // (alias, backend, expected backend, expected index, expect warning)
        let cases: &[(&str, Backend, BackendHint, u32, bool)] = &[
            // Vulkan (default) — HIP aliases accepted, remapped, warned.
            ("ROCm0", Backend::Vulkan, BackendHint::Vulkan, 0, true),
            ("ROCm1", Backend::Vulkan, BackendHint::Vulkan, 1, true),
            ("CUDA2", Backend::Vulkan, BackendHint::Vulkan, 2, true),
            ("Vulkan0", Backend::Vulkan, BackendHint::Vulkan, 0, false),
            ("GPU.1", Backend::Vulkan, BackendHint::Vulkan, 1, false),
            ("CPU", Backend::Vulkan, BackendHint::Cpu, 0, false),
            // HIP backend — native aliases silent, Vulkan aliases remapped+warned.
            ("ROCm0", Backend::Hip, BackendHint::Hip, 0, false),
            ("ROCm1", Backend::Hip, BackendHint::Hip, 1, false),
            ("CUDA2", Backend::Hip, BackendHint::Hip, 2, false),
            ("Vulkan0", Backend::Hip, BackendHint::Hip, 0, true),
            ("GPU.1", Backend::Hip, BackendHint::Hip, 1, true),
            ("CPU", Backend::Hip, BackendHint::Cpu, 0, false),
            // Auto — alias family wins, nothing is remapped.
            ("ROCm0", Backend::Auto, BackendHint::Hip, 0, false),
            ("ROCm1", Backend::Auto, BackendHint::Hip, 1, false),
            ("CUDA2", Backend::Auto, BackendHint::Hip, 2, false),
            ("Vulkan0", Backend::Auto, BackendHint::Vulkan, 0, false),
            ("GPU.1", Backend::Auto, BackendHint::Vulkan, 1, false),
            ("CPU", Backend::Auto, BackendHint::Cpu, 0, false),
        ];

        for &(alias, backend, want_backend, want_index, want_warning) in cases {
            let plan = plan_with(backend, &[alias]);
            let resolution = plan.resolve_devices_detailed();
            let dev = resolution.resolved[0];
            assert_eq!(dev.backend, want_backend, "{alias} on {backend}: backend");
            assert_eq!(dev.index, want_index, "{alias} on {backend}: index");
            assert_eq!(dev.requested, DeviceRef::parse(alias).unwrap());
            assert_eq!(
                dev.is_remapped(),
                want_warning,
                "{alias} on {backend}: is_remapped"
            );

            // The literal-signature API must agree with the detailed one.
            assert_eq!(plan.resolve_devices(), resolution.resolved);

            if want_warning {
                assert_eq!(resolution.warnings.len(), 1, "{alias} on {backend}");
                let warning = &resolution.warnings[0];
                let expected_code = match backend {
                    Backend::Vulkan => WarningCode::HipAliasOnVulkan,
                    Backend::Hip => WarningCode::VulkanAliasOnHip,
                    Backend::Auto => unreachable!("auto never warns"),
                };
                assert_eq!(warning.code, expected_code);
                assert_eq!(warning.resolved_index, want_index);
                assert_eq!(warning.resolved_backend, want_backend);
                assert!(
                    warning.message.contains("resolves") || warning.message.contains("resolving")
                );
            } else {
                assert!(
                    resolution.warnings.is_empty(),
                    "{alias} on {backend} should not warn: {:?}",
                    resolution.warnings
                );
            }
        }
    }

    #[test]
    fn resolution_preserves_order_and_duplicates() {
        let plan = plan_with(Backend::Vulkan, &["ROCm0", "ROCm1", "ROCm0"]);
        let resolved = plan.resolve_devices();
        assert_eq!(resolved.len(), 3);
        assert_eq!(resolved[0].index, 0);
        assert_eq!(resolved[1].index, 1);
        assert_eq!(resolved[2].index, 0);
        assert_eq!(plan.warnings().len(), 3);
    }

    #[test]
    fn empty_device_list_resolves_to_nothing() {
        let plan = LaunchPlan::default();
        assert!(plan.resolve_devices().is_empty());
        assert!(plan.warnings().is_empty());
    }

    #[test]
    fn render_omits_absent_flags_and_defaults() {
        assert!(LaunchPlan::default().render_args().is_empty());
    }

    #[test]
    fn render_emits_non_default_backend_only() {
        let plan = LaunchPlan {
            backend: Backend::Hip,
            ..LaunchPlan::default()
        };
        assert_eq!(plan.render_args(), vec!["--backend", "hip"]);
        let plan = LaunchPlan {
            backend: Backend::Auto,
            ..LaunchPlan::default()
        };
        assert_eq!(plan.render_args(), vec!["--backend", "auto"]);
    }

    #[test]
    fn render_is_canonical_for_every_field() {
        let plan = LaunchPlan {
            backend: Backend::Vulkan,
            binary: Some(PathBuf::from("/bin/llama-server")),
            model: Some(PathBuf::from("/m/model.gguf")),
            devices: vec![DeviceRef::parse("ROCm0").unwrap()],
            gpu_layers: Some(99),
            tensor_split: Some(vec![9.0, 15.0]),
            override_tensors: vec![(r"blk\.0\.".to_string(), DeviceRef::cpu())],
            cpu_moe: true,
            moe_cache_mib: Some(0),
            lzm: Some(Switch::On),
            lm: Some("none".to_string()),
            no_host: true,
            ctx_size: Some(200_000),
            kv_unified: true,
            parallel: Some(1),
            kv_unified_per_slot: Some(200_000),
            cache_type_k: Some(KvCacheType::Q4_0),
            cache_type_v: Some(KvCacheType::Q4_0),
            flash_attn: Some(Switch::On),
            cram: Some(0),
            ctxcp: Some(1),
            slot_save_path: Some(PathBuf::from("/slots/")),
            mmproj: Some(PathBuf::from("/m/mmproj.gguf")),
            mmproj_device: Some(DeviceRef::parse("ROCm0").unwrap()),
            image_min_tokens: Some(1024),
            image_max_tokens: Some(2048),
            log_verbosity: Some(4),
            reasoning: Some(ReasoningMode::Off),
            threads: Some(16),
            threads_batch: Some(16),
            host: Some("127.0.0.1".to_string()),
            port: Some(8160),
            jinja: true,
            metrics: true,
            fit: Some(Switch::Off),
            no_op_offload: true,
            ubatch_size: Some(256),
        };
        let argv = plan.render_args();
        let expected: Vec<String> = [
            "-D",
            "/bin/llama-server",
            "-m",
            "/m/model.gguf",
            "--device",
            "ROCm0",
            "-ngl",
            "99",
            "-ts",
            "9,15",
            "-ot",
            r"blk\.0\.=CPU",
            "-cmoe",
            "--moe-cache-mib",
            "0",
            "-lzm",
            "on",
            "-lm",
            "none",
            "--no-host",
            "-c",
            "200000",
            "-kvu",
            "-np",
            "1",
            "--kv-unified-per-slot",
            "200000",
            "-ctk",
            "q4_0",
            "-ctv",
            "q4_0",
            "--flash-attn",
            "on",
            "-cram",
            "0",
            "-ctxcp",
            "1",
            "--slot-save-path",
            "/slots/",
            "--mmproj",
            "/m/mmproj.gguf",
            "-mmdev",
            "ROCm0",
            "--image-min-tokens",
            "1024",
            "--image-max-tokens",
            "2048",
            "-lv",
            "4",
            "--reasoning",
            "off",
            "-t",
            "16",
            "-tb",
            "16",
            "--host",
            "127.0.0.1",
            "--port",
            "8160",
            "--jinja",
            "--metrics",
            "--fit",
            "off",
            "--no-op-offload",
            "-ub",
            "256",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(argv, expected);
    }

    #[test]
    fn plan_serialises_and_deserialises() {
        let plan = plan_with(Backend::Vulkan, &["ROCm0", "ROCm1"]);
        let json = serde_json::to_string(&plan).unwrap();
        let back: LaunchPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(plan, back);
        // Missing fields fall back to defaults, so partial configs are loadable.
        let partial: LaunchPlan = serde_json::from_str("{}").unwrap();
        assert_eq!(partial, LaunchPlan::default());
        assert_eq!(partial.backend, Backend::Vulkan);
    }

    #[test]
    fn warnings_have_machine_readable_codes() {
        let plan = plan_with(Backend::Vulkan, &["ROCm1"]);
        let json = serde_json::to_string(&plan.warnings()).unwrap();
        assert!(json.contains("hip_alias_on_vulkan"), "{json}");
        assert!(json.contains("\"resolved_index\":1"), "{json}");
    }
}
