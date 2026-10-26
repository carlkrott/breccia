//! Device aliases, backend selection, and alias→device resolution.
//!
//! llama.cpp names devices after the backend that enumerated them (`ROCm0`,
//! `CUDA1`, `Vulkan0`, `GPU.0`, `CPU`). Breccia must keep accepting those
//! spellings because the production launch line is written for the HIP build,
//! while the *engine* runs on Vulkan by default. The parser therefore keeps the
//! alias family ([`BackendHint`]) and the ordinal separately, and resolution
//! (which backend actually serves the device) is a separate, testable step.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// The backend family a device *alias* came from.
///
/// This is the spelling family, not the backend the device will be served by —
/// see [`crate::LaunchPlan::resolve_devices`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendHint {
    /// `ROCmN`, `HIPN`, `CUDAN` — llama.cpp's HIP/CUDA enumeration.
    Hip,
    /// `VulkanN`, `GPU.N` — llama.cpp's Vulkan enumeration.
    Vulkan,
    /// `CPU` — the host device (no ordinal).
    Cpu,
}

impl BackendHint {
    /// Canonical alias prefix for this family, e.g. `ROCm` for [`BackendHint::Hip`].
    pub const fn alias_prefix(self) -> &'static str {
        match self {
            BackendHint::Hip => "ROCm",
            BackendHint::Vulkan => "Vulkan",
            BackendHint::Cpu => "CPU",
        }
    }
}

impl fmt::Display for BackendHint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BackendHint::Hip => "hip",
            BackendHint::Vulkan => "vulkan",
            BackendHint::Cpu => "cpu",
        })
    }
}

/// One device reference from the launch line: an alias family plus an ordinal.
///
/// Parsing normalises the spelling (`CUDAN`, `HIPN` and `GPU.0` are stored as
/// their canonical family + index), so `ROCm01` and `ROCm1` compare equal. The
/// original spelling is deliberately not retained — resolution only depends on
/// (family, index).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceRef {
    /// Alias family the reference was written in.
    pub backend_hint: BackendHint,
    /// 0-based device ordinal; always `0` for [`BackendHint::Cpu`].
    pub index: u32,
}

impl DeviceRef {
    /// Build a reference from a family and ordinal.
    pub const fn new(backend_hint: BackendHint, index: u32) -> Self {
        Self {
            backend_hint,
            index,
        }
    }

    /// The host CPU device.
    pub const fn cpu() -> Self {
        Self::new(BackendHint::Cpu, 0)
    }

    /// `true` for the CPU pseudo-device.
    pub const fn is_cpu(self) -> bool {
        matches!(self.backend_hint, BackendHint::Cpu)
    }

    /// Parse one alias: `ROCmN`, `HIPN`, `CUDAN`, `VulkanN`, `GPU.N`, `GPUN`, `CPU`.
    ///
    /// Case-insensitive; the ordinal must be a non-negative decimal integer that
    /// fits in `u32`. Returns a human-readable reason on failure (callers wrap it
    /// in [`crate::FlagError::InvalidDevice`] with the flag position).
    pub fn parse(spec: &str) -> Result<Self, String> {
        let trimmed = spec.trim();
        if trimmed.is_empty() {
            return Err("empty device reference".to_string());
        }
        let upper = trimmed.to_ascii_uppercase();
        if upper == "CPU" {
            return Ok(Self::cpu());
        }

        let (backend_hint, digits) = if let Some(rest) = upper.strip_prefix("ROCM") {
            (BackendHint::Hip, rest)
        } else if let Some(rest) = upper.strip_prefix("HIP") {
            (BackendHint::Hip, rest)
        } else if let Some(rest) = upper.strip_prefix("CUDA") {
            (BackendHint::Hip, rest)
        } else if let Some(rest) = upper.strip_prefix("VULKAN") {
            (BackendHint::Vulkan, rest)
        } else if let Some(rest) = upper.strip_prefix("GPU.") {
            (BackendHint::Vulkan, rest)
        } else if let Some(rest) = upper.strip_prefix("GPU") {
            (BackendHint::Vulkan, rest)
        } else {
            return Err(format!(
                "`{trimmed}` has no recognised backend prefix \
                 (expected ROCmN, HIPN, CUDAN, VulkanN, GPU.N or CPU)"
            ));
        };

        if digits.is_empty() {
            return Err(format!(
                "`{trimmed}` is missing an index (expected a non-negative integer after the backend prefix)"
            ));
        }
        if let Some(rest) = digits.strip_prefix('-') {
            return Err(format!(
                "`{trimmed}` has negative index `-{rest}`; device indices must be non-negative integers"
            ));
        }
        if !digits.bytes().all(|b| b.is_ascii_digit()) {
            return Err(format!(
                "`{trimmed}` has non-numeric index `{digits}`; device indices must be non-negative integers"
            ));
        }
        let index: u32 = digits
            .parse()
            .map_err(|_| format!("`{trimmed}` index `{digits}` does not fit in a 32-bit index"))?;
        Ok(Self {
            backend_hint,
            index,
        })
    }
}

impl fmt::Display for DeviceRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.backend_hint {
            BackendHint::Cpu => f.write_str("CPU"),
            hint => write!(f, "{}{}", hint.alias_prefix(), self.index),
        }
    }
}

impl FromStr for DeviceRef {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

/// Engine backend selection for the plan (`--backend`, default `vulkan`).
///
/// Per the Phase-1 decision note, Vulkan is the primary backend while the
/// llama.cpp/HIP flag surface keeps working unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// Vulkan/RADV (default). HIP-flavoured aliases are accepted and remapped.
    #[default]
    Vulkan,
    /// ROCm/HIP — the llama.cpp baseline backend.
    Hip,
    /// Take the backend from each device alias (`ROCmN`→HIP, `VulkanN`→Vulkan).
    Auto,
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Backend::Vulkan => "vulkan",
            Backend::Hip => "hip",
            Backend::Auto => "auto",
        })
    }
}

impl FromStr for Backend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "vulkan" | "vk" | "radv" => Ok(Backend::Vulkan),
            "hip" | "rocm" | "cuda" => Ok(Backend::Hip),
            "auto" => Ok(Backend::Auto),
            _ => Err(format!(
                "`{}` is not a backend; expected one of vulkan, hip, auto",
                s.trim()
            )),
        }
    }
}

/// A device reference after the plan's [`Backend`] has been applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedDevice {
    /// The reference exactly as parsed from the launch line.
    pub requested: DeviceRef,
    /// Backend that will actually own the device.
    pub backend: BackendHint,
    /// Ordinal on that backend (equal to `requested.index`; ordinal-preserving).
    pub index: u32,
}

impl ResolvedDevice {
    /// The resolved device as a canonical reference.
    pub const fn effective(&self) -> DeviceRef {
        DeviceRef::new(self.backend, self.index)
    }

    /// `true` when resolution changed the alias family.
    pub const fn is_remapped(&self) -> bool {
        !matches!(
            (self.requested.backend_hint, self.backend),
            (BackendHint::Cpu, BackendHint::Cpu)
                | (BackendHint::Hip, BackendHint::Hip)
                | (BackendHint::Vulkan, BackendHint::Vulkan)
        )
    }
}

impl fmt::Display for ResolvedDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.backend == BackendHint::Cpu {
            return f.write_str("CPU");
        }
        write!(
            f,
            "{}:{}{}",
            self.backend,
            self.backend.alias_prefix(),
            self.index
        )
    }
}

/// Machine-readable reason for a non-fatal launch warning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarningCode {
    /// A HIP-flavoured alias (`ROCmN`/`CUDAN`) was launched on the Vulkan backend.
    HipAliasOnVulkan,
    /// A Vulkan-flavoured alias (`VulkanN`/`GPU.N`) was launched on the HIP backend.
    VulkanAliasOnHip,
}

/// A structured, non-fatal launch warning.
///
/// The Phase-1 decision requires `ROCm0`/`ROCm1` to keep working on the Vulkan
/// backend. That is a warning — recorded here — never a parse error.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchWarning {
    /// Stable machine-readable code.
    pub code: WarningCode,
    /// Device reference as written.
    pub requested: DeviceRef,
    /// Backend that will serve the device.
    pub resolved_backend: BackendHint,
    /// Ordinal on that backend.
    pub resolved_index: u32,
    /// Human-readable sentence for logs.
    pub message: String,
}

impl fmt::Display for LaunchWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

/// Full result of resolving a plan's device list: devices plus warnings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceResolution {
    /// Resolved devices, in launch-line order (duplicates preserved).
    pub resolved: Vec<ResolvedDevice>,
    /// Non-fatal warnings, in device order.
    pub warnings: Vec<LaunchWarning>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_alias_families() {
        assert_eq!(
            DeviceRef::parse("ROCm0").unwrap(),
            DeviceRef::new(BackendHint::Hip, 0)
        );
        assert_eq!(
            DeviceRef::parse("ROCm1").unwrap(),
            DeviceRef::new(BackendHint::Hip, 1)
        );
        assert_eq!(
            DeviceRef::parse("rocm12").unwrap(),
            DeviceRef::new(BackendHint::Hip, 12)
        );
        assert_eq!(
            DeviceRef::parse("CUDA0").unwrap(),
            DeviceRef::new(BackendHint::Hip, 0)
        );
        assert_eq!(
            DeviceRef::parse("HIP3").unwrap(),
            DeviceRef::new(BackendHint::Hip, 3)
        );
        assert_eq!(
            DeviceRef::parse("Vulkan1").unwrap(),
            DeviceRef::new(BackendHint::Vulkan, 1)
        );
        assert_eq!(
            DeviceRef::parse("GPU.1").unwrap(),
            DeviceRef::new(BackendHint::Vulkan, 1)
        );
        assert_eq!(
            DeviceRef::parse("GPU0").unwrap(),
            DeviceRef::new(BackendHint::Vulkan, 0)
        );
        assert_eq!(DeviceRef::parse("CPU").unwrap(), DeviceRef::cpu());
        assert_eq!(DeviceRef::parse(" cpu ").unwrap(), DeviceRef::cpu());
    }

    #[test]
    fn rejects_malformed_aliases() {
        for (spec, needle) in [
            ("", "empty device reference"),
            ("ROCm", "missing an index"),
            ("ROCm-1", "negative index"),
            ("ROCm1.5", "non-numeric index"),
            ("ROCm 1", "non-numeric index"),
            ("ROCmabc", "non-numeric index"),
            ("ROCm99999999999", "does not fit"),
            ("1", "no recognised backend prefix"),
            ("Metal0", "no recognised backend prefix"),
        ] {
            let err = DeviceRef::parse(spec).expect_err(spec);
            assert!(
                err.contains(needle),
                "spec {spec:?} error {err:?} should mention {needle:?}"
            );
        }
    }

    #[test]
    fn display_is_canonical_and_round_trips() {
        for spec in ["ROCm0", "ROCm7", "Vulkan2", "GPU.3", "CPU", "CUDA1", "HIP4"] {
            let parsed = DeviceRef::parse(spec).unwrap();
            let rendered = parsed.to_string();
            assert_eq!(DeviceRef::parse(&rendered).unwrap(), parsed, "{spec}");
        }
        assert_eq!(DeviceRef::parse("CUDA1").unwrap().to_string(), "ROCm1");
        assert_eq!(DeviceRef::parse("GPU.3").unwrap().to_string(), "Vulkan3");
        assert_eq!(DeviceRef::cpu().to_string(), "CPU");
    }

    #[test]
    fn backend_from_str() {
        assert_eq!("vulkan".parse::<Backend>().unwrap(), Backend::Vulkan);
        assert_eq!("VULKAN".parse::<Backend>().unwrap(), Backend::Vulkan);
        assert_eq!("rocm".parse::<Backend>().unwrap(), Backend::Hip);
        assert_eq!("hip".parse::<Backend>().unwrap(), Backend::Hip);
        assert_eq!("auto".parse::<Backend>().unwrap(), Backend::Auto);
        assert!("metal".parse::<Backend>().is_err());
        assert_eq!(Backend::default(), Backend::Vulkan);
    }
}
