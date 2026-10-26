//! Small typed values for the llama.cpp flag surface.
//!
//! These types exist so that a bad value fails at *parse* time with a flag name
//! and position, instead of silently reaching the engine.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

/// A llama.cpp on/off flag value (`-fa on`, `--fit off`, `-lzm on`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Switch {
    /// Enabled.
    On,
    /// Disabled.
    Off,
}

impl Switch {
    /// Parse `on`/`off`/`true`/`false`/`yes`/`no`/`1`/`0` (case-insensitive).
    pub fn parse_value(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "on" | "true" | "yes" | "1" | "enable" | "enabled" => Ok(Switch::On),
            "off" | "false" | "no" | "0" | "disable" | "disabled" => Ok(Switch::Off),
            _ => Err(format!(
                "`{}` is not an on/off value; expected on, off, true, false, yes, no, 1 or 0",
                value.trim()
            )),
        }
    }

    /// `true` for [`Switch::On`].
    pub const fn is_on(self) -> bool {
        matches!(self, Switch::On)
    }
}

impl fmt::Display for Switch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Switch::On => "on",
            Switch::Off => "off",
        })
    }
}

impl FromStr for Switch {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse_value(s)
    }
}

/// `--reasoning` mode. The fork accepts `on`/`off`; `auto` is kept because
/// upstream chat templates use it as the tri-state default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningMode {
    /// Let the chat template decide.
    Auto,
    /// Force reasoning traces on.
    On,
    /// Suppress reasoning traces.
    Off,
}

impl ReasoningMode {
    /// Parse `off`/`on`/`auto` (case-insensitive).
    ///
    /// `on`/`off`/`true`/`false` are accepted through [`Switch`] spellings.
    pub fn parse_value(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(ReasoningMode::Auto),
            other => Switch::parse_value(other).map(|s| match s {
                Switch::On => ReasoningMode::On,
                Switch::Off => ReasoningMode::Off,
            }),
        }
    }
}

impl fmt::Display for ReasoningMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ReasoningMode::Auto => "auto",
            ReasoningMode::On => "on",
            ReasoningMode::Off => "off",
        })
    }
}

impl FromStr for ReasoningMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse_value(s)
    }
}

/// KV-cache element type (`-ctk`/`-ctv`).
///
/// The set mirrors the types llama.cpp's `ggml_type` KV path accepts; anything
/// else is rejected at parse time rather than reaching the allocator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvCacheType {
    /// 32-bit float.
    #[serde(rename = "f32")]
    F32,
    /// 16-bit float (llama.cpp default).
    #[serde(rename = "f16")]
    F16,
    /// bfloat16.
    #[serde(rename = "bf16")]
    Bf16,
    /// Q8_0 blocks.
    #[serde(rename = "q8_0")]
    Q8_0,
    /// Q5_1 blocks.
    #[serde(rename = "q5_1")]
    Q5_1,
    /// Q5_0 blocks.
    #[serde(rename = "q5_0")]
    Q5_0,
    /// Q4_1 blocks.
    #[serde(rename = "q4_1")]
    Q4_1,
    /// Q4_0 blocks — the production choice for both K and V.
    #[serde(rename = "q4_0")]
    Q4_0,
    /// IQ4_NL blocks.
    #[serde(rename = "iq4_nl")]
    Iq4Nl,
}

impl KvCacheType {
    /// Flag spelling of this type, e.g. `q4_0`.
    pub const fn as_flag_str(self) -> &'static str {
        match self {
            KvCacheType::F32 => "f32",
            KvCacheType::F16 => "f16",
            KvCacheType::Bf16 => "bf16",
            KvCacheType::Q8_0 => "q8_0",
            KvCacheType::Q5_1 => "q5_1",
            KvCacheType::Q5_0 => "q5_0",
            KvCacheType::Q4_1 => "q4_1",
            KvCacheType::Q4_0 => "q4_0",
            KvCacheType::Iq4Nl => "iq4_nl",
        }
    }

    /// Parse a KV-cache type (case-insensitive).
    pub fn parse_value(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "f32" => Ok(KvCacheType::F32),
            "f16" => Ok(KvCacheType::F16),
            "bf16" => Ok(KvCacheType::Bf16),
            "q8_0" => Ok(KvCacheType::Q8_0),
            "q5_1" => Ok(KvCacheType::Q5_1),
            "q5_0" => Ok(KvCacheType::Q5_0),
            "q4_1" => Ok(KvCacheType::Q4_1),
            "q4_0" => Ok(KvCacheType::Q4_0),
            "iq4_nl" => Ok(KvCacheType::Iq4Nl),
            _ => Err(format!(
                "`{}` is not a KV-cache type; expected one of f32, f16, bf16, q8_0, q5_1, q5_0, q4_1, q4_0, iq4_nl",
                value.trim()
            )),
        }
    }
}

impl fmt::Display for KvCacheType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_flag_str())
    }
}

impl FromStr for KvCacheType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse_value(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switch_spellings() {
        for s in ["on", "ON", "true", "1", "yes", "enabled"] {
            assert_eq!(Switch::parse_value(s).unwrap(), Switch::On, "{s}");
        }
        for s in ["off", "OFF", "false", "0", "no", "disabled"] {
            assert_eq!(Switch::parse_value(s).unwrap(), Switch::Off, "{s}");
        }
        assert!(Switch::parse_value("maybe").is_err());
        assert!(Switch::parse_value("").is_err());
        assert_eq!(Switch::On.to_string(), "on");
        assert_eq!(Switch::Off.to_string(), "off");
    }

    #[test]
    fn reasoning_spellings() {
        assert_eq!(
            ReasoningMode::parse_value("off").unwrap(),
            ReasoningMode::Off
        );
        assert_eq!(
            ReasoningMode::parse_value("AUTO").unwrap(),
            ReasoningMode::Auto
        );
        assert_eq!(ReasoningMode::parse_value("on").unwrap(), ReasoningMode::On);
        assert!(ReasoningMode::parse_value("sometimes").is_err());
    }

    #[test]
    fn kv_types_round_trip() {
        for t in [
            KvCacheType::F32,
            KvCacheType::F16,
            KvCacheType::Bf16,
            KvCacheType::Q8_0,
            KvCacheType::Q5_1,
            KvCacheType::Q5_0,
            KvCacheType::Q4_1,
            KvCacheType::Q4_0,
            KvCacheType::Iq4Nl,
        ] {
            assert_eq!(KvCacheType::parse_value(t.as_flag_str()).unwrap(), t);
        }
        assert_eq!(KvCacheType::parse_value("Q4_0").unwrap(), KvCacheType::Q4_0);
        assert!(KvCacheType::parse_value("q3_k").is_err());
    }
}
