//! Quant types the kernels implement, expressed in *kernel* geometry.
//!
//! The GGML block table lives in `breccia_format::quant` (verified against the
//! real shards) and is the source of truth; this module only adds the view the
//! kernels need:
//!
//! | kernel type | kernel block | bytes | notes |
//! |---|---|---|---|
//! | IQ2_XXS / IQ2_XS / IQ3_XXS / IQ3_S / Q5_K | 256 elems | 66 / 74 / 98 / 110 / 176 | one warp per super-block |
//! | IQ4_NL | 32 | 18 | 4 units per block, 8 blocks per warp iteration |
//! | Q2_0 (fork) | 64 | 18 | 16 B 2-bit codes + fp16 scale |
//! | BF16 / F32 | 8 | 16 / 32 | *virtual* blocks: GGML stores these per element, the kernels treat 8 elements as one block so every type shares the same unit-of-8 decode contract |
//!
//! Every supported type therefore decodes as units of 8 consecutive elements,
//! and `cols` must be handled with the tail path when it is not a multiple of
//! the kernel block.

use breccia_format::GgmlType;

/// A quant type with a fused dequant+GEMV kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum QuantType {
    Iq2Xxs,
    Iq2Xs,
    Iq3Xxs,
    Iq3S,
    Iq4Nl,
    Q5K,
    Q2_0,
    Bf16,
    F32,
}

impl QuantType {
    /// Every type this crate implements a kernel for.
    pub const ALL: [QuantType; 9] = [
        QuantType::Iq3S,
        QuantType::Q2_0,
        QuantType::Iq2Xxs,
        QuantType::Iq2Xs,
        QuantType::Iq3Xxs,
        QuantType::Iq4Nl,
        QuantType::Q5K,
        QuantType::Bf16,
        QuantType::F32,
    ];

    /// Canonical GGML name (`IQ3_S`, `Q2_0`, …).
    pub const fn name(self) -> &'static str {
        match self {
            QuantType::Iq2Xxs => "IQ2_XXS",
            QuantType::Iq2Xs => "IQ2_XS",
            QuantType::Iq3Xxs => "IQ3_XXS",
            QuantType::Iq3S => "IQ3_S",
            QuantType::Iq4Nl => "IQ4_NL",
            QuantType::Q5K => "Q5_K",
            QuantType::Q2_0 => "Q2_0",
            QuantType::Bf16 => "BF16",
            QuantType::F32 => "F32",
        }
    }

    /// GGML type id (the fork's table: `Q2_0 = 42`, not upstream's).
    pub const fn ggml_id(self) -> u32 {
        match self {
            QuantType::F32 => 0,
            QuantType::Q5K => 13,
            QuantType::Iq2Xxs => 16,
            QuantType::Iq2Xs => 17,
            QuantType::Iq3Xxs => 18,
            QuantType::Iq4Nl => 20,
            QuantType::Iq3S => 21,
            QuantType::Bf16 => 30,
            QuantType::Q2_0 => 42,
        }
    }

    /// Shader stem (`shaders/gemv_<stem>.comp`, `shaders/dequant_<stem>.comp`).
    pub const fn shader_stem(self) -> &'static str {
        match self {
            QuantType::Iq2Xxs => "iq2_xxs",
            QuantType::Iq2Xs => "iq2_xs",
            QuantType::Iq3Xxs => "iq3_xxs",
            QuantType::Iq3S => "iq3_s",
            QuantType::Iq4Nl => "iq4_nl",
            QuantType::Q5K => "q5_k",
            QuantType::Q2_0 => "q2_0",
            QuantType::Bf16 => "bf16",
            QuantType::F32 => "f32",
        }
    }

    /// Elements per kernel block (8 for BF16/F32 — a virtual block).
    pub const fn block_elems(self) -> usize {
        match self {
            QuantType::Iq4Nl => 32,
            QuantType::Q2_0 => 64,
            QuantType::Bf16 | QuantType::F32 => 8,
            _ => 256,
        }
    }

    /// Bytes per kernel block.
    pub const fn block_bytes(self) -> usize {
        match self {
            QuantType::Iq2Xxs => 66,
            QuantType::Iq2Xs => 74,
            QuantType::Iq3Xxs => 98,
            QuantType::Iq3S => 110,
            QuantType::Iq4Nl => 18,
            QuantType::Q5K => 176,
            QuantType::Q2_0 => 18,
            QuantType::Bf16 => 16,
            QuantType::F32 => 32,
        }
    }

    /// Effective bits per weight *as stored by GGML* (BF16/F32 use their real
    /// per-element geometry, not the 8-element kernel view).
    pub fn bits_per_value(self) -> f64 {
        match self {
            QuantType::Bf16 => 16.0,
            QuantType::F32 => 32.0,
            _ => (self.block_bytes() as f64 * 8.0) / self.block_elems() as f64,
        }
    }

    /// Map a GGUF storage type onto a kernel, or `None` if unimplemented.
    pub fn from_ggml(ty: GgmlType) -> Option<QuantType> {
        QuantType::ALL.into_iter().find(|q| q.ggml_id() == ty.id())
    }

    /// Parse the canonical name (case-insensitive, `IQ3_XXS`/`iq3-xxs`/`q2_0`).
    pub fn from_name(name: &str) -> Option<QuantType> {
        let norm: String = name
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .map(|c| c.to_ascii_uppercase())
            .collect();
        QuantType::ALL
            .into_iter()
            .find(|q| q.name().replace('_', "") == norm)
    }
}

impl std::fmt::Display for QuantType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for QuantType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        QuantType::from_name(s).ok_or_else(|| {
            format!(
                "unknown quant type {s:?}; implemented: {}",
                QuantType::ALL
                    .iter()
                    .map(|q| q.name())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
    }
}

/// Types present in the target model that this work order deliberately leaves
/// for the next kernel pass (counts from `docs/specs/model-artifacts-inventory.md`
/// §4.2: tensors per type in shard 1).
pub const NOT_IMPLEMENTED: &[(&str, &str, u32)] = &[
    ("IQ4_XS", "256/136", 67),
    ("Q6_K", "256/210", 71),
    ("Q4_K", "256/144", 45),
    ("IQ2_S", "256/82", 20),
    ("Q2_K", "256/84", 0),
    ("Q3_K", "256/110", 0),
    ("Q8_0", "32/34", 0),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_ids_round_trip() {
        for q in QuantType::ALL {
            assert_eq!(QuantType::from_name(q.name()), Some(q), "{q}");
            assert_eq!(QuantType::from_name(&q.name().to_lowercase()), Some(q));
            assert_eq!(
                QuantType::from_ggml(GgmlType::from_id(q.ggml_id())),
                Some(q)
            );
            assert_eq!(q.to_string(), q.name());
        }
    }

    #[test]
    fn geometry_matches_the_format_crate_for_real_block_types() {
        // BF16/F32 are the documented exception (virtual 8-element blocks).
        for q in QuantType::ALL {
            if matches!(q, QuantType::Bf16 | QuantType::F32) {
                continue;
            }
            let g = GgmlType::from_id(q.ggml_id())
                .block_geometry()
                .expect("geometry");
            assert_eq!(g.elements as usize, q.block_elems(), "{q} elements");
            assert_eq!(g.bytes as usize, q.block_bytes(), "{q} bytes");
        }
    }

    #[test]
    fn q2_0_is_the_fork_block_not_upstreams() {
        // 64 values in 18 bytes = 2.25 bpw (docs/specs §2.5).
        assert_eq!(QuantType::Q2_0.block_elems(), 64);
        assert_eq!(QuantType::Q2_0.block_bytes(), 18);
        assert!((QuantType::Q2_0.bits_per_value() - 2.25).abs() < 1e-12);
        assert_eq!(QuantType::Iq4Nl.bits_per_value(), 4.5);
        assert!((QuantType::Iq2Xxs.bits_per_value() - 2.0625).abs() < 1e-12);
        assert!((QuantType::Iq3S.bits_per_value() - 3.4375).abs() < 1e-12);
        assert_eq!(QuantType::Q5K.bits_per_value(), 5.5);
    }
}
