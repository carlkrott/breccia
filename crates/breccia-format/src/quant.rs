//! GGML storage types and block geometry.
//!
//! The block table is the one verified in
//! `docs/specs/model-artifacts-inventory.md` §2.5 (byte counts cross-checked
//! against actual offset deltas in the real shards) and in the reference
//! engines (`strata/include/strata/artifact/gguf_reader.hpp:120`,
//! `llama.cpp-qwen4exp-mmap/ggml/include/ggml.h:390-433`).
//!
//! **Fork divergence:** type `42` is *not* upstream ggml's `Q2_0`. In this fork
//! it is 64 values in 18 bytes (16 B of 2-bit codes + 2 B fp16 scale) =
//! 2.25 bpw. Do not substitute upstream values.

use std::fmt;

/// Byte/element geometry of one quantization block.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BlockGeometry {
    /// Values covered by one block.
    pub elements: u64,
    /// Bytes one block occupies.
    pub bytes: u64,
}

impl BlockGeometry {
    pub const fn new(elements: u64, bytes: u64) -> Self {
        Self { elements, bytes }
    }

    /// Bytes needed for `elements` values, or `None` if `elements` is not a
    /// whole number of blocks.
    pub const fn bytes_for(&self, elements: u64) -> Option<u64> {
        if self.elements == 0 {
            return None;
        }
        if !elements.is_multiple_of(self.elements) {
            return None;
        }
        Some(elements / self.elements * self.bytes)
    }

    /// Effective bits per value (informational only).
    pub fn bits_per_value(&self) -> f64 {
        (self.bytes as f64 * 8.0) / self.elements as f64
    }
}

/// A GGML storage type id as written in a GGUF tensor record.
///
/// Kept as a newtype over the raw `u32` so unknown ids survive parsing and can
/// be reported with the offending tensor name instead of silently remapping.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GgmlType(u32);

impl GgmlType {
    pub const F32: Self = Self(0);
    pub const F16: Self = Self(1);
    pub const Q4_0: Self = Self(2);
    pub const Q4_1: Self = Self(3);
    pub const Q5_0: Self = Self(6);
    pub const Q5_1: Self = Self(7);
    pub const Q8_0: Self = Self(8);
    pub const Q8_1: Self = Self(9);
    pub const Q2_K: Self = Self(10);
    pub const Q3_K: Self = Self(11);
    pub const Q4_K: Self = Self(12);
    pub const Q5_K: Self = Self(13);
    pub const Q6_K: Self = Self(14);
    pub const Q8_K: Self = Self(15);
    pub const IQ2_XXS: Self = Self(16);
    pub const IQ2_XS: Self = Self(17);
    pub const IQ3_XXS: Self = Self(18);
    pub const IQ1_S: Self = Self(19);
    pub const IQ4_NL: Self = Self(20);
    pub const IQ3_S: Self = Self(21);
    pub const IQ2_S: Self = Self(22);
    pub const IQ4_XS: Self = Self(23);
    pub const I8: Self = Self(24);
    pub const I16: Self = Self(25);
    pub const I32: Self = Self(26);
    pub const I64: Self = Self(27);
    pub const F64: Self = Self(28);
    pub const IQ1_M: Self = Self(29);
    pub const BF16: Self = Self(30);
    pub const TQ1_0: Self = Self(34);
    pub const TQ2_0: Self = Self(35);
    /// Fork-only 1-bit type (see `llama.cpp-qwen4exp-mmap/ggml/include/ggml.h`).
    pub const Q1_0: Self = Self(41);
    /// **Fork-only** 2-bit type: 64 values / 18 bytes (2.25 bpw).
    pub const Q2_0: Self = Self(42);

    /// Wrap a raw type id.
    pub const fn from_id(id: u32) -> Self {
        Self(id)
    }

    /// Raw type id as stored in the file.
    pub const fn id(self) -> u32 {
        self.0
    }

    /// Canonical ggml name, or `None` for ids this engine does not know.
    pub fn name(self) -> Option<&'static str> {
        Some(match self.0 {
            0 => "F32",
            1 => "F16",
            2 => "Q4_0",
            3 => "Q4_1",
            6 => "Q5_0",
            7 => "Q5_1",
            8 => "Q8_0",
            9 => "Q8_1",
            10 => "Q2_K",
            11 => "Q3_K",
            12 => "Q4_K",
            13 => "Q5_K",
            14 => "Q6_K",
            15 => "Q8_K",
            16 => "IQ2_XXS",
            17 => "IQ2_XS",
            18 => "IQ3_XXS",
            19 => "IQ1_S",
            20 => "IQ4_NL",
            21 => "IQ3_S",
            22 => "IQ2_S",
            23 => "IQ4_XS",
            24 => "I8",
            25 => "I16",
            26 => "I32",
            27 => "I64",
            28 => "F64",
            29 => "IQ1_M",
            30 => "BF16",
            34 => "TQ1_0",
            35 => "TQ2_0",
            41 => "Q1_0",
            42 => "Q2_0",
            _ => return None,
        })
    }

    /// Block geometry, or `None` for ids this engine cannot size.
    pub fn block_geometry(self) -> Option<BlockGeometry> {
        Some(match self.0 {
            0 => BlockGeometry::new(1, 4),      // F32
            1 => BlockGeometry::new(1, 2),      // F16
            2 => BlockGeometry::new(32, 18),    // Q4_0
            3 => BlockGeometry::new(32, 20),    // Q4_1
            6 => BlockGeometry::new(32, 22),    // Q5_0
            7 => BlockGeometry::new(32, 24),    // Q5_1
            8 => BlockGeometry::new(32, 34),    // Q8_0
            9 => BlockGeometry::new(32, 36),    // Q8_1
            10 => BlockGeometry::new(256, 84),  // Q2_K
            11 => BlockGeometry::new(256, 110), // Q3_K
            12 => BlockGeometry::new(256, 144), // Q4_K
            13 => BlockGeometry::new(256, 176), // Q5_K
            14 => BlockGeometry::new(256, 210), // Q6_K
            15 => BlockGeometry::new(256, 292), // Q8_K
            16 => BlockGeometry::new(256, 66),  // IQ2_XXS
            17 => BlockGeometry::new(256, 74),  // IQ2_XS
            18 => BlockGeometry::new(256, 98),  // IQ3_XXS
            19 => BlockGeometry::new(256, 50),  // IQ1_S
            20 => BlockGeometry::new(32, 18),   // IQ4_NL
            21 => BlockGeometry::new(256, 110), // IQ3_S
            22 => BlockGeometry::new(256, 82),  // IQ2_S
            23 => BlockGeometry::new(256, 136), // IQ4_XS
            24 => BlockGeometry::new(1, 1),     // I8
            25 => BlockGeometry::new(1, 2),     // I16
            26 => BlockGeometry::new(1, 4),     // I32
            27 => BlockGeometry::new(1, 8),     // I64
            28 => BlockGeometry::new(1, 8),     // F64
            29 => BlockGeometry::new(256, 56),  // IQ1_M
            30 => BlockGeometry::new(1, 2),     // BF16
            34 => BlockGeometry::new(256, 54),  // TQ1_0
            35 => BlockGeometry::new(256, 66),  // TQ2_0
            // 41 = Q1_0 exists in this fork but its block size was not verified
            // against an artifact; report it as unknown rather than invent one.
            42 => BlockGeometry::new(64, 18), // Q2_0 (fork-verified, NOT upstream)
            _ => return None,
        })
    }

    /// Byte size for `elements` values, or `None` if unknown or not aligned.
    pub fn checked_bytes(self, elements: u64) -> Option<u64> {
        self.block_geometry()?.bytes_for(elements)
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.name() {
            Some(name) => f.write_str(name),
            None => write!(f, "type({})", self.0),
        }
    }
}

impl fmt::Debug for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fork_q2_0_geometry_is_verified_value() {
        // docs/specs/model-artifacts-inventory.md §2.5: 64 values / 18 bytes.
        let g = GgmlType::Q2_0.block_geometry().unwrap();
        assert_eq!(g, BlockGeometry::new(64, 18));
        assert_eq!(g.bytes_for(1_638_400), Some(460_800));
        // One `blk.N.ffn_down_exps.weight` at Q2_0: 640 × 2560 × 512 elements.
        assert_eq!(
            GgmlType::Q2_0.checked_bytes(640 * 2560 * 512),
            Some(460_800 * 512)
        );
    }

    #[test]
    fn documented_block_table() {
        let table = [
            (0, 1, 4),
            (1, 1, 2),
            (12, 256, 144),
            (13, 256, 176),
            (14, 256, 210),
            (16, 256, 66),
            (17, 256, 74),
            (18, 256, 98),
            (20, 32, 18),
            (21, 256, 110),
            (22, 256, 82),
            (23, 256, 136),
            (30, 1, 2),
            (42, 64, 18),
        ];
        for (id, elements, bytes) in table {
            let ty = GgmlType::from_id(id);
            assert_eq!(
                ty.block_geometry(),
                Some(BlockGeometry::new(elements, bytes)),
                "type {id} ({ty})"
            );
        }
    }

    #[test]
    fn unknown_type_has_no_geometry() {
        assert_eq!(GgmlType::from_id(999).block_geometry(), None);
        assert_eq!(GgmlType::from_id(999).name(), None);
        assert_eq!(GgmlType::from_id(999).to_string(), "type(999)");
    }

    #[test]
    fn unaligned_element_counts_are_rejected() {
        assert_eq!(GgmlType::IQ4_NL.checked_bytes(100), None); // 32-element blocks
        assert_eq!(GgmlType::Q6_K.checked_bytes(300), None); // 256-element blocks
    }
}
