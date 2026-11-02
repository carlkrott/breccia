//! Multi-user fused dequant + mat-vector: one expert, `m` users, one decode.
//!
//! The single-user kernel ([`crate::Gemv`]) decodes a quantised weight block once
//! per output row and dots it against one activation vector, so a prefill chunk of
//! `m` tokens that share an expert pays the decode and the DRAM read `m` times.
//! [`GemmMany`] decodes each block **once into registers** and dots it with all `m`
//! users before moving on (`shaders/gemm_body.glsl`,
//! `docs/phase-4/01-forward-pass.md` §8v).
//!
//! Delivered for the `down` role's `IQ4_NL` first; every other role/type keeps the
//! fused single-user path (the blend is reported in the phase-4 document).

use anyhow::{anyhow, Result};
use ash::vk;
use breccia_vk::ctx::{Buf, Gpu, Qc};

use crate::gemv::{table_bytes, upload, GemvTiming, ROWS_PER_WG, WG_X_MAX};
use crate::quant::QuantType;

/// Users one dispatch can hold in registers (the shader's `USERS_MAX`).
pub const USERS_MAX: usize = 8;

/// A compiled multi-user fused dequant + mat-vector pipeline.
pub struct GemmMany<'g> {
    ty: QuantType,
    gpu: &'g Gpu,
    kernel: breccia_vk::ctx::Kernel,
    /// The int8-dot variant (q8_1 activations, §11j), for the types that have one.
    q8: Option<breccia_vk::ctx::Kernel>,
    /// The fused gate + up + SwiGLU variant (§11l item 4), for the types that have one.
    pair: Option<breccia_vk::ctx::Kernel>,
    /// The register-blocked builds (§12 item 1): `((rows per lane group, users
    /// per decode), LDS-staged X, kernel)`, LDS builds first.
    blocked: Vec<(Variant, bool, breccia_vk::ctx::Kernel)>,
    /// Wanted variant of the plain-X and the LDS-X builds (`(1, USERS_MAX)` =
    /// the plain build).
    want: [Variant; 2],
    /// The tiled builds (`shaders/gemm_tile.glsl`): `((RM, RN), kernel)`, a
    /// `16*RM x 16*RN` output tile per workgroup.
    tiles: Vec<(Tile, breccia_vk::ctx::Kernel)>,
    /// Tiled build [`GemmMany::run_at`] / [`GemmMany::run_timed`] use for wide-enough
    /// shapes; `None` = the GEMV-shaped builds.
    want_tile: Option<Tile>,
    /// The 4x4 tiled BF16 build with the HC gated-mean + inject epilogue
    /// (`gemm_tile.glsl`, `BRECCIA_HC_TILE_GATED`); built on demand by
    /// [`GemmMany::enable_hc_tile_gated`].
    hcgate: Option<breccia_vk::ctx::Kernel>,
    /// Forced users per z slice (1..=USERS_MAX); `None` = [`upz_for`] per dispatch.
    upz: Option<usize>,
    /// The mixed-float-dot body, when the device advertises the extension. Only
    /// dispatched when the caller declared its weights to be f16 pairs
    /// ([`GemmMany::set_dot_weights`]) *and* the calibration chose `dot` here.
    dot: Option<DotGemm<'g>>,
    /// The caller's declaration that this matrix's weights are f16 pairs (two per
    /// word). Default false: quantised bits must never be read as f16.
    dot_weights: bool,
    /// The caller's declaration that this matrix's weights are **BF16** and may be
    /// converted to f16 pairs once ([`GemmMany::set_dot_bf16`]). Kept separate from
    /// `dot_weights` so an already-f16 buffer is never re-interpreted as BF16.
    dot_bf16: bool,
    /// BF16 weight tensors converted to f16 pairs, keyed by source `Buf::id`. The
    /// conversion is the prepass the dot needs: BF16 and f16 are both 2 bytes, so
    /// the copy is the same size as the tensor it mirrors, made once on first use.
    dot_cache: std::sync::Mutex<std::collections::HashMap<u64, Buf>>,
    /// `set_tile(Some(_))` was called: use exactly that tile, bypassing both the
    /// env override and the measured rule (calibration and A/B rely on this).
    tile_locked: bool,
    tables: Buf,
}

/// `BRECCIA_GEMM_UPZ` (1..=8) forces the users per z slice (A/B only).
fn default_upz() -> Option<usize> {
    static UPZ: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *UPZ.get_or_init(|| {
        std::env::var("BRECCIA_GEMM_UPZ")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .map(|n: usize| n.clamp(1, USERS_MAX))
    })
}

/// `(rows per lane group, users per decode)` of a plain-GEMM build.
pub type Variant = (usize, usize);

/// `(RM, RN)` register tile of a tiled build: a `16*RM x 16*RN` (rows x users)
/// output tile per workgroup. Keep in step with `GEMM_TILES` in `build.rs`.
pub type Tile = (usize, usize);

/// Rows x users a tiled build covers per workgroup.
pub fn tile_dims(t: Tile) -> (usize, usize) {
    (t.0 * 16, t.1 * 16)
}

/// `BRECCIA_GEMM_TILE`: `0` = never tiled, `1` = the per-device default tile,
/// `RMxRN` (e.g. `8x4`) = that tile (A/B only). Unset = [`tile_default`].
fn env_tile(subgroup: u32, ty: QuantType) -> Option<Tile> {
    match std::env::var("BRECCIA_GEMM_TILE")
        .ok()
        .as_deref()
        .map(str::trim)
    {
        Some("0") => None,
        Some("1") => Some(TILE_FALLBACK),
        Some(v) if v.contains('x') => v
            .split_once('x')
            .and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))
            .filter(|t| TILES.contains(t)),
        _ => tile_default(ty, subgroup),
    }
}

const TILES: [Tile; 4] = [(4, 2), (4, 4), (8, 4), (8, 8)];
const TILE_FALLBACK: Tile = (8, 4);

/// Measured-faster tiled build (m = 1024, attn_gate / hc_ffn_up, `tests/gemm_many.rs`
/// `tiled_gemm_matches_plain`): 4x4 beat or matched 8x4 / 8x8 on RDNA2 (wave32) and
/// Vega20 (wave64) for every type measured, at 0.4x-0.5x the previous best's time.
fn tile_default(_ty: QuantType, _subgroup: u32) -> Option<Tile> {
    Some((4, 4))
}

/// `BRECCIA_GEMM_TILE` set: the caller wants exactly that tile, no measured rule.
fn tile_forced() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| std::env::var("BRECCIA_GEMM_TILE").is_ok())
}

/// One calibrated cell: at `(rows, users)` on this device, `tile` measured
/// `us` microseconds. Written by [`calibrate_tiles`] and installed from
/// `hw_profile.json` at startup, so the choice follows the hardware instead of a
/// constant compiled for the developer's card.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct TileCell {
    /// Quant type name (`IQ3_S`, `BF16`, …): the optimum differs by type, because
    /// the inner loop's latency does (BF16 wants the small tile at every user count
    /// on the narrow shape; IQ3_S does not).
    pub ty: String,
    pub rows: usize,
    pub users: usize,
    /// `"4x2"`, `"4x4"`, `"8x4"`, `"8x8"`.
    pub tile: String,
    pub us: f64,
}

/// Cells installed from the profile (per process). Empty = fall back to
/// [`tile_for`]'s built-in rule, which was measured on this host.
fn tile_table() -> &'static std::sync::RwLock<Vec<TileCell>> {
    static T: std::sync::OnceLock<std::sync::RwLock<Vec<TileCell>>> = std::sync::OnceLock::new();
    T.get_or_init(|| std::sync::RwLock::new(Vec::new()))
}

/// Install the calibrated cells (startup, from `hw_profile.json`). Replaces any
/// previous table; an empty vector restores the built-in rule.
pub fn set_tile_table(cells: Vec<TileCell>) {
    if let Ok(mut t) = tile_table().write() {
        *t = cells;
    }
}

fn parse_tile(s: &str) -> Option<Tile> {
    let (a, b) = s.split_once('x')?;
    let t: Tile = (a.trim().parse().ok()?, b.trim().parse().ok()?);
    TILES.contains(&t).then_some(t)
}

/// Sweep the tiled builds over `cells` for `types` and return the winner per cell.
///
/// This is the function an install/startup step calls: it needs no model (the cost
/// depends on shape and type, not weight values), touches only the device, and
/// returns rows that go straight into `hw_profile.json` and back through
/// [`set_tile_table`]. Times are summed across the types swept so one tile serves
/// every type at that shape; ties go to the larger tile (fewer workgroups, less
/// LDS churn).
pub fn calibrate_tiles(
    gpu: &breccia_vk::ctx::Gpu,
    cells: &[(usize, usize, usize)],
    types: &[QuantType],
) -> Result<Vec<TileCell>> {
    let mut out = Vec::new();
    for &(rows, cols, users) in cells {
        for &ty in types {
            let Ok(mut gemm) = GemmMany::new(gpu, ty) else {
                continue;
            };
            let wbytes = gemm.weight_bytes(rows, cols);
            let w = crate::gemv::alloc(gpu, wbytes)?;
            let x = crate::gemv::alloc(gpu, (users * cols * 4) as u64)?;
            let o = crate::gemv::alloc(gpu, (users * rows * 4) as u64)?;
            crate::gemv::upload_into(gpu, &w, &vec![0u8; wbytes as usize])?;
            crate::gemv::upload_into(gpu, &x, &vec![0u8; users * cols * 4])?;
            let mut totals = vec![f64::INFINITY; TILES.len()];
            for (i, t) in TILES.iter().enumerate() {
                gemm.set_tile(Some(*t));
                for pass in 0..4 {
                    let t0 = std::time::Instant::now();
                    if gemm.run_at(&w, &x, &o, rows, cols, users, 0).is_err() {
                        break;
                    }
                    if pass > 0 {
                        let dt = t0.elapsed().as_secs_f64() * 1e6;
                        if dt < totals[i] {
                            totals[i] = dt;
                        }
                    }
                }
            }
            // Mixed-dot candidate: same shapes, f16-pair weights. It is measured
            // whenever the device advertises the extension, so the table can say
            // "the dot wins here"; the *runtime* additionally requires the caller to
            // declare f16 weights (`set_dot_weights`), so a quantised tensor can
            // never be routed to it by a stale verdict.
            let mut dot_us = f64::INFINITY;
            if let Ok(dot) = DotGemm::new_tiled(gpu) {
                for pass in 0..4 {
                    let t0 = std::time::Instant::now();
                    if dot.run_tiled(&w, &x, &o, rows, cols, users).is_err() {
                        break;
                    }
                    if pass > 0 {
                        let dt = t0.elapsed().as_secs_f64() * 1e6;
                        if dt < dot_us {
                            dot_us = dt;
                        }
                    }
                }
            }
            gemm.set_tile(None);
            gpu.destroy_buf(&w);
            gpu.destroy_buf(&x);
            gpu.destroy_buf(&o);
            // Ties (or holes) resolve to the later, larger tile.
            let mut best = 0usize;
            for i in 1..TILES.len() {
                if totals[i] <= totals[best] {
                    best = i;
                }
            }
            if !totals[best].is_finite() && !dot_us.is_finite() {
                continue;
            }
            let (tile, us) = if dot_us.is_finite() && dot_us < totals[best] * 0.99 {
                ("dot".to_string(), dot_us)
            } else {
                let t = TILES[best];
                (format!("{}x{}", t.0, t.1), totals[best])
            };
            out.push(TileCell {
                ty: ty.name().to_string(),
                rows,
                users,
                tile,
                us,
            });
        }
    }
    Ok(out)
}

/// What the calibrated table says to run at a shape: one of the GLSL tiles, or the
/// mixed-float-dot kernel (`DotGemm`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Choice {
    Tile(Tile),
    Dot,
}

/// The nearest calibrated cell in log space, per type, as a [`Choice`].
fn choice_from_table(ty: QuantType, rows: usize, users: usize) -> Option<Choice> {
    let t = tile_table().read().ok()?;
    let want = ty.name();
    let mut near: Option<&TileCell> = None;
    let d = |c: &TileCell| {
        ((rows.max(1) as f64).ln() - (c.rows.max(1) as f64).ln()).powi(2)
            + ((users.max(1) as f64).ln() - (c.users.max(1) as f64).ln()).powi(2)
    };
    for c in t.iter().filter(|c| c.ty == want) {
        near = match near {
            Some(b) if d(b) <= d(c) => Some(b),
            _ => Some(c),
        };
    }
    let c = near?;
    if c.tile == "dot" {
        return Some(Choice::Dot);
    }
    parse_tile(&c.tile).map(Choice::Tile)
}

/// The nearest calibrated cell in log-space, so a shape between two samples uses
/// the closer one. `None` when nothing has been calibrated.
fn tile_from_table(ty: QuantType, rows: usize, users: usize) -> Option<Tile> {
    let t = tile_table().read().ok()?;
    let want = ty.name();
    let mut near: Option<&TileCell> = None;
    let lr = (rows.max(1) as f64).ln();
    let lu = (users.max(1) as f64).ln();
    // A calibrated `dot` cell is not a tile: skip it so the GLSL path still has a
    // sensible answer when the caller cannot use the dot.
    for c in t.iter().filter(|c| c.ty == want && c.tile != "dot") {
        let d = |c: &TileCell| {
            (lr - (c.rows.max(1) as f64).ln()).powi(2) + (lu - (c.users.max(1) as f64).ln()).powi(2)
        };
        near = match near {
            Some(b) if d(b) <= d(c) => Some(b),
            _ => Some(c),
        };
    }
    near.and_then(|c| parse_tile(&c.tile))
}

/// The tile for a concrete dispatch, measured (`gemm_calib` `gemm_tile_sweep`,
/// RX, `out/gemm-tiles-rx.csv`): the 4x4 build is right for wide shapes and large
/// batches, and the **4x2** build (half the user tile, twice the workgroups) wins
/// by 1.4-8.2x when the row count is small enough that `rows/64` leaves the tiled
/// grid starved. Numbers behind the rule:
///
/// | rows | users | 4x2 vs 4x4 |
/// |---|---|---|
/// | 640 (gate) | 64-256 | 1.37-3.08x (IQ3_S), 4.2-8.2x (BF16) |
/// | 640 | 512-1024 | 4x4 better (0.4-0.8x for 4x2) |
/// | 2560 (down) | 64 | 1.16-1.35x |
/// | 2560 | >= 96 | 4x4 better |
fn tile_for(ty: QuantType, rows: usize, users: usize) -> Option<Tile> {
    // A calibrated table (installed from the profile at startup) wins: it is the
    // only source that travelled with the hardware it was measured on.
    if let Some(t) = tile_from_table(ty, rows, users) {
        return Some(t);
    }
    // Fallback rule, measured on this host (RX 6950 XT / MI50, 2026-11).
    // A narrow row count means only rows/64 workgroups per user tile, so the
    // smaller user tile is what fills the GPU.
    let narrow = rows <= 1024;
    if narrow && (users <= 256 || ty == QuantType::Bf16) {
        return Some((4, 2));
    }
    // Tiny batches starve every tile; the narrow one starves least.
    if users <= 96 {
        return Some((4, 2));
    }
    Some((4, 4))
}

/// Users below which the tiled build is not used (tiles would be mostly padding).
const TILE_MIN_USERS: usize = 64;

/// `BRECCIA_GEMM_UMAX` (8 or 16): users one decode serves; overrides [`tuned`].
fn env_umax() -> Option<usize> {
    static UMAX: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *UMAX.get_or_init(|| {
        std::env::var("BRECCIA_GEMM_UMAX")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|n| [8, 16].contains(n))
    })
}

/// `BRECCIA_GEMM_RPL` (1, 2 or 4): rows per lane group; overrides [`tuned`].
fn env_rpl() -> Option<usize> {
    static RPL: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *RPL.get_or_init(|| {
        std::env::var("BRECCIA_GEMM_RPL")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|n| [1, 2, 4].contains(n))
    })
}

/// Measured best build per GPU family and type (plan §12 item 1, m = 1024,
/// `tests/gemm_many.rs`): wave64 = Vega20 (MI50), wave32 = RDNA2 (RX). Types not
/// measured keep the plain build.
fn tuned(ty: QuantType, subgroup: u32, ldsx: bool) -> Variant {
    use QuantType::*;
    match (subgroup, ldsx, ty) {
        (64, true, Iq3S | Iq4Xs | Q4K | Q5K | Bf16) => (2, 8),
        (64, true, Q6K) => (1, 16),
        (64, false, Iq3S | Iq4Xs | Q4K | Bf16) => (2, 8),
        (64, false, Q5K | Q6K) => (1, 16),
        (32, _, Iq3S | Iq4Xs | Bf16) => (2, 8),
        _ => (1, USERS_MAX),
    }
}

/// Whether the blocked builds stage X in LDS (`gemm_rpl.glsl`): measured faster
/// on wave64 (MI50) only; `BRECCIA_GEMM_LDSX=0|1` overrides.
fn ldsx(gpu: &Gpu) -> bool {
    match std::env::var("BRECCIA_GEMM_LDSX").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => subgroup_size(gpu) == 64,
    }
}

/// Users per z slice a workgroup needs before the LDS-X build pays for its
/// barriers: decode (2 streams) measured ~1 % slower with it on the MI50 (plan §12.1b).
const LDSX_MIN_USERS: usize = 8;

fn subgroup_size(gpu: &Gpu) -> u32 {
    let mut sg = vk::PhysicalDeviceSubgroupProperties::default();
    let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sg);
    unsafe {
        gpu.instance
            .get_physical_device_properties2(gpu.pdev, &mut p2)
    };
    sg.subgroup_size
}

/// Workgroups a dispatch should reach before users share a z slice.
// shortcut: 320 = 4 WGs per CU on the RX's 80 CUs (MI50 has 60); per-device
// target if MI50 numbers disagree.
const WG_TARGET: usize = 320;

/// Users per z slice: `USERS_MAX` when rows x slices already fill the GPU, else
/// the largest split that reaches [`WG_TARGET`], floor 2. A 640-row expert
/// GEMM at 8 users gets 2 (1.9x, per-user bits unchanged); down (2560 rows)
/// and wide dense GEMMs keep 8.
fn upz_for(rows: usize, users: usize, umax: usize) -> usize {
    let groups = rows.div_ceil(ROWS_PER_WG).max(1);
    // Powers of two only: 16 users at 5 per slice would leave a 1-user slice.
    [umax, 8, 4, 2]
        .into_iter()
        .filter(|&z| z <= umax)
        .find(|&z| groups * users.max(1).div_ceil(z) >= WG_TARGET)
        .unwrap_or(2)
}

/// The users per z slice a push-constant block carries (`pc.upz`).
fn pushed_upz(pc: &[u8; 24]) -> usize {
    u32::from_le_bytes(pc[20..24].try_into().unwrap()).max(1) as usize
}

macro_rules! blocked_spv {
    ($($t:ident => $s:literal),* $(,)?) => {
        /// The register-blocked builds of `ty`'s plain GEMM (`build.rs`, `_r2`/`_r4`),
        /// the LDS-staged-X ones (`..x.spv`) when `ldsx`.
        fn blocked_spv(ty: QuantType, ldsx: bool) -> Vec<(Variant, &'static [u8])> {
            match (ty, ldsx) {
                $((QuantType::$t, false) => vec![
                    ((2, 8), include_bytes!(concat!(env!("OUT_DIR"), "/gemm_", $s, "_r2u8.spv"))),
                    ((4, 8), include_bytes!(concat!(env!("OUT_DIR"), "/gemm_", $s, "_r4u8.spv"))),
                    ((1, 16), include_bytes!(concat!(env!("OUT_DIR"), "/gemm_", $s, "_r1u16.spv"))),
                    ((2, 16), include_bytes!(concat!(env!("OUT_DIR"), "/gemm_", $s, "_r2u16.spv"))),
                ],
                (QuantType::$t, true) => vec![
                    ((2, 8), include_bytes!(concat!(env!("OUT_DIR"), "/gemm_", $s, "_r2u8x.spv"))),
                    ((4, 8), include_bytes!(concat!(env!("OUT_DIR"), "/gemm_", $s, "_r4u8x.spv"))),
                    ((1, 16), include_bytes!(concat!(env!("OUT_DIR"), "/gemm_", $s, "_r1u16x.spv"))),
                    ((2, 16), include_bytes!(concat!(env!("OUT_DIR"), "/gemm_", $s, "_r2u16x.spv"))),
                ],)*
                _ => Vec::new(),
            }
        }
    };
}
blocked_spv!(
    Iq4Nl => "iq4_nl", Q2_0 => "q2_0", Iq2Xxs => "iq2_xxs", Iq2Xs => "iq2_xs",
    Iq2S => "iq2_s", Iq3Xxs => "iq3_xxs", Iq3S => "iq3_s", Iq4Xs => "iq4_xs",
    Q4K => "q4_k", Q5K => "q5_k", Q6K => "q6_k", Bf16 => "bf16",
);

macro_rules! tile_spv {
    ($($t:ident => $s:literal),* $(,)?) => {
        /// The tiled builds of `ty`'s GEMM (`build.rs`, `_t{RM}x{RN}`).
        fn tile_spv(ty: QuantType) -> Vec<(Tile, &'static [u8])> {
            match ty {
                $(QuantType::$t => vec![
                    ((4, 4), include_bytes!(concat!(env!("OUT_DIR"), "/gemm_", $s, "_t4x4.spv"))),
                    ((8, 4), include_bytes!(concat!(env!("OUT_DIR"), "/gemm_", $s, "_t8x4.spv"))),
                    ((8, 8), include_bytes!(concat!(env!("OUT_DIR"), "/gemm_", $s, "_t8x8.spv"))),
                ],)*
                _ => Vec::new(),
            }
        }
    };
}
tile_spv!(
    Iq4Nl => "iq4_nl", Q2_0 => "q2_0", Iq2Xxs => "iq2_xxs", Iq2Xs => "iq2_xs",
    Iq2S => "iq2_s", Iq3Xxs => "iq3_xxs", Iq3S => "iq3_s", Iq4Xs => "iq4_xs",
    Q4K => "q4_k", Q5K => "q5_k", Q6K => "q6_k", Bf16 => "bf16",
);

/// Whether [`GemmMany::run_pair`] has a shader for `ty` (gate and up both `ty`).
pub fn has_pair_shader(ty: QuantType) -> bool {
    ty == QuantType::Iq3S
}

impl<'g> GemmMany<'g> {
    /// Compile the pipeline for `ty` (only the types with a multi-user shader).
    pub fn new(gpu: &'g Gpu, ty: QuantType) -> Result<Self> {
        let spv: &[u8] = match ty {
            QuantType::Iq4Nl => include_bytes!(concat!(env!("OUT_DIR"), "/gemm_iq4_nl.spv")),
            QuantType::Q2_0 => include_bytes!(concat!(env!("OUT_DIR"), "/gemm_q2_0.spv")),
            QuantType::Iq2Xxs => include_bytes!(concat!(env!("OUT_DIR"), "/gemm_iq2_xxs.spv")),
            QuantType::Iq2Xs => include_bytes!(concat!(env!("OUT_DIR"), "/gemm_iq2_xs.spv")),
            QuantType::Iq2S => include_bytes!(concat!(env!("OUT_DIR"), "/gemm_iq2_s.spv")),
            QuantType::Iq3Xxs => include_bytes!(concat!(env!("OUT_DIR"), "/gemm_iq3_xxs.spv")),
            QuantType::Iq3S => include_bytes!(concat!(env!("OUT_DIR"), "/gemm_iq3_s.spv")),
            QuantType::Iq4Xs => include_bytes!(concat!(env!("OUT_DIR"), "/gemm_iq4_xs.spv")),
            QuantType::Q4K => include_bytes!(concat!(env!("OUT_DIR"), "/gemm_q4_k.spv")),
            QuantType::Q5K => include_bytes!(concat!(env!("OUT_DIR"), "/gemm_q5_k.spv")),
            QuantType::Q6K => include_bytes!(concat!(env!("OUT_DIR"), "/gemm_q6_k.spv")),
            QuantType::Bf16 => include_bytes!(concat!(env!("OUT_DIR"), "/gemm_bf16.spv")),
            other => {
                return Err(anyhow!(
                    "no multi-user GEMM shader for {}: the delivered set is \
                     IQ4_NL, Q2_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, Q4_K, \
                     Q5_K, Q6_K",
                    other.name()
                ))
            }
        };
        let kernel = gpu
            .kernel(spv, 4)
            .map_err(|e| anyhow!("creating the {ty} multi-user GEMM pipeline: {e}"))?;
        let q8_spv: Option<&[u8]> = match ty {
            QuantType::Iq3S => Some(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/gemm_iq3_s_q8.spv"
            ))),
            _ => None,
        };
        let q8 = match q8_spv.map(|spv| gpu.kernel(spv, 6)).transpose() {
            Ok(k) => k,
            Err(e) => {
                gpu.destroy_kernel(&kernel);
                return Err(anyhow!("creating the {ty} int8 GEMM pipeline: {e}"));
            }
        };
        let pair_spv: Option<&[u8]> = match ty {
            QuantType::Iq3S => Some(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/gemm_iq3_s_pair.spv"
            ))),
            _ => None,
        };
        debug_assert_eq!(pair_spv.is_some(), has_pair_shader(ty));
        let pair = match pair_spv.map(|spv| gpu.kernel(spv, 4)).transpose() {
            Ok(k) => k,
            Err(e) => {
                gpu.destroy_kernel(&kernel);
                if let Some(k) = &q8 {
                    gpu.destroy_kernel(k);
                }
                return Err(anyhow!("creating the {ty} gate+up GEMM pipeline: {e}"));
            }
        };
        let mut blocked = Vec::new();
        let sets: &[bool] = if ldsx(gpu) { &[true, false] } else { &[false] };
        for (x, (rpl, spv)) in sets
            .iter()
            .flat_map(|&x| blocked_spv(ty, x).into_iter().map(move |b| (x, b)))
        {
            match gpu.kernel(spv, 4) {
                Ok(k) => blocked.push((rpl, x, k)),
                Err(e) => {
                    for k in [&kernel].into_iter().chain(q8.iter()).chain(pair.iter()) {
                        gpu.destroy_kernel(k);
                    }
                    for (_, _, k) in &blocked {
                        gpu.destroy_kernel(k);
                    }
                    return Err(anyhow!("creating the {ty} {rpl:?} GEMM pipeline: {e}"));
                }
            }
        }
        let mut tiles = Vec::new();
        for (t, spv) in tile_spv(ty) {
            match gpu.kernel(spv, 4) {
                Ok(k) => tiles.push((t, k)),
                Err(e) => {
                    gpu.destroy_kernel(&kernel);
                    for k in q8
                        .iter()
                        .chain(pair.iter())
                        .chain(blocked.iter().map(|(_, _, k)| k))
                    {
                        gpu.destroy_kernel(k);
                    }
                    for (_, k) in &tiles {
                        gpu.destroy_kernel(k);
                    }
                    return Err(anyhow!("creating the {ty} {t:?} tiled GEMM pipeline: {e}"));
                }
            }
        }
        let tables = match upload(gpu, &table_bytes()) {
            Ok(b) => b,
            Err(e) => {
                gpu.destroy_kernel(&kernel);
                for k in q8
                    .iter()
                    .chain(pair.iter())
                    .chain(blocked.iter().map(|(_, _, k)| k))
                    .chain(tiles.iter().map(|(_, k)| k))
                {
                    gpu.destroy_kernel(k);
                }
                return Err(e.context("uploading the quant tables"));
            }
        };
        Ok(GemmMany {
            ty,
            gpu,
            kernel,
            q8,
            pair,
            blocked,
            tiles,
            want_tile: env_tile(subgroup_size(gpu), ty),
            hcgate: None,
            want: [false, true].map(|x| {
                let t = tuned(ty, subgroup_size(gpu), x);
                (env_rpl().unwrap_or(t.0), env_umax().unwrap_or(t.1))
            }),
            upz: default_upz(),
            dot: if DotGemm::supported(gpu) {
                DotGemm::new_tiled(gpu).ok()
            } else {
                None
            },
            dot_weights: false,
            dot_bf16: false,
            dot_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
            tile_locked: false,
            tables,
        })
    }

    pub fn ty(&self) -> QuantType {
        self.ty
    }

    /// Force the users per z slice (clamped to 1..=USERS_MAX); `None` = adaptive.
    pub fn set_users_per_z(&mut self, n: Option<usize>) {
        self.upz = n.map(|n| n.clamp(1, USERS_MAX));
    }

    /// Force the tiled build (`None` = the GEMV-shaped builds); A/B and tests.
    /// Declare that this matrix's weights are **f16 pairs**, which is what the
    /// mixed-dot body reads. Without this the dot can never be dispatched, whatever
    /// the calibration says, so a quantised tensor cannot be misread as f16.
    pub fn set_dot_weights(&mut self, on: bool) {
        self.dot_weights = on;
    }

    /// Declare that this matrix's weights are **BF16** and may be converted to f16
    /// pairs on first use (the prepass). Separate from [`GemmMany::set_dot_weights`],
    /// which promises the buffer is already f16 pairs.
    pub fn set_dot_bf16(&mut self, on: bool) {
        self.dot_bf16 = on;
    }

    /// Whether the dot body is available and declared usable for this matrix.
    pub fn dot_available(&self) -> bool {
        self.dot.is_some() && self.dot_weights
    }

    pub fn set_tile(&mut self, t: Option<Tile>) {
        self.want_tile = t;
        self.tile_locked = t.is_some();
    }

    /// Build the HC `up` + gated-mean pipeline (BF16 only; default on, `BRECCIA_FUSE_HC=0` opts out).
    /// A no-op for other types. See [`GemmMany::run_up_gated`].
    pub fn enable_hc_tile_gated(&mut self) -> Result<()> {
        if self.ty != QuantType::Bf16 || self.hcgate.is_some() {
            return Ok(());
        }
        let spv: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_bf16_t4x4_hcgate.spv"));
        let k = self
            .gpu
            .kernel(spv, 8)
            .map_err(|e| anyhow!("creating the BF16 HC up + gated-mean GEMM pipeline: {e}"))?;
        self.hcgate = Some(k);
        Ok(())
    }

    /// `true` when [`GemmMany::run_up_gated`] would replace the plain `run` for
    /// `users`: the pipeline is built, and the plain `run` would itself take the
    /// default 4x4 tile (so the fused arithmetic is the unfused arithmetic).
    pub fn hc_tile_gated_for(&self, users: usize) -> bool {
        self.hcgate.is_some() && self.want_tile == Some((4, 4)) && users >= TILE_MIN_USERS
    }

    /// The HC `up` GEMM (`rows = 4 * n_embd`, hc = 4) with the gated mean and the
    /// inject partials as its epilogue. Writes `mixed` `[users, n_embd]` and
    /// `part` `[users, 4, n_embd / 16]`; `gate` is NOT written (`gate_dummy` only
    /// satisfies binding 2). Needs `n_embd % 16 == 0` and [`GemmMany::hc_tile_gated_for`].
    #[allow(clippy::too_many_arguments)]
    pub fn run_up_gated(
        &self,
        w: &Buf,
        x: &Buf,
        gate_dummy: &Buf,
        xn: &Buf,
        winj: &Buf,
        part: &Buf,
        mixed: &Buf,
        n_embd: usize,
        cols: usize,
        users: usize,
    ) -> Result<()> {
        let kernel = self
            .hcgate
            .as_ref()
            .filter(|_| self.hc_tile_gated_for(users) && n_embd.is_multiple_of(16))
            .ok_or_else(|| anyhow!("HC up + gated-mean GEMM not available for {users} users"))?;
        let mut pc = [0u8; 28];
        pc[..24].copy_from_slice(&self.pc(4 * n_embd, cols, users, 0));
        pc[24..28].copy_from_slice(&(n_embd as u32).to_le_bytes());
        self.dispatch_grid(
            kernel,
            &[w, x, gate_dummy, &self.tables, xn, winj, part, mixed],
            &pc,
            ((n_embd / 16) as u32, users.div_ceil(64) as u32, 1),
        )
    }

    /// The tiled pipeline `run_at` would use for `rows x users`, if any.
    fn tiled(&self, rows: usize, users: usize) -> Option<(&breccia_vk::ctx::Kernel, Tile)> {
        if users < TILE_MIN_USERS {
            return None;
        }
        let t = if self.tile_locked || tile_forced() {
            self.want_tile
        } else {
            tile_for(self.ty, rows, users)
        }
        .or(self.want_tile)?;
        self.tiles
            .iter()
            .find(|(c, _)| *c == t)
            .map(|(c, k)| (k, *c))
    }

    /// The tiled pipeline for exactly `t` (tests), ignoring the user threshold.
    fn tile_kernel(&self, t: Tile) -> Option<&breccia_vk::ctx::Kernel> {
        self.tiles.iter().find(|(c, _)| *c == t).map(|(_, k)| k)
    }

    /// Dispatch grid of tile `t`: `(ceil(rows / TM), ceil(users / TN), 1)`.
    fn tile_grid(t: Tile, rows: usize, users: usize) -> (u32, u32, u32) {
        let (tm, tn) = tile_dims(t);
        (
            rows.div_ceil(tm).max(1) as u32,
            users.div_ceil(tn).max(1) as u32,
            1,
        )
    }

    /// [`GemmMany::run_at`] on the tiled build `t` (tests / A/B); `None` if `t` is not built.
    #[allow(clippy::too_many_arguments)]
    pub fn run_tiled(
        &self,
        t: Tile,
        w: &Buf,
        x: &Buf,
        out: &Buf,
        rows: usize,
        cols: usize,
        users: usize,
    ) -> Result<()> {
        if users == 0 {
            return Err(anyhow!("users must be at least 1"));
        }
        let kernel = self
            .tile_kernel(t)
            .ok_or_else(|| anyhow!("no tiled build {t:?} for {}", self.ty))?;
        let pc = self.pc(rows, cols, users, 0);
        self.dispatch_grid(
            kernel,
            &[w, x, out, &self.tables],
            &pc,
            Self::tile_grid(t, rows, users),
        )
    }

    /// Timed [`GemmMany::run_tiled`].
    #[allow(clippy::too_many_arguments)]
    pub fn run_timed_tiled(
        &self,
        t: Tile,
        w: &Buf,
        x: &Buf,
        out: &Buf,
        rows: usize,
        cols: usize,
        users: usize,
        target_s: f64,
    ) -> Result<GemvTiming> {
        let kernel = self
            .tile_kernel(t)
            .ok_or_else(|| anyhow!("no tiled build {t:?} for {}", self.ty))?;
        let pc = self.pc(rows, cols, users, 0);
        self.timed_grid(
            kernel,
            &[w, x, out, &self.tables],
            &pc,
            Self::tile_grid(t, rows, users),
            target_s,
        )
    }

    /// Force the build: rows per lane group (1, 2 or 4) and users per decode
    /// (8 or 16); A/B and tests.
    pub fn set_variant(&mut self, rpl: usize, umax: usize) {
        self.want = [(rpl, umax); 2];
    }

    fn upz_max(&self, rows: usize, users: usize, umax: usize) -> usize {
        self.upz.unwrap_or_else(|| upz_for(rows, users, umax))
    }

    /// Whether [`GemmMany::run_at_q8`] exists for this type.
    pub fn has_q8(&self) -> bool {
        self.q8.is_some()
    }

    /// Blocks per row for `cols`.
    pub fn blocks_per_row(&self, cols: usize) -> usize {
        cols.div_ceil(self.ty.block_elems())
    }

    /// Padded activation columns: `X` must hold this many f32 per user (the same
    /// zero-padded layout the single-user kernels take).
    pub fn padded_cols(&self, cols: usize) -> usize {
        self.blocks_per_row(cols) * self.ty.block_elems()
    }

    fn pc(&self, rows: usize, cols: usize, users: usize, base_words: u32) -> [u8; 24] {
        self.pc_v(rows, cols, users, base_words, (1, USERS_MAX))
    }

    /// [`GemmMany::pc`] for a `(rpl, umax)` build: the z split sees `rows / rpl`
    /// lane groups (the count [`GemmMany::dispatch`] gets) and up to `umax` users.
    fn pc_v(
        &self,
        rows: usize,
        cols: usize,
        users: usize,
        base_words: u32,
        v: Variant,
    ) -> [u8; 24] {
        let (rpl, umax) = v;
        let bpr = self.blocks_per_row(cols);
        let mut pc = [0u8; 24];
        pc[0..4].copy_from_slice(&(rows as u32).to_le_bytes());
        pc[4..8].copy_from_slice(&(cols as u32).to_le_bytes());
        pc[8..12].copy_from_slice(&(bpr as u32).to_le_bytes());
        pc[12..16].copy_from_slice(&(users as u32).to_le_bytes());
        pc[16..20].copy_from_slice(&base_words.to_le_bytes());
        pc[20..24].copy_from_slice(&(self.upz_max(rows / rpl, users, umax) as u32).to_le_bytes());
        pc
    }

    /// Run one GEMM: `out[u*rows + r] = W[r] · x_u`, `u < users`.
    pub fn run(
        &self,
        w: &Buf,
        x: &Buf,
        out: &Buf,
        rows: usize,
        cols: usize,
        users: usize,
    ) -> Result<()> {
        self.run_at(w, x, out, rows, cols, users, 0)
    }

    /// [`GemmMany::run`] with the expert's blob at `base_words` (u16 words) inside
    /// `w` — the resident arena's addressing (`MoeMap`'s convention, §8v).
    #[allow(clippy::too_many_arguments)]
    pub fn run_at(
        &self,
        w: &Buf,
        x: &Buf,
        out: &Buf,
        rows: usize,
        cols: usize,
        users: usize,
        base_words: u32,
    ) -> Result<()> {
        if users == 0 {
            return Err(anyhow!("users must be at least 1"));
        }
        // Calibrated dot cell + declared weights: run the mixed-dot body.
        // `dot_weights` promises f16 pairs already; `dot_bf16` converts the tensor
        // once here (the prepass) and caches it by source buffer.
        if base_words == 0 {
            let choice = choice_from_table(self.ty, rows, users);
            if choice != Some(Choice::Dot) && (self.dot_weights || self.dot_bf16) {
                dot_log_reject(rows, cols, users, choice, self.dot.is_some());
            }
            if let (Some(Choice::Dot), Some(dot)) = (choice, self.dot.as_ref()) {
                if self.dot_weights {
                    dot_log(rows, cols, users);
                    return dot.run_tiled(w, x, out, rows, cols, users);
                }
                if self.dot_bf16 && self.ty == QuantType::Bf16 {
                    let mut cache = self.dot_cache.lock().unwrap_or_else(|e| e.into_inner());
                    if let std::collections::hash_map::Entry::Vacant(slot) = cache.entry(w.id) {
                        // A refused conversion keeps the BF16 path: running on a
                        // different matrix than the caller passed is not an option.
                        if let Ok(b) = f16_pairs_from_bf16(self.gpu, w, rows, cols) {
                            slot.insert(b);
                        }
                    }
                    if let Some(b) = cache.get(&w.id) {
                        dot_log(rows, cols, users);
                        return dot.run_tiled(b, x, out, rows, cols, users);
                    }
                }
            }
        }
        if let Some((kernel, t)) = self.tiled(rows, users) {
            let pc = self.pc(rows, cols, users, base_words);
            return self.dispatch_grid(
                kernel,
                &[w, x, out, &self.tables],
                &pc,
                Self::tile_grid(t, rows, users),
            );
        }
        let (kernel, v) = self.plain(rows, users);
        let pc = self.pc_v(rows, cols, users, base_words, v);
        self.dispatch(kernel, &[w, x, out, &self.tables], &pc, rows / v.0, users)
    }

    /// The variant `run` would use for this shape (`(1, USERS_MAX)` = plain build).
    pub fn variant_for(&self, rows: usize, users: usize) -> Variant {
        self.plain(rows, users).1
    }

    /// The plain-GEMM pipeline for `rows x users` and its variant: the wanted
    /// build when `rows` divides and its fewer workgroups still fill the GPU.
    fn plain(&self, rows: usize, users: usize) -> (&breccia_vk::ctx::Kernel, Variant) {
        self.blocked
            .iter()
            .find(|((r, u), x, _)| {
                (*r, *u) == self.want[*x as usize]
                    && (!*x || self.upz_max(rows / r, users, *u).min(users) >= LDSX_MIN_USERS)
                    && rows.is_multiple_of(*r)
                    && (rows / r).div_ceil(ROWS_PER_WG) * users.div_ceil(*u) >= WG_TARGET
            })
            .map_or((&self.kernel, (1, USERS_MAX)), |(v, _, k)| {
                static SEEN: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !SEEN.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!(
                        "gemm: blocked build {v:?} in use ({} {rows} x {users})",
                        self.ty.name()
                    );
                }
                (k, *v)
            })
    }

    /// Whether [`GemmMany::run_pair`] exists for this type.
    pub fn has_pair(&self) -> bool {
        self.pair.is_some()
    }

    /// Fused gate + up + SwiGLU: `w` holds `rows` gate rows then `rows` up rows,
    /// `out[u*rows + r] = silu(gate_r · x_u) * (up_r · x_u)`.
    pub fn run_pair(
        &self,
        w: &Buf,
        x: &Buf,
        out: &Buf,
        rows: usize,
        cols: usize,
        users: usize,
    ) -> Result<()> {
        self.run_pair_at(w, x, out, rows, cols, users, 0)
    }

    /// [`GemmMany::run_pair`] with the gate rows starting `base_words` (u16 words)
    /// into `w`; the up rows follow the gate rows exactly (no padding between).
    #[allow(clippy::too_many_arguments)]
    pub fn run_pair_at(
        &self,
        w: &Buf,
        x: &Buf,
        out: &Buf,
        rows: usize,
        cols: usize,
        users: usize,
        base_words: u32,
    ) -> Result<()> {
        if users == 0 {
            return Err(anyhow!("users must be at least 1"));
        }
        let kernel = self
            .pair
            .as_ref()
            .ok_or_else(|| anyhow!("no gate+up GEMM shader for {}", self.ty))?;
        let pc = self.pc(rows, cols, users, base_words);
        self.dispatch(kernel, &[w, x, out, &self.tables], &pc, rows, users)
    }

    /// [`GemmMany::run_at`] on int8 dot products: `xq` holds `users * cols` int8
    /// (row-major per user) and `xd` one f16 scale per 32 of them, as
    /// [`crate::Q8Quantize`] writes. Needs `cols` a multiple of the block size.
    #[allow(clippy::too_many_arguments)]
    pub fn run_at_q8(
        &self,
        w: &Buf,
        xq: &Buf,
        xd: &Buf,
        out: &Buf,
        rows: usize,
        cols: usize,
        users: usize,
        base_words: u32,
    ) -> Result<()> {
        let (kernel, pc) = self.q8_launch(rows, cols, users, base_words)?;
        // Binding 1 (the fp32 X the body does not read) is bound to xq.
        self.dispatch(
            kernel,
            &[w, xq, out, &self.tables, xq, xd],
            &pc,
            rows,
            users,
        )
    }

    fn q8_launch(
        &self,
        rows: usize,
        cols: usize,
        users: usize,
        base_words: u32,
    ) -> Result<(&breccia_vk::ctx::Kernel, [u8; 24])> {
        let kernel = self
            .q8
            .as_ref()
            .ok_or_else(|| anyhow!("no int8 GEMM for {}", self.ty))?;
        if users == 0 {
            return Err(anyhow!("users must be at least 1"));
        }
        if !cols.is_multiple_of(self.ty.block_elems()) {
            return Err(anyhow!(
                "int8 GEMM needs cols ({cols}) a multiple of {}",
                self.ty.block_elems()
            ));
        }
        Ok((kernel, self.pc(rows, cols, users, base_words)))
    }

    fn dispatch(
        &self,
        kernel: &breccia_vk::ctx::Kernel,
        bufs: &[&Buf],
        pc: &[u8; 24],
        rows: usize,
        users: usize,
    ) -> Result<()> {
        self.dispatch_grid(kernel, bufs, pc, Self::std_grid(rows, users, pc))
    }

    /// Grid of the GEMV-shaped builds: 8 row groups per x, one z slice per `upz` users.
    fn std_grid(rows: usize, users: usize, pc: &[u8; 24]) -> (u32, u32, u32) {
        let groups = rows.div_ceil(ROWS_PER_WG).max(1) as u32;
        (
            groups.min(WG_X_MAX),
            groups.div_ceil(WG_X_MAX),
            users.div_ceil(pushed_upz(pc)) as u32,
        )
    }

    fn dispatch_grid(
        &self,
        kernel: &breccia_vk::ctx::Kernel,
        bufs: &[&Buf],
        pc: &[u8],
        (gx, gy, gz): (u32, u32, u32),
    ) -> Result<()> {
        // Inside a dispatch batch the dispatch is **recorded** (the same rule
        // `Pipe::run`, `Gemv::run` and `GemvBatch::run` follow): a chunked prefill
        // records ~200 expert groups per block, and one submit+wait each would be
        // 40-60 µs of round trip per GEMM.
        if crate::dispatch_batch::is_open() {
            return crate::dispatch_batch::record(self.gpu, kernel, bufs, pc, (gx, gy, gz));
        }
        let handles: Vec<vk::Buffer> = bufs.iter().map(|b| b.buf).collect();
        self.gpu
            .bind_kernel(kernel, &handles)
            .map_err(|e| anyhow!("{e}"))?;
        self.gpu
            .wall_run(Qc::C, |d, cmd| unsafe {
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kernel.pipe);
                d.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    kernel.layout,
                    0,
                    &[kernel.set],
                    &[],
                );
                d.cmd_push_constants(cmd, kernel.layout, vk::ShaderStageFlags::COMPUTE, 0, pc);
                d.cmd_dispatch(cmd, gx, gy, gz);
            })
            .map_err(|e| anyhow!("{e}"))?;
        Ok(())
    }

    /// Time `users` GEMMs in one dispatch, GPU-timestamped.
    #[allow(clippy::too_many_arguments)]
    pub fn run_timed(
        &self,
        w: &Buf,
        x: &Buf,
        out: &Buf,
        rows: usize,
        cols: usize,
        users: usize,
        target_s: f64,
    ) -> Result<GemvTiming> {
        if let Some((_, t)) = self.tiled(rows, users) {
            return self.run_timed_tiled(t, w, x, out, rows, cols, users, target_s);
        }
        let (kernel, v) = self.plain(rows, users);
        let pc = self.pc_v(rows, cols, users, 0, v);
        self.timed(
            kernel,
            &[w, x, out, &self.tables],
            &pc,
            rows / v.0,
            users,
            target_s,
        )
    }

    /// Timed variant of [`GemmMany::run_at_q8`] (base 0).
    #[allow(clippy::too_many_arguments)]
    pub fn run_timed_q8(
        &self,
        w: &Buf,
        xq: &Buf,
        xd: &Buf,
        out: &Buf,
        rows: usize,
        cols: usize,
        users: usize,
        target_s: f64,
    ) -> Result<GemvTiming> {
        let (kernel, pc) = self.q8_launch(rows, cols, users, 0)?;
        self.timed(
            kernel,
            &[w, xq, out, &self.tables, xq, xd],
            &pc,
            rows,
            users,
            target_s,
        )
    }

    fn timed(
        &self,
        kernel: &breccia_vk::ctx::Kernel,
        bufs: &[&Buf],
        pc: &[u8; 24],
        rows: usize,
        users: usize,
        target_s: f64,
    ) -> Result<GemvTiming> {
        self.timed_grid(kernel, bufs, pc, Self::std_grid(rows, users, pc), target_s)
    }

    fn timed_grid(
        &self,
        kernel: &breccia_vk::ctx::Kernel,
        bufs: &[&Buf],
        pc: &[u8; 24],
        (gx, gy, gz): (u32, u32, u32),
        target_s: f64,
    ) -> Result<GemvTiming> {
        let handles: Vec<vk::Buffer> = bufs.iter().map(|b| b.buf).collect();
        self.gpu
            .bind_kernel(kernel, &handles)
            .map_err(|e| anyhow!("{e}"))?;
        let t = self
            .gpu
            .run_timed(Qc::C, target_s, |d, cmd, n| {
                for _ in 0..n {
                    unsafe {
                        d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, kernel.pipe);
                        d.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::COMPUTE,
                            kernel.layout,
                            0,
                            &[kernel.set],
                            &[],
                        );
                        d.cmd_push_constants(
                            cmd,
                            kernel.layout,
                            vk::ShaderStageFlags::COMPUTE,
                            0,
                            pc,
                        );
                        d.cmd_dispatch(cmd, gx, gy, gz);
                        Gpu::barrier_all(d, cmd);
                    }
                }
            })
            .map_err(|e| anyhow!("{e}"))?;
        Ok(GemvTiming {
            per_iter_s: t.per_iter_s,
            iters: t.iters,
            total_s: t.total_s,
            used_timestamps: t.used_timestamps,
        })
    }

    /// Bytes of quantised weights the dispatch reads (one expert's role tensor).
    pub fn weight_bytes(&self, rows: usize, cols: usize) -> u64 {
        (rows * self.blocks_per_row(cols) * self.ty.block_bytes()) as u64
    }
}

impl Drop for GemmMany<'_> {
    fn drop(&mut self) {
        self.gpu.destroy_kernel(&self.kernel);
        for k in self.q8.iter().chain(self.pair.iter()) {
            self.gpu.destroy_kernel(k);
        }
        for (_, _, k) in &self.blocked {
            self.gpu.destroy_kernel(k);
        }
        for (_, k) in &self.tiles {
            self.gpu.destroy_kernel(k);
        }
        if let Some(k) = &self.hcgate {
            self.gpu.destroy_kernel(k);
        }
        self.gpu.destroy_buf(&self.tables);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn upz_splits_only_narrow_dispatches() {
        use super::upz_for;
        // 640-row expert gate/up: 80 WGs per slice.
        assert_eq!(upz_for(640, 8, super::USERS_MAX), 2);
        assert_eq!(upz_for(640, 16, super::USERS_MAX), 4);
        assert_eq!(upz_for(640, 1, super::USERS_MAX), 2);
        // down (2560 rows) and a 1024-token dense chunk keep USERS_MAX.
        assert_eq!(upz_for(2560, 8, super::USERS_MAX), 8);
        assert_eq!(upz_for(640, 1024, super::USERS_MAX), 8);
    }
}

#[cfg(test)]
mod table_lock {
    /// The calibrated table is process-global, so the tests that install one must
    /// not run concurrently.
    pub static L: std::sync::Mutex<()> = std::sync::Mutex::new(());
}

#[cfg(test)]
mod tile_table_tests {
    use super::*;

    fn cell(ty: &str, rows: usize, users: usize, tile: &str) -> TileCell {
        TileCell {
            ty: ty.into(),
            rows,
            users,
            tile: tile.into(),
            us: 1.0,
        }
    }

    /// A calibrated table installed at startup must win over the built-in rule, and
    /// must be per type: BF16 and IQ3_S legitimately disagree at the same shape.
    #[test]
    fn the_installed_table_overrides_the_builtin_rule() {
        let _g = super::table_lock::L
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // Built-in rule says 4x2 for a narrow shape at 128 users.
        assert_eq!(tile_for(QuantType::Iq3S, 640, 128), Some((4, 2)));
        set_tile_table(vec![
            cell("IQ3_S", 640, 128, "8x8"),
            cell("BF16", 640, 128, "4x4"),
        ]);
        assert_eq!(tile_for(QuantType::Iq3S, 640, 128), Some((8, 8)));
        assert_eq!(tile_for(QuantType::Bf16, 640, 128), Some((4, 4)));
        // A type the table does not cover falls back to the rule.
        assert_eq!(tile_for(QuantType::Q6K, 640, 128), Some((4, 2)));
        // Nearest cell wins for a shape between samples.
        set_tile_table(vec![
            cell("IQ3_S", 640, 64, "4x2"),
            cell("IQ3_S", 2560, 1024, "8x8"),
        ]);
        assert_eq!(tile_for(QuantType::Iq3S, 700, 70), Some((4, 2)));
        assert_eq!(tile_for(QuantType::Iq3S, 2400, 1000), Some((8, 8)));
        // An empty table restores the rule.
        set_tile_table(Vec::new());
        assert_eq!(tile_for(QuantType::Iq3S, 640, 128), Some((4, 2)));
    }
}

/// A mixed-float-dot GEMM: `OpFDot2MixAcc32VALVE` does two f16 MACs into an f32
/// accumulator per instruction, measured at 1.70x (gfx1030) / 1.78x (gfx906) the
/// fp32 FMA rate. Unreachable from GLSL, so the body is assembly
/// (`shaders/gemm_f16_dot.spvasm`, built by `build.rs` with spirv-as + spirv-val).
///
/// **Quant-agnostic**: the kernel takes weights as f16 pairs, so any format the
/// caller can present as f16 (F16 directly, BF16/F32 through a range check, IQ*
/// through its dequant state) uses the same kernel. **System-agnostic**: it is
/// refused up front unless the device advertises
/// `VK_VALVE_shader_mixed_float_dot_product`, and the choice between it and the
/// tiled builds belongs to calibration (`breccia profile --kernels`), not to a
/// compile-time switch.
///
/// Layout: `w` holds `rows*cols` f16 (two per 32-bit word), `x` is f32 `users*cols`
/// and `o` f32 `users*rows` at `O[u * rows + r]` — the same output contract as
/// every other GEMM build. `cols` must be even.
pub struct DotGemm<'g> {
    kernel: breccia_vk::ctx::Kernel,
    gpu: &'g Gpu,
}

impl<'g> DotGemm<'g> {
    /// Does this device advertise the extension the kernel needs?
    pub fn supported(gpu: &Gpu) -> bool {
        gpu.feats.mixed_float_dot
    }

    /// Tile of the tiled body: TM rows per workgroup, TN users (`build.rs` panel).
    pub const TILE_ROWS: usize = 64;
    pub const TILE_USERS: usize = 8;

    /// The **tiled** body: X staged in LDS as f16 pairs, one W row per thread
    /// reused across the workgroup's users. Same arithmetic, same bindings.
    pub fn new_tiled(gpu: &'g Gpu) -> Result<Self> {
        if !Self::supported(gpu) {
            anyhow::bail!(
                "{} does not advertise VK_VALVE_shader_mixed_float_dot_product",
                gpu.pci_address
            );
        }
        let kernel = gpu
            .kernel(crate::MIXED_DOT_GEMM_TILED, 3)
            .map_err(|e| anyhow!("creating the tiled mixed-dot GEMM pipeline: {e}"))?;
        Ok(DotGemm { kernel, gpu })
    }

    /// Compile the pipeline. Errors (rather than silently falling back) when the
    /// device cannot run it, so a caller's capability check is not a guess.
    pub fn new(gpu: &'g Gpu) -> Result<Self> {
        if !Self::supported(gpu) {
            anyhow::bail!(
                "{} does not advertise VK_VALVE_shader_mixed_float_dot_product",
                gpu.pci_address
            );
        }
        let kernel = gpu
            .kernel(crate::MIXED_DOT_GEMM, 3)
            .map_err(|e| anyhow!("creating the mixed-dot GEMM pipeline: {e}"))?;
        Ok(DotGemm { kernel, gpu })
    }

    /// [`DotGemm::run`] with the tiled grid (`ceil(rows/64) x ceil(users/8)`).
    pub fn run_tiled(
        &self,
        w: &Buf,
        x: &Buf,
        o: &Buf,
        rows: usize,
        cols: usize,
        users: usize,
    ) -> Result<()> {
        if rows == 0 || cols == 0 || users == 0 {
            return Ok(());
        }
        if !cols.is_multiple_of(2) {
            anyhow::bail!("mixed-dot GEMM needs an even `cols`, got {cols}");
        }
        let grid = (
            rows.div_ceil(Self::TILE_ROWS).max(1) as u32,
            users.div_ceil(Self::TILE_USERS).max(1) as u32,
            1,
        );
        self.launch(w, x, o, rows, cols, users, grid)
    }

    /// `o[u * rows + r] = sum_k w[r*cols + k] * x[u*cols + k]`, f32 accumulation.
    pub fn run(
        &self,
        w: &Buf,
        x: &Buf,
        o: &Buf,
        rows: usize,
        cols: usize,
        users: usize,
    ) -> Result<()> {
        if rows == 0 || cols == 0 || users == 0 {
            return Ok(());
        }
        if !cols.is_multiple_of(2) {
            anyhow::bail!("mixed-dot GEMM needs an even `cols`, got {cols}");
        }
        let n = (rows * users) as u64;
        let groups = n.div_ceil(64).max(1) as u32;
        self.launch(w, x, o, rows, cols, users, (groups, 1, 1))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch(
        &self,
        w: &Buf,
        x: &Buf,
        o: &Buf,
        rows: usize,
        cols: usize,
        users: usize,
        grid: (u32, u32, u32),
    ) -> Result<()> {
        let pc = crate::ops::Pc::new()
            .u(rows as u32)
            .u(cols as u32)
            .u(users as u32);
        self.gpu
            .bind_kernel(&self.kernel, &[w.buf, x.buf, o.buf])
            .map_err(|e| anyhow!("{e}"))?;
        self.gpu
            .wall_run(breccia_vk::ctx::Qc::C, |d, cmd| unsafe {
                d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.kernel.pipe);
                d.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.kernel.layout,
                    0,
                    &[self.kernel.set],
                    &[],
                );
                d.cmd_push_constants(
                    cmd,
                    self.kernel.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    pc.as_bytes(),
                );
                d.cmd_dispatch(cmd, grid.0, grid.1, grid.2);
            })
            .map_err(|e| anyhow!("mixed-dot GEMM dispatch: {e}"))?;
        Ok(())
    }
}

#[cfg(test)]
mod dot_selection_tests {
    use super::*;

    fn cell(ty: &str, rows: usize, users: usize, tile: &str) -> TileCell {
        TileCell {
            ty: ty.into(),
            rows,
            users,
            tile: tile.into(),
            us: 1.0,
        }
    }

    /// A calibrated `dot` cell must be what the dispatcher consults, and it must not
    /// disturb the tile choice (which skips `dot` cells entirely).
    #[test]
    fn a_dot_cell_is_a_choice_and_not_a_tile() {
        let _g = super::table_lock::L
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        set_tile_table(vec![
            cell("IQ3_S", 640, 128, "dot"),
            cell("IQ3_S", 640, 1024, "4x4"),
        ]);
        assert_eq!(
            choice_from_table(QuantType::Iq3S, 640, 128),
            Some(Choice::Dot)
        );
        assert_eq!(
            choice_from_table(QuantType::Iq3S, 640, 1024),
            Some(Choice::Tile((4, 4)))
        );
        // `tile_for` ignores the dot cell and answers with the nearest real tile.
        assert_eq!(tile_for(QuantType::Iq3S, 640, 128), Some((4, 4)));
        // A type with no cells falls back to the built-in rule.
        assert_eq!(choice_from_table(QuantType::Q6K, 640, 128), None);
        set_tile_table(Vec::new());
    }
}

/// BF16 (`u16` bits, 8-bit exponent) as f32: exact, since f32 holds every BF16.
fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// f32 -> f16 bits, round-to-nearest-even. `None` when the value does not fit f16
/// (overflow to inf, NaN), which makes the caller keep the BF16 path instead of
/// silently running on a different matrix.
fn f32_to_f16(v: f32) -> Option<u16> {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x7f_ffff;
    if exp == 0xff {
        return None; // inf / NaN
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return None; // magnitude above f16's 65504
    }
    if e <= 0 {
        // Subnormal f16 (or zero): shift the implicit bit into place.
        if e < -10 {
            return Some(sign); // rounds to zero
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let tie = 1u32 << (shift - 1);
        let up = rem > tie || (rem == tie && (half & 1) == 1);
        return Some(sign | (half + up as u32) as u16);
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    let up = rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1);
    let rounded = half + up as u32;
    if (rounded & 0x7c00) == 0x7c00 {
        return None; // rounded up into inf
    }
    Some(sign | rounded as u16)
}

/// The prepass: read a BF16 weight matrix off the device, convert it to f16 pairs
/// (two per 32-bit word, low half first, the layout `gemm_f16_dot.spvasm` reads) and
/// upload the copy. One-time per tensor; same size as the source.
fn f16_pairs_from_bf16(gpu: &Gpu, w: &Buf, rows: usize, cols: usize) -> Result<Buf> {
    if !cols.is_multiple_of(2) {
        anyhow::bail!("f16 prepass needs an even `cols`, got {cols}");
    }
    let n = rows * cols;
    let bytes =
        crate::gemv::download(gpu, w, n * 2).map_err(|e| anyhow!("f16 prepass read: {e}"))?;
    if bytes.len() != n * 2 {
        anyhow::bail!("f16 prepass read {} of {} bytes", bytes.len(), n * 2);
    }
    let mut out = Vec::with_capacity(n * 2);
    for pair in bytes.chunks_exact(4) {
        let lo = u16::from_le_bytes([pair[0], pair[1]]);
        let hi = u16::from_le_bytes([pair[2], pair[3]]);
        let a = f32_to_f16(bf16_to_f32(lo))
            .ok_or_else(|| anyhow!("weight does not fit f16 ({})", bf16_to_f32(lo)))?;
        let b = f32_to_f16(bf16_to_f32(hi))
            .ok_or_else(|| anyhow!("weight does not fit f16 ({})", bf16_to_f32(hi)))?;
        let word = (a as u32) | ((b as u32) << 16);
        out.extend_from_slice(&word.to_le_bytes());
    }
    crate::gemv::upload(gpu, &out).map_err(|e| anyhow!("f16 prepass upload: {e}"))
}

/// `BRECCIA_DOT_LOG=1`: name each (shape, users) cell the BF16 prepass hands to the
/// mixed-dot body, once per process — the evidence that a real model run uses it.
fn dot_log(rows: usize, cols: usize, users: usize) {
    if std::env::var("BRECCIA_DOT_LOG").map_or(true, |v| v == "0") {
        return;
    }
    static SEEN: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<(usize, usize)>>> =
        std::sync::OnceLock::new();
    let seen = SEEN.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()));
    if seen
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert((rows, users))
    {
        eprintln!("breccia: mixed-dot on BF16 {rows}x{cols} x {users} users");
    }
}

/// `BRECCIA_DOT_LOG=1`: once per shape, why a BF16 GEMM eligible for the prepass did
/// not take the dot (no calibrated cell, or no pipeline). Diagnostic only.
fn dot_log_reject(rows: usize, cols: usize, users: usize, c: Option<Choice>, have: bool) {
    if std::env::var("BRECCIA_DOT_LOG").map_or(true, |v| v == "0") {
        return;
    }
    static SEEN: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<(usize, usize)>>> =
        std::sync::OnceLock::new();
    let seen = SEEN.get_or_init(|| std::sync::Mutex::new(std::collections::HashSet::new()));
    if seen
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert((rows, users))
    {
        eprintln!(
            "breccia: dot declined {rows}x{cols} x {users} users: cell {:?}, pipeline {have}",
            c
        );
    }
}
