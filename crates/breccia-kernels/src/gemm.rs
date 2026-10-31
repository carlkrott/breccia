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
    match std::env::var("BRECCIA_GEMM_TILE").ok().as_deref().map(str::trim) {
        Some("0") => None,
        Some("1") => Some(TILE_FALLBACK),
        Some(v) if v.contains('x') => v
            .split_once('x')
            .and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))
            .filter(|t| TILES.contains(t)),
        _ => tile_default(ty, subgroup),
    }
}

const TILES: [Tile; 3] = [(4, 4), (8, 4), (8, 8)];
const TILE_FALLBACK: Tile = (8, 4);

/// Measured-faster tiled build (m = 1024, attn_gate / hc_ffn_up, `tests/gemm_many.rs`
/// `tiled_gemm_matches_plain`): 4x4 beat or matched 8x4 / 8x8 on RDNA2 (wave32) and
/// Vega20 (wave64) for every type measured, at 0.4x-0.5x the previous best's time.
fn tile_default(_ty: QuantType, _subgroup: u32) -> Option<Tile> {
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
    unsafe { gpu.instance.get_physical_device_properties2(gpu.pdev, &mut p2) };
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
            QuantType::Iq3S => Some(include_bytes!(concat!(env!("OUT_DIR"), "/gemm_iq3_s_q8.spv"))),
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
            QuantType::Iq3S => Some(include_bytes!(concat!(env!("OUT_DIR"), "/gemm_iq3_s_pair.spv"))),
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
        for (x, (rpl, spv)) in sets.iter().flat_map(|&x| blocked_spv(ty, x).into_iter().map(move |b| (x, b))) {
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
                    for k in q8.iter().chain(pair.iter()).chain(blocked.iter().map(|(_, _, k)| k)) {
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
    pub fn set_tile(&mut self, t: Option<Tile>) {
        self.want_tile = t;
    }

    /// Build the HC `up` + gated-mean pipeline (BF16 only; `BRECCIA_FUSE_HC=1`).
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
    fn tiled(&self, users: usize) -> Option<(&breccia_vk::ctx::Kernel, Tile)> {
        let t = self.want_tile.filter(|_| users >= TILE_MIN_USERS)?;
        self.tiles.iter().find(|(c, _)| *c == t).map(|(c, k)| (k, *c))
    }

    /// The tiled pipeline for exactly `t` (tests), ignoring the user threshold.
    fn tile_kernel(&self, t: Tile) -> Option<&breccia_vk::ctx::Kernel> {
        self.tiles.iter().find(|(c, _)| *c == t).map(|(_, k)| k)
    }

    /// Dispatch grid of tile `t`: `(ceil(rows / TM), ceil(users / TN), 1)`.
    fn tile_grid(t: Tile, rows: usize, users: usize) -> (u32, u32, u32) {
        let (tm, tn) = tile_dims(t);
        (rows.div_ceil(tm).max(1) as u32, users.div_ceil(tn).max(1) as u32, 1)
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
        let kernel = self.tile_kernel(t).ok_or_else(|| anyhow!("no tiled build {t:?} for {}", self.ty))?;
        let pc = self.pc(rows, cols, users, 0);
        self.dispatch_grid(kernel, &[w, x, out, &self.tables], &pc, Self::tile_grid(t, rows, users))
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
        let kernel = self.tile_kernel(t).ok_or_else(|| anyhow!("no tiled build {t:?} for {}", self.ty))?;
        let pc = self.pc(rows, cols, users, 0);
        self.timed_grid(kernel, &[w, x, out, &self.tables], &pc, Self::tile_grid(t, rows, users), target_s)
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
    fn pc_v(&self, rows: usize, cols: usize, users: usize, base_words: u32, v: Variant) -> [u8; 24] {
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
        if let Some((kernel, t)) = self.tiled(users) {
            let pc = self.pc(rows, cols, users, base_words);
            return self.dispatch_grid(kernel, &[w, x, out, &self.tables], &pc, Self::tile_grid(t, rows, users));
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
                    && rows % r == 0
                    && (rows / r).div_ceil(ROWS_PER_WG) * users.div_ceil(*u) >= WG_TARGET
            })
            .map_or((&self.kernel, (1, USERS_MAX)), |(v, _, k)| {
                static SEEN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                if !SEEN.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!("gemm: blocked build {v:?} in use ({} {rows} x {users})", self.ty.name());
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
        self.dispatch(kernel, &[w, xq, out, &self.tables, xq, xd], &pc, rows, users)
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
        if cols % self.ty.block_elems() != 0 {
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
        (groups.min(WG_X_MAX), groups.div_ceil(WG_X_MAX), users.div_ceil(pushed_upz(pc)) as u32)
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
        if let Some((_, t)) = self.tiled(users) {
            return self.run_timed_tiled(t, w, x, out, rows, cols, users, target_s);
        }
        let (kernel, v) = self.plain(rows, users);
        let pc = self.pc_v(rows, cols, users, 0, v);
        self.timed(kernel, &[w, x, out, &self.tables], &pc, rows / v.0, users, target_s)
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
        self.timed(kernel, &[w, xq, out, &self.tables, xq, xd], &pc, rows, users, target_s)
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
