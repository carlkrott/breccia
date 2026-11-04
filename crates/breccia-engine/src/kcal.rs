//! GEMM kernel calibration (`breccia profile --kernels`) and its automatic re-run.
//!
//! The per-card tile table is keyed on `breccia_kernels::GEMM_SPIRV_HASH`: a change
//! to any GEMM/dequant shader drops the table and the kernels fall back to their
//! built-in rule. So a load that finds a card's table stale re-calibrates that card
//! on the spot (minutes, once per shader change) instead of silently running slower.
//! A card with no table is left alone: first-time setup is `breccia profile
//! --kernels`. `BRECCIA_AUTO_CALIBRATE=0` turns the re-run off.

use anyhow::Result;
use breccia_format::ShardSet;
use breccia_kernels::gemm::{CalibShape, TileCell};
use breccia_kernels::{Device, QuantType};
use breccia_profile::schema::HwProfile;
use breccia_vk::ctx::Gpu;

/// User counts the engine produces: experts = merged-prefill groups, dense =
/// chunked prefill.
const EXPERT_USERS: [usize; 8] = [64, 96, 128, 192, 256, 512, 1024, 2048];
const DENSE_USERS: [usize; 6] = [256, 512, 1024, 2048, 4096, 8192];

/// Every distinct (type, rows, cols) of the model's `blk.*` weights (one expert's
/// slice for a 3-D tensor) with the user counts it runs at.
pub fn calib_shapes(shards: &ShardSet) -> Vec<CalibShape<'_>> {
    let mut shapes: Vec<CalibShape> = Vec::new();
    for t in shards.tensors() {
        if !t.name().starts_with("blk.") || !(2..=3).contains(&t.n_dims()) {
            continue;
        }
        let Some(ty) = QuantType::from_ggml(t.ty()) else {
            continue;
        };
        let cols = t.ne(0).unwrap_or(0) as usize;
        let rows = t.ne(1).unwrap_or(0) as usize;
        let n = t.ne(2).unwrap_or(1).max(1) as usize;
        let users: &[usize] = if t.n_dims() == 3 {
            &EXPERT_USERS
        } else {
            &DENSE_USERS
        };
        match shapes
            .iter_mut()
            .find(|s| (s.ty, s.rows, s.cols) == (ty, rows, cols))
        {
            Some(s) => {
                s.users.extend_from_slice(users);
                s.users.sort_unstable();
                s.users.dedup();
            }
            None => shapes.push(CalibShape {
                ty,
                rows,
                cols,
                weights: &t.as_slice()[..(t.n_bytes() as usize / n)],
                users: users.to_vec(),
            }),
        }
    }
    shapes
}

fn same_pci(a: &str, b: &str) -> bool {
    !a.is_empty() && breccia_vk::pci::normalize_address(a) == breccia_vk::pci::normalize_address(b)
}

/// Calibrate `gpu` and store its verdict in `prof` (not saved). `Ok(None)` when the
/// card has no profile entry.
pub fn calibrate_card(
    prof: &mut HwProfile,
    gpu: &Gpu,
    pci: &str,
    driver: String,
    shapes: &[CalibShape],
) -> Result<Option<Vec<TileCell>>> {
    let tiles = breccia_kernels::gemm::calibrate_shapes(gpu, shapes)?;
    let Some(entry) = prof
        .devices
        .iter_mut()
        .find(|e| same_pci(&e.identity.pci_address, pci))
    else {
        return Ok(None);
    };
    entry.kernel = Some(breccia_profile::schema::KernelVerdict {
        driver_version: driver,
        spirv_hash: breccia_kernels::GEMM_SPIRV_HASH.to_string(),
        timestamp: breccia_profile::schema::iso8601(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
        ),
        tiles: tiles
            .iter()
            .map(|t| breccia_profile::schema::TileChoice {
                ty: t.ty.clone(),
                rows: t.rows,
                cols: t.cols,
                users: t.users,
                tile: t.tile.clone(),
                us: t.us,
                err: t.err,
            })
            .collect(),
    });
    Ok(Some(tiles))
}

/// Re-calibrate every card of this load whose stored table was measured by other
/// GEMM shaders. Never fails the load: an error keeps the built-in rule.
pub fn recalibrate_stale(devices: &[Device], shards: &ShardSet, path: &std::path::Path) {
    if std::env::var("BRECCIA_AUTO_CALIBRATE").as_deref() == Ok("0") {
        return;
    }
    let Ok(Some(mut prof)) = breccia_profile::load_profile(path) else {
        return;
    };
    let stale: Vec<&Device> = devices
        .iter()
        .filter(|d| {
            prof.devices.iter().any(|e| {
                same_pci(&e.identity.pci_address, &d.info.pci_address)
                    && e.kernel
                        .as_ref()
                        .is_some_and(|k| k.spirv_hash != breccia_kernels::GEMM_SPIRV_HASH)
            })
        })
        .collect();
    if stale.is_empty() {
        return;
    }
    let shapes = calib_shapes(shards);
    for d in stale {
        eprintln!(
            "breccia: kernels: the tile table for {} ({}) was measured by other GEMM shaders; \
             re-calibrating {} shapes (once per shader change, BRECCIA_AUTO_CALIBRATE=0 skips)",
            d.info.name,
            d.info.pci_address,
            shapes.len()
        );
        let t0 = std::time::Instant::now();
        let r = calibrate_card(
            &mut prof,
            d.gpu(),
            &d.info.pci_address,
            d.info.driver.clone(),
            &shapes,
        )
        .and_then(|_| breccia_profile::save_profile(&prof, path));
        match r {
            Ok(()) => eprintln!(
                "breccia: kernels: {} re-calibrated in {:.0} s, stored in {}",
                d.info.pci_address,
                t0.elapsed().as_secs_f64(),
                path.display()
            ),
            Err(e) => eprintln!(
                "breccia: kernels: re-calibrating {} failed ({e:#}); built-in kernel rule",
                d.info.pci_address
            ),
        }
    }
}
