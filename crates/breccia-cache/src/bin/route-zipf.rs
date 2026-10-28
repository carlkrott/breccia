//! `route-zipf` — read a `BRECCIA_TRACE_ROUTE_FILE` sidecar and print the
//! per-block hit histogram plus the zipf exponent and skew, per block and
//! pooled over the whole trace.
//!
//!     cargo run -p breccia-cache --bin route-zipf -- routes.txt

use anyhow::{bail, Context, Result};

fn main() -> Result<()> {
    let path = std::env::args_os()
        .nth(1)
        .context("usage: route-zipf SIDECAR")?;
    let path = std::path::PathBuf::from(path);
    let rows = breccia_cache::read_route_sidecar(&path)?;
    if rows.is_empty() {
        bail!("{} holds no routing rows", path.display());
    }
    let (blocks, overall) = breccia_cache::analyze_routes(&rows);
    println!("# sidecar {}: {} rows", path.display(), rows.len());
    for b in blocks.iter().chain(std::iter::once(&overall)) {
        let who = b
            .layer
            .map_or_else(|| "overall".to_string(), |l| format!("block {l}"));
        let histogram = b
            .histogram
            .iter()
            .map(|(e, n)| format!("{e}:{n}"))
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "{who} draws={} experts={} zipf={:.4} skew={:.4} histogram={histogram}",
            b.draws, b.experts, b.zipf, b.skew
        );
    }
    Ok(())
}
