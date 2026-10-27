//! Expert-cache and locality primitives (BRIEF L59).
//!
//! Round 19 (doc `docs/research/19-mlx-and-llm-in-a-flash.md`) delivers the
//! CPU-only halves of its Top 3:
//!
//! * **R1** — routing-trace records + an offline residency replay that pits
//!   three host-lane cache policies against each other at a fixed expert
//!   budget: static frequency (doc 02 §3), LRU (doc 05 E3) and the paper's
//!   last-k-token window (§3.1). Pure simulation, no GPU.
//! * **R2** — an expert *span* sidecar: one contiguous, 4096-byte-aligned
//!   `[gate | up | down]` blob per (layer, expert), written once offline so
//!   each streamed expert becomes a single read. Role byte sizes differ here
//!   (gate/up 17/16-bit types, down 42), so the span carries per-role offsets.
//!
//! `shortcut: R2 writes the sidecar and its index but no wire-in; the loader/
//! `fill_stream_slot` hookup is the GPU qualification card's job, <when the
//! dual A/B schedule opens a window for it>.

use anyhow::{bail, Context, Result};
use std::io::Seek;

/// One decoded token's routed experts for one block, in router order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceRecord {
    pub token: u64,
    pub layer: usize,
    /// The `k` expert ids the router chose (order preserved).
    pub experts: Vec<u32>,
}

impl TraceRecord {
    pub fn parse(line: &str) -> Result<Option<Self>> {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return Ok(None);
        }
        let mut f = line.split_whitespace();
        let token: u64 = f
            .next()
            .context("missing token")?
            .parse()
            .context("bad token")?;
        let layer: usize = f
            .next()
            .context("missing layer")?
            .parse()
            .context("bad layer")?;
        let experts = f
            .map(|s| {
                s.parse()
                    .map_err(|e| anyhow::anyhow!("bad expert id {s}: {e}"))
            })
            .collect::<Result<Vec<_>>>()?;
        if experts.is_empty() {
            bail!("row with no experts");
        }
        Ok(Some(Self {
            token,
            layer,
            experts,
        }))
    }

    fn to_line(&self) -> String {
        let mut s = format!("{} {}", self.token, self.layer);
        for e in &self.experts {
            s.push(' ');
            s.push_str(&e.to_string());
        }
        s
    }
}

/// Read a trace file: `token layer e1..ek` per line, `#` comments allowed.
pub fn read_trace(path: &std::path::Path) -> Result<Vec<TraceRecord>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading trace {}", path.display()))?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        match TraceRecord::parse(line) {
            Ok(Some(r)) => out.push(r),
            Ok(None) => {}
            Err(e) => return Err(e).with_context(|| format!("trace line {}", n + 1)),
        }
    }
    Ok(out)
}

/// Write a trace file; [`read_trace`] round-trips it.
pub fn write_trace(path: &std::path::Path, records: &[TraceRecord]) -> Result<()> {
    let mut s = String::from("# token layer e1..ek\n");
    for r in records {
        s.push_str(&r.to_line());
        s.push('\n');
    }
    std::fs::write(path, s).with_context(|| format!("writing trace {}", path.display()))
}

/// A residency policy replayed over a trace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    /// Static: the budget's worth of most-frequently-routed experts is pinned
    /// before the run and never evicted (doc 02 §3).
    Frequency,
    /// LRU per (layer, expert) (doc 05 E3).
    Lru,
    /// The last `window` distinct experts the trace routed to, capped at the
    /// budget (paper §3.1).
    Window,
}

/// How the replay measured one policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReplayResult {
    pub policy: Policy,
    /// Cacheable experts (per the budget).
    pub budget: usize,
    /// Experts whose layer is streamed; hits can only be scored there.
    pub streamed_probes: usize,
    /// Of those, how many were served from the cache.
    pub hits: usize,
}

impl ReplayResult {
    /// Hit rate over streamed probes.
    pub fn hit_rate(&self) -> f64 {
        if self.streamed_probes == 0 {
            return 0.0;
        }
        self.hits as f64 / self.streamed_probes as f64
    }

    /// Host-lane expert computations per token: `(1 - h) × k` per streamed
    /// block (`doc 19` §4.1a).
    pub fn computations_per_token(
        &self,
        tokens: usize,
        k_per_block: usize,
        streamed_blocks: usize,
    ) -> f64 {
        if tokens == 0 {
            return 0.0;
        }
        ((self.streamed_probes - self.hits) as f64 / tokens as f64) * k_per_block as f64
            / streamed_blocks.max(1) as f64
    }
}

/// Replay `records` under `policy` with at most `budget` experts cached.
///
/// `streamed` marks the expert streaming blocks; a resident block's experts
/// come from VRAM either way, so only routed experts in streamed blocks count
/// as probes. Records must be in issue order (the replay is online for LRU and
/// window).
pub fn replay(
    records: &[TraceRecord],
    streamed: &[bool],
    budget: usize,
    policy: Policy,
) -> Result<ReplayResult> {
    use std::collections::{HashMap, HashSet};
    if streamed.is_empty() {
        bail!("no streamed-block mask");
    }
    let mut hits = 0usize;
    let mut probes = 0usize;
    match policy {
        Policy::Frequency => {
            let mut freq: HashMap<(usize, u32), usize> = HashMap::new();
            for r in records {
                for &e in &r.experts {
                    *freq.entry((r.layer, e)).or_default() += 1;
                }
            }
            let mut by_freq: Vec<((usize, u32), usize)> = freq.into_iter().collect();
            by_freq.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            let cached: HashSet<(usize, u32)> =
                by_freq.into_iter().take(budget).map(|(k, _)| k).collect();
            for r in records {
                if *streamed.get(r.layer).unwrap_or(&false) {
                    for &e in &r.experts {
                        probes += 1;
                        if cached.contains(&(r.layer, e)) {
                            hits += 1;
                        }
                    }
                }
            }
        }
        Policy::Lru => {
            // Order-preserving map: an entry's removal index encodes recency.
            let mut lru: Vec<(usize, u32)> = Vec::new();
            let probe = |lru: &mut Vec<(usize, u32)>,
                         layer: usize,
                         e: u32,
                         hits: &mut usize,
                         probes: &mut usize| {
                *probes += 1;
                let key = (layer, e);
                if let Some(i) = lru.iter().position(|&k| k == key) {
                    *hits += 1;
                    lru.remove(i);
                }
                lru.insert(0, key);
                if lru.len() > budget {
                    lru.truncate(budget);
                }
            };
            for r in records {
                if *streamed.get(r.layer).unwrap_or(&false) {
                    for &e in &r.experts {
                        probe(&mut lru, r.layer, e, &mut hits, &mut probes);
                    }
                }
            }
        }
        Policy::Window => {
            // The window is a FIFO of the most recent distinct experts.
            let mut win: std::collections::VecDeque<(usize, u32)> =
                std::collections::VecDeque::new();
            let mut ws: HashSet<(usize, u32)> = HashSet::new();
            for r in records {
                if *streamed.get(r.layer).unwrap_or(&false) {
                    for &e in &r.experts {
                        probes += 1;
                        let key = (r.layer, e);
                        if ws.contains(&key) {
                            hits += 1;
                            // Refresh recency: the window is LRU, not FIFO.
                            win.retain(|&k| k != key);
                            win.push_front(key);
                            continue;
                        }
                        if win.len() >= budget {
                            if let Some(ev) = win.pop_back() {
                                ws.remove(&ev);
                            }
                        }
                        win.push_front(key);
                        ws.insert(key);
                    }
                }
            }
        }
    }
    let statics = match policy {
        Policy::Frequency => None,
        _ => Some(()),
    };
    let _ = statics;
    Ok(ReplayResult {
        policy,
        budget,
        streamed_probes: probes,
        hits,
    })
}

/// Replay a trace under all three policies at one budget.
pub fn replay_all(
    records: &[TraceRecord],
    streamed: &[bool],
    budget: usize,
) -> Result<Vec<ReplayResult>> {
    [Policy::Frequency, Policy::Lru, Policy::Window]
        .iter()
        .map(|&p| replay(records, streamed, budget, p))
        .collect()
}

/// `minImportedHostPointerAlignment` floor for the expert-span sidecar
/// (`\srv\strata-data` requirement of the levers doc §3.2).
pub const SPAN_ALIGN: u64 = 4096;

fn align_up(x: u64, a: u64) -> u64 {
    if a == 0 {
        return x;
    }
    (x + a - 1) / a * a
}

/// Plan and indices for an expert-span sidecar.
///
/// One file: a 64-byte text header (`breccia expert-span v1\nlayers L\n`,
/// padded with `#` to 64 bytes and NULs), then per layer a directory of
/// `[u32 layer][u32 n_experts][2×u32 pad][u64 blob_bytes]` rows followed by
/// spans sorted by frequency rank. Simpler and still one contiguous read per
/// expert: a flat second index file `span-index.bin`:
/// layer rows of `n_experts × 12 bytes` = `[u32 offset_lo][u32 offset_hi][u32 blob_bytes]`,
/// offsets absolute, page-aligned, blob = `[gate|up|down]` with per-role
/// offsets derivable from `blob_bytes` and the layer's quant types.
#[derive(Clone, Debug)]
pub struct SpanLayout {
    /// `span-index.bin` per-layer rows, offsets in the span file.
    pub rows: Vec<Vec<SpanEntry>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpanEntry {
    /// Blob start in the span file (4096-aligned).
    pub offset: u64,
    /// Total blob bytes (`[gate|up|down]`).
    pub blob_bytes: u64,
}

impl SpanLayout {
    /// Build the layout for `layers` × `n_experts` blobs from `(layer, expert)
    /// → blob_bytes`, **ordered by rank**: `ranks[(layer, expert)]` is the
    /// frequency rank of that expert (0 = hottest; ties by id). Hottest first,
    /// so the arena fill is a contiguous prefix (levers doc §3.2).
    pub fn build(
        n_layers: usize,
        n_experts: usize,
        blob_bytes: impl Fn(usize, usize) -> Result<u64>,
        ranks: &[(usize, u32)],
    ) -> Result<Self> {
        let mut ordered: Vec<(usize, u32)> = (0..n_layers)
            .flat_map(|l| (0..n_experts).map(move |e| (l, e as u32)))
            .collect();
        ordered.sort_unstable_by_key(|k| {
            (ranks.iter().position(|r| r == k).unwrap_or(usize::MAX), *k)
        });
        let mut rows = vec![
            vec![
                SpanEntry {
                    offset: 0,
                    blob_bytes: 0,
                };
                n_experts
            ];
            n_layers
        ];
        let mut off = align_up(SPAN_HEADER_BYTES as u64, SPAN_ALIGN);
        for (l, e) in ordered {
            let bytes = blob_bytes(l, e as usize).with_context(|| format!("blk.{l} expert {e}"))?;
            rows[l][e as usize] = SpanEntry {
                offset: off,
                blob_bytes: bytes,
            };
            off = align_up(off + bytes, SPAN_ALIGN);
        }
        Ok(Self { rows })
    }

    /// The span for `(layer, expert)`.
    pub fn entry(&self, layer: usize, expert: usize) -> Result<SpanEntry> {
        self.rows
            .get(layer)
            .and_then(|r| r.get(expert))
            .copied()
            .with_context(|| format!("no span for blk.{layer} expert {expert}"))
    }

    /// Per-role offsets inside one blob, given the three role byte sizes.
    pub fn role_offsets(blob_bytes: u64, gate: u64, up: u64, down: u64) -> Result<[u64; 4]> {
        if gate + up + down != blob_bytes {
            bail!("role sizes {gate}+{up}+{down} != blob {blob_bytes}");
        }
        Ok([0, gate, gate + up, gate + up + down])
    }
}

const SPAN_HEADER_BYTES: usize = 64;

/// Write the span sidecar pair next to the model shards.
///
/// `blob(l, e)` must supply the exact `[gate | up | down]` bytes for one
/// expert (a copy — never a memory-map write-through into the GGUF shards).
pub fn write_spans(
    dir: &std::path::Path,
    layout: &SpanLayout,
    blob: impl Fn(usize, usize) -> Result<Vec<u8>>,
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let mut index: Vec<u8> = Vec::new();
    for row in &layout.rows {
        index.extend_from_slice(&(row.len() as u32).to_le_bytes());
        for e in row {
            index.extend_from_slice(&e.offset.to_le_bytes());
            index.extend_from_slice(&e.blob_bytes.to_le_bytes());
        }
    }
    std::fs::write(dir.join("span-index.bin"), &index)?;
    let mut file = std::fs::File::create(dir.join("expert-spans.bin"))?;
    let _ = SPAN_HEADER_BYTES;
    use std::io::Write;
    for (l, row) in layout.rows.iter().enumerate() {
        for (e, entry) in row.iter().enumerate() {
            let bytes = blob(l, e).with_context(|| format!("blk.{l} expert {e}"))?;
            if bytes.len() as u64 != entry.blob_bytes {
                bail!(
                    "blk.{l} expert {e}: supplied {} bytes vs layout Blob {}",
                    bytes.len(),
                    entry.blob_bytes
                );
            }
            let start = entry.offset as usize;
            let cur = file.seek(std::io::SeekFrom::Start(start as u64))?;
            if cur != start as u64 {
                bail!("span file seek to {start} returned {cur}");
            }
            file.write_all(&bytes)?;
            // Pad to the next 4096 boundary so the following blob stays
            // aligned.
            let pad = (align_up(start as u64 + bytes.len() as u64, SPAN_ALIGN) as usize)
                - start
                - bytes.len();
            file.write_all(&vec![0u8; pad])?;
        }
    }
    file.flush()?;
    Ok(())
}

/// Read a span sidecar pair.
pub fn open_spans(dir: &std::path::Path) -> Result<(SpanLayout, std::fs::File)> {
    let idx = std::fs::read(dir.join("span-index.bin"))?;
    let mut cursor = 0usize;
    let mut rows = Vec::new();
    while cursor < idx.len() {
        let n = u32::from_le_bytes(
            idx.get(cursor..cursor + 4)
                .with_context(|| "index truncated")?
                .try_into()
                .unwrap(),
        ) as usize;
        cursor += 4;
        let mut row = Vec::with_capacity(n);
        for _ in 0..n {
            let (o, b) = idx
                .get(cursor..cursor + 16)
                .with_context(|| "index truncated")?
                .split_at(8);
            row.push(SpanEntry {
                offset: u64::from_le_bytes(o.try_into().unwrap()),
                blob_bytes: u64::from_le_bytes(b.try_into().unwrap()),
            });
            cursor += 16;
        }
        rows.push(row);
    }
    let file = std::fs::File::open(dir.join("expert-spans.bin"))?;
    Ok((SpanLayout { rows }, file))
}

/// One routing row from the `BRECCIA_TRACE_ROUTE_FILE` sidecar:
/// `token block e1:w1 .. ek:wk` (weights in router order, `k` = top-k).
#[derive(Clone, Debug, PartialEq)]
pub struct RouteRow {
    pub token: u64,
    pub layer: usize,
    pub experts: Vec<(u32, f32)>,
}

impl RouteRow {
    pub fn parse(line: &str) -> Result<Option<Self>> {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            return Ok(None);
        }
        let mut f = line.split_whitespace();
        let token: u64 = f
            .next()
            .context("missing token")?
            .parse()
            .context("bad token")?;
        let layer: usize = f
            .next()
            .context("missing block")?
            .parse()
            .context("bad block")?;
        let experts = f
            .map(|s| {
                let (e, w) = s.split_once(':').context("expected expert:weight")?;
                Ok::<_, anyhow::Error>((
                    e.parse().with_context(|| format!("bad expert id {e}"))?,
                    w.parse().with_context(|| format!("bad weight {w}"))?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        if experts.is_empty() {
            bail!("row with no experts");
        }
        Ok(Some(Self {
            token,
            layer,
            experts,
        }))
    }
}

/// Read a route sidecar written by `BRECCIA_TRACE_ROUTE_FILE`.
pub fn read_route_sidecar(path: &std::path::Path) -> Result<Vec<RouteRow>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading sidecar {}", path.display()))?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        match RouteRow::parse(line) {
            Ok(Some(r)) => out.push(r),
            Ok(None) => {}
            Err(e) => return Err(e).with_context(|| format!("sidecar line {}", n + 1)),
        }
    }
    Ok(out)
}

/// Routing stats for one block — or for every block pooled (`layer: None`).
#[derive(Clone, Debug)]
pub struct BlockStats {
    pub layer: Option<usize>,
    /// Total expert picks counted (`rows × k`).
    pub draws: usize,
    /// Distinct experts routed to.
    pub experts: usize,
    /// `(expert, picks)`, hottest first (ties by id).
    pub histogram: Vec<(u32, usize)>,
    /// Power-law exponent from a log-log fit of picks against rank; higher is
    /// hotter. 0 when the block has a single distinct expert.
    pub zipf: f64,
    /// Share of draws taken by the hottest 10% of experts (at least one) —
    /// the one-number skew to compare blocks by.
    pub skew: f64,
}

fn stats(layer: Option<usize>, counts: &std::collections::BTreeMap<u32, usize>) -> BlockStats {
    let mut histogram: Vec<(u32, usize)> = counts.iter().map(|(&e, &n)| (e, n)).collect();
    histogram.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let draws: usize = histogram.iter().map(|x| x.1).sum();
    let n = histogram.len();
    let top = ((n as f64 * 0.1).ceil() as usize).max(1).min(n);
    let hot: usize = histogram.iter().take(top).map(|x| x.1).sum();
    BlockStats {
        layer,
        draws,
        experts: n,
        zipf: zipf_exponent(&histogram),
        skew: if draws == 0 {
            0.0
        } else {
            hot as f64 / draws as f64
        },
        histogram,
    }
}

/// Least-squares slope of `ln(picks)` against `ln(rank)`, negated so a hotter
/// block scores higher.
fn zipf_exponent(ranked: &[(u32, usize)]) -> f64 {
    let n = ranked.len() as f64;
    if ranked.len() < 2 {
        return 0.0;
    }
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for (i, &(_, count)) in ranked.iter().enumerate() {
        let x = ((i + 1) as f64).ln();
        let y = (count as f64).ln();
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    let den = n * sxx - sx * sx;
    if den == 0.0 {
        0.0
    } else {
        -(n * sxy - sx * sy) / den
    }
}

/// Per-block stats (in block order) plus the pooled row for the whole trace.
pub fn analyze_routes(rows: &[RouteRow]) -> (Vec<BlockStats>, BlockStats) {
    let mut per_block: std::collections::BTreeMap<usize, std::collections::BTreeMap<u32, usize>> =
        std::collections::BTreeMap::new();
    let mut pooled: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
    for r in rows {
        let counts = per_block.entry(r.layer).or_default();
        for &(e, _) in &r.experts {
            *counts.entry(e).or_default() += 1;
            *pooled.entry(e).or_default() += 1;
        }
    }
    let blocks = per_block
        .iter()
        .map(|(&layer, counts)| stats(Some(layer), counts))
        .collect();
    (blocks, stats(None, &pooled))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<TraceRecord> {
        vec![
            TraceRecord {
                token: 1,
                layer: 7,
                experts: vec![1, 2, 3],
            },
            TraceRecord {
                token: 2,
                layer: 7,
                experts: vec![3, 4],
            },
            TraceRecord {
                token: 3,
                layer: 8,
                experts: vec![9],
            },
        ]
    }

    #[test]
    fn trace_roundtrips() {
        let p = std::env::temp_dir().join(format!("breccia-trace-{}", std::process::id()));
        write_trace(&p, &sample()).unwrap();
        assert_eq!(read_trace(&p).unwrap(), sample());
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn frequency_beats_lru_on_repeated_skew() {
        // Layer 0 hot expert 1 made large; layer 1 is resident and must not be
        // counted.
        let mut recs = Vec::new();
        for t in 0..100u64 {
            recs.push(TraceRecord {
                token: t,
                layer: 0,
                experts: vec![1, 2],
            });
        }
        recs.push(TraceRecord {
            token: 999,
            layer: 1,
            experts: vec![100],
        });
        let streamed = [true, false];
        let f = replay(&recs, &streamed, 1, Policy::Frequency).unwrap();
        assert_eq!(f.hits, 100);
        assert_eq!(f.streamed_probes, 200);
        let res = replay_all(&recs, &streamed, 1).unwrap();
        assert_eq!(res.len(), 3);
        // One slot cannot hold both of an alternating pair: only the frequency
        // policy keeps the hot one, which is the point of the test.
        assert_eq!(res[0].hits, 100);
        assert!(res[1].hits < res[0].hits, "{res:?}");
        assert!(res[2].hits < res[0].hits, "{res:?}");
    }

    #[test]
    fn lru_survives_a_large_recency_set() {
        // 10 experts rotate round-robin; budget 10 never evicts, budget 1 is
        // a miss machine.
        let recs: Vec<TraceRecord> = (0..50u64)
            .map(|t| TraceRecord {
                token: t,
                layer: 0,
                experts: (0..10).map(|i| ((t as usize + i) % 11) as u32).collect(),
            })
            .collect();
        let all_touched = replay(&recs, &[true], 11, Policy::Lru).unwrap();
        // 11 distinct experts seen once each; the window is wide enough that
        // every one of those cold loads is the last miss.
        assert_eq!(all_touched.streamed_probes, 500);
        assert_eq!(all_touched.hits, 489);
        let small = replay(&recs, &[true], 1, Policy::Lru).unwrap();
        assert_eq!(small.hit_rate(), 0.0);
    }

    #[test]
    fn window_keeps_last_k() {
        let recs = vec![
            TraceRecord {
                token: 1,
                layer: 0,
                experts: vec![5],
            },
            TraceRecord {
                token: 2,
                layer: 0,
                experts: vec![6],
            },
            TraceRecord {
                token: 3,
                layer: 0,
                experts: vec![5, 6],
            },
        ];
        let w = replay(&recs, &[true], 2, Policy::Window).unwrap();
        assert_eq!(w.streamed_probes, 4);
        // Only the two cold loads miss; token 3 finds both experts resident.
        assert_eq!(w.hits, 2);
    }

    #[test]
    fn route_sidecar_stats_on_synthetic_trace() {
        // Block 0 is hot (expert 1 half the draws), block 1 is flat.
        let mut text = String::from("# synthetic\n");
        for t in 0..4u64 {
            text.push_str(&format!(
                "{t} 0 1:0.5 {}:0.5\n2 1 {}:0.5 {}:0.5\n",
                t + 2,
                (t % 4) * 2 + 10,
                (t % 4) * 2 + 11
            ));
        }
        let p = std::env::temp_dir().join(format!("breccia-routes-{}", std::process::id()));
        std::fs::write(&p, &text).unwrap();
        let rows = read_route_sidecar(&p).unwrap();
        assert_eq!(rows.len(), 8);
        let (blocks, overall) = analyze_routes(&rows);
        assert_eq!(blocks.len(), 2);
        // 6 draws on expert 1, 2 on expert 2 across 8 rows of k=2.
        assert_eq!(overall.draws, 16);
        assert_eq!(overall.histogram[0], (1, 4));
        assert!(blocks[0].zipf > blocks[1].zipf, "{blocks:?}");
        assert!(blocks[0].skew > blocks[1].skew);
        assert_eq!(blocks[1].zipf, 0.0);
        assert!(RouteRow::parse("3 0 1").is_err());
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn span_layout_roundtrips_with_align_and_rank() {
        let ranks = vec![(0usize, 3u32), (0, 1), (0, 7), (0, 0)];
        let l = SpanLayout::build(1, 4, |_, e| Ok(1000 + e as u64), &ranks).unwrap();
        // Hottest (rank 0 → expert 3) goes first.
        let a = l.entry(0, 3).unwrap();
        assert_eq!(a.offset, align_up(SPAN_HEADER_BYTES as u64, SPAN_ALIGN));
        assert_eq!(a.blob_bytes, 1003);
        let b = l.entry(0, 1).unwrap();
        assert_eq!(b.offset, align_up(a.offset + a.blob_bytes, SPAN_ALIGN));
        // (layer, rank) pin ordering — every span must be 4096-aligned.
        for e in &l.rows[0] {
            assert_eq!(e.offset % SPAN_ALIGN, 0);
        }
        let dir = std::env::temp_dir().join(format!("breccia-span-{}", std::process::id()));
        write_spans(&dir, &l, |_, e| Ok(vec![e as u8; (1000 + e) as usize])).unwrap();
        let (r, mut file) = open_spans(&dir).unwrap();
        assert_eq!(r.rows, l.rows);
        use std::io::Read;
        let mut buf = vec![0u8; 1003];
        file.seek(std::io::SeekFrom::Start(a.offset)).unwrap();
        file.read_exact(&mut buf).unwrap();
        assert_eq!(&buf, &[3u8; 1003][..]);
        // Per-role offsets must sum exactly to the blob.
        let n = SpanLayout::role_offsets(3000, 1000, 1000, 1000).unwrap();
        assert_eq!(n, [0, 1000, 2000, 3000]);
        assert!(SpanLayout::role_offsets(3000, 999, 1000, 1000).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
