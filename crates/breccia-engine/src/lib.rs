//! `breccia-engine` — the forward-pass graph (Phase 4 / M2): placement, the
//! 48-block layer graph, the host expert lane, and sampling.
//!
//! # What this crate is
//!
//! Orchestration and nothing else. Every number is computed by
//! `breccia-kernels` — GEMV/dequant, the ops (`docs/phase-3/04`), GDN
//! (`docs/phase-3/06`), QSA (`docs/phase-3/06`) and the host expert lane
//! (`docs/phase-3/08`) — and this crate only decides *what runs, on which device,
//! in which order*, plus the two things the kernels do not own: the GGUF/dense
//! placement plan ([`placement`]) and the per-token transfer discipline
//! ([`xfer`]).
//!
//! # The shape of a token
//!
//! [`graph::Session::generate`] feeds the prompt one token at a time, then runs
//! the decode loop. One token is
//!
//! ```text
//! 1 embedding gather · 48 × (2 HC_READ, 2 HC_WRITE, 1 mixer, router + 10 experts
//! + shared expert) · 1 final HC_READ · 1 head GEMV · 1 argmax
//! ```
//!
//! # Deviations
//!
//! The ones a reader of the numbers has to know about are listed, with their
//! measured impact, in `docs/phase-4/01-forward-pass.md` §"Known deviations".
//! The two structural ones live here: the PLE block runs on the host
//! ([`ple`]) because it has no kernel yet, and the expert lane is host-resident
//! (`-cmoe`) with no `-ot` per-block GPU overrides, because that is M3 work.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! use breccia_engine::{graph::{Session, SessionCfg}, model::{LoadOptions, Model}, sampler::Sampling};
//!
//! let model = Model::load("/srv/strata-data/models/IQ3_XXS/*-0000?-of-00002.gguf",
//!                         &LoadOptions { devices: vec!["0000:0e:00.0".into()], ..Default::default() })?;
//! let mut s = Session::new(&model, &SessionCfg::default())?;
//! s.sampling = Sampling::greedy();
//! let gen = s.generate(&[1, 2, 3], 8, None)?;
//! println!("{:?} at {:.2} tok/s", gen.tokens, gen.tokens_per_s());
//! # Ok(())
//! # }
//! ```

pub mod config;
pub mod graph;
pub mod model;
pub mod multi;
pub mod placement;
pub mod ple;
pub mod report;
pub mod sampler;
pub mod stats;
pub mod xfer;

pub use config::{LayerKind, ModelConfig};
pub use graph::{
    prefill_cap, prefill_plan, Generation, GpuStageRow, Session, SessionCfg, Timers, STAGES,
};

pub use model::{LoadOptions, LoadStats, Model};
pub use multi::{Multi, MultiGeneration, StreamReport};
pub use placement::{DeviceSpec, Placement};
pub use report::{LogitSummary, RunReport, StageRow, TokenReport, TOKEN_SCHEMA_VERSION};
pub use sampler::{Sampler, Sampling};
pub use stats::{Checksum, LayerTrace};

/// The parity baseline this milestone is judged against: 48.1 ms/token, i.e.
/// 20.79 tok/s decode (`docs/phase-0/01`, `docs/phase-3/05` §1).
pub const PARITY_TOKENS_PER_S: f64 = 20.79;

/// The parity gate the design doc sets for numerics (`docs/phase-3/05` §5):
/// top-1 agreement on at least this share of steps.
pub const PARITY_TOP1_BAR: f64 = 0.95;
