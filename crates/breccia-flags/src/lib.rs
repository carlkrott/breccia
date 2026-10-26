//! llama.cpp/HIP-compatible launch-flag parser for Breccia.
//!
//! The Phase-1 decision (see `docs/plan/BRECCIA-PLAN.md`) makes **Vulkan the
//! primary backend while keeping the existing llama.cpp/HIP flag surface**. This
//! crate is the seam that makes that true: it parses the production
//! `qwen-dual-serve.sh` argument vector — including `ROCm0`/`ROCm1` device names,
//! fork-only flags such as `-cmoe`, `-kvu`, `--fit` — into a typed
//! [`LaunchPlan`], and resolves those alias families against the selected
//! [`Backend`].
//!
//! Design rules:
//!
//! - **No silent drops.** Unknown flags are errors ([`FlagError::UnknownFlag`])
//!   naming the flag and its 0-based argv position.
//! - **Typed values.** `-ctk q4_0` becomes [`KvCacheType::Q4_0`], `--port 70000`
//!   fails at parse time, not at bind time.
//! - **Alias ≠ backend.** `ROCm1` on the default Vulkan backend is accepted and
//!   resolves to Vulkan device 1 with a structured warning — never an error.
//! - **Round-trips.** [`LaunchPlan::render_args`] re-emits an equivalent argv, so
//!   a plan can be logged, diffed, or handed to the reference binary.
//!
//! ```
//! use breccia_flags::{parse_str, Backend, BackendHint};
//!
//! let plan = parse_str("-m model.gguf --device ROCm0,ROCm1 -ngl 99").unwrap();
//! assert_eq!(plan.backend, Backend::Vulkan); // default
//! assert_eq!(plan.devices[0].backend_hint, BackendHint::Hip);
//!
//! // HIP flags, Vulkan hardware.
//! let resolved = plan.resolve_devices();
//! assert_eq!(resolved[0].backend, BackendHint::Vulkan);
//! assert_eq!(resolved[1].index, 1);
//! assert_eq!(plan.warnings().len(), 2); // one per remapped alias
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod device;
pub mod error;
pub mod overrides;
pub mod parse;
pub mod plan;
pub mod value;

pub use device::{
    Backend, BackendHint, DeviceRef, DeviceResolution, LaunchWarning, ResolvedDevice, WarningCode,
};
pub use error::FlagError;
pub use overrides::{split_override_entry, split_top_level, CompiledOverride};
pub use parse::{parse_llama_server_args, parse_str, tokenize};
pub use plan::LaunchPlan;
pub use value::{KvCacheType, ReasoningMode, Switch};
