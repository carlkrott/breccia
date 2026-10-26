//! breccia CLI entry (`breccia profile|serve|bench`).
//!
//! Phase 1 wires `breccia profile` (hardware discovery + calibration →
//! hw_profile.json) on top of `breccia-vk` + `breccia-profile`, and parses the
//! llama.cpp/HIP-compatible launch surface via `breccia-flags` so existing
//! service invocations (e.g. qwen-dual-serve.sh) can launch breccia unchanged.

fn main() {
    println!("breccia {} (phase-1 scaffold)", env!("CARGO_PKG_VERSION"));
}
