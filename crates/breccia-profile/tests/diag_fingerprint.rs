//! Self-consistency check for the stored hardware profiles (Phase 1 report).
//!
//! Recomputes the fingerprint from each `hw_profile.json`'s *stored* identity
//! fields and asserts it equals the fingerprint that file claims — i.e. the
//! cached fast path can never be invalidated by a hashing change alone.
//!
//! Skips (passes without checking) when the profile files are absent, so the
//! workspace test suite stays green on a clean machine.

use breccia_profile::fingerprint::{fingerprint, FingerprintInput};
use breccia_profile::schema::HwProfile;

fn input_of(p: &HwProfile) -> FingerprintInput {
    let ids: Vec<_> = p.devices.iter().map(|d| d.identity.clone()).collect();
    FingerprintInput::from_identities(&ids, &p.host)
}

fn load(path: &str) -> Option<HwProfile> {
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

#[test]
fn stored_fingerprints_match_stored_inputs() {
    let quick_path = "/tmp/hw_profile_test.json";
    let full_path = format!(
        "{}/.config/breccia/hw_profile.json",
        std::env::var("HOME").unwrap_or_default()
    );
    let (Some(quick), Some(full)) = (load(quick_path), load(full_path.as_str())) else {
        eprintln!("skipped: no stored hw_profile.json files on this machine");
        return;
    };

    let dq = fingerprint(&input_of(&quick));
    let df = fingerprint(&input_of(&full));
    println!(
        "stored claim  : quick={} full={}",
        quick.fingerprint, full.fingerprint
    );
    println!("recomputed    : quick={} full={}", dq, df);
    println!("inputs equal  : {}", input_of(&quick) == input_of(&full));
    println!(
        "live          : {}",
        breccia_profile::current_fingerprint()
            .expect("live fingerprint")
            .fingerprint
    );

    assert_eq!(
        dq, quick.fingerprint,
        "quick profile: stored fingerprint != digest of its own stored inputs"
    );
    assert_eq!(
        df, full.fingerprint,
        "full profile: stored fingerprint != digest of its own stored inputs"
    );
    // Both files describe the same hardware → they must agree with each other.
    assert_eq!(
        quick.fingerprint, full.fingerprint,
        "quick and full profiles disagree on the fingerprint"
    );
}
