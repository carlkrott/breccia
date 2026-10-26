//! One-off diagnostic (Phase 1 report): recompute the fingerprint from the
//! *stored* identity fields of both hw_profile.json files and compare with the
//! fingerprint each file claims.

use breccia_profile::fingerprint::{fingerprint, FingerprintInput};
use breccia_profile::schema::HwProfile;

fn input_of(p: &HwProfile) -> FingerprintInput {
    let ids: Vec<_> = p.devices.iter().map(|d| d.identity.clone()).collect();
    FingerprintInput::from_identities(&ids, &p.host)
}

#[test]
fn stored_fingerprints_match_stored_inputs() {
    let quick: HwProfile =
        serde_json::from_str(&std::fs::read_to_string("/tmp/hw_profile_test.json").unwrap())
            .unwrap();
    let full: HwProfile = serde_json::from_str(
        &std::fs::read_to_string(
            std::env::var("HOME").unwrap() + "/.config/breccia/hw_profile.json",
        )
        .unwrap(),
    )
    .unwrap();

    let dq = fingerprint(&input_of(&quick));
    let df = fingerprint(&input_of(&full));
    println!("stored claim  : quick={} full={}", quick.fingerprint, full.fingerprint);
    println!("recomputed    : quick={} full={}", dq, df);
    println!("inputs equal  : {}", input_of(&quick) == input_of(&full));
    println!("live          : {}", breccia_profile::current_fingerprint().unwrap().fingerprint);

    assert_eq!(dq, quick.fingerprint, "quick file: stored fp != digest of its own inputs");
    assert_eq!(df, full.fingerprint, "full file: stored fp != digest of its own inputs");
}
