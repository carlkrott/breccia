//! Where the LM head runs — and why that is not "the last block's device".
//!
//! The head (`output.weight`, 416 MiB, plus the final mixer) is the second
//! largest single item of a decode token, and its device used to be a *consequence*
//! of the block split: `placement.device_of(n_layers - 1)`. Under any contiguous
//! split that is the last device, which on this host is the MI50 — the
//! bandwidth card, not the fast one.
//!
//! Three ways to say where it goes (`--head-device`), in increasing portability:
//!
//! * `0`, `1`, … — an **ordinal into `--devices`**, i.e. the order the caller
//!   listed the cards in. This is the one that is easy to misread: `--head-device 0`
//!   is the *first* `--devices` entry, which is the MI50 when the split is written
//!   `0000:0e:00.0,0000:03:00.0`. [`HeadSpec::parse`] therefore also accepts
//!   `pci:0000:03:00.0` and a bare PCI address, which cannot be misread at all.
//! * a **PCI address** (`--head-device 0000:03:00.0`) — resolved against the
//!   devices this model actually opened; no ordering assumption survives.
//! * `auto` (the default) — **the compute role the profile already measures**
//!   ([`breccia_profile`]'s `streaming_compute` device, tie-broken by measured
//!   fp32 GFLOPs), falling back to the device that owns the last block when no
//!   profile covers these cards.
//!
//! The fallback is deliberately the *old* behaviour: a machine without a profile
//! keeps exactly the placement it had, and the banner says which rule fired, so a
//! different answer between two machines is explainable rather than mysterious.

use anyhow::{bail, Result};

use breccia_profile::role::DeviceRole;

/// What `--head-device` asked for.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum HeadSpec {
    /// `auto` / unused: the measured compute role, else the last block's device.
    #[default]
    Auto,
    /// An ordinal into `--devices` (0 = the first entry, whatever card that is).
    Ordinal(usize),
    /// A PCI address (`0000:03:00.0`, `03:00.0`), resolved by identity.
    Pci(String),
}

impl HeadSpec {
    /// Parse the CLI value. `auto`, `compute` and the empty string mean [`HeadSpec::Auto`];
    /// a plain integer is an ordinal; anything with a `:` is a PCI address (an
    /// optional `pci:` prefix is stripped) — so `--head-device 0` and
    /// `--head-device 0000:03:00.0` are never confused for one another.
    pub fn parse(s: &str) -> Result<Self> {
        let t = s.trim();
        if t.is_empty() || t.eq_ignore_ascii_case("auto") || t.eq_ignore_ascii_case("compute") {
            return Ok(HeadSpec::Auto);
        }
        if let Ok(n) = t.parse::<usize>() {
            return Ok(HeadSpec::Ordinal(n));
        }
        let pci = t
            .strip_prefix("pci:")
            .map(str::trim)
            .unwrap_or(t)
            .to_string();
        if pci.contains(':') {
            return Ok(HeadSpec::Pci(breccia_vk::pci::normalize_address(&pci)));
        }
        bail!(
            "--head-device {t:?} is not an ordinal, a PCI address or `auto` \
             (e.g. `--head-device 1`, `--head-device 0000:03:00.0`, `--head-device auto`)"
        )
    }

    /// The CLI's value as a `String`, for the banner.
    pub fn as_str(&self) -> String {
        match self {
            HeadSpec::Auto => "auto".to_string(),
            HeadSpec::Ordinal(n) => n.to_string(),
            HeadSpec::Pci(p) => p.clone(),
        }
    }
}

/// One device of the placement, plus whatever the profile measured about it.
#[derive(Clone, Debug, Default)]
pub struct HeadCandidate {
    pub pci: String,
    pub name: String,
    /// `None` = the profile has no entry for this card.
    pub role: Option<DeviceRole>,
    /// Measured fp32 GFLOPs, the tie-break between two compute-role cards.
    pub fp32_gflops: Option<f64>,
}

/// The decision, with the sentence that explains it (printed in the load banner).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadChoice {
    pub device: usize,
    pub reason: String,
}

/// Pick the head device.
///
/// `last` is the device that owns block `n_layers - 1` — the historical answer and
/// the `auto` fallback. The function is pure (no Vulkan, no filesystem) so the
/// ordinal/PCI/role semantics can be pinned by tests, which is exactly what was
/// missing when `--head-device 0` was first reported as "the RX".
pub fn resolve(spec: &HeadSpec, candidates: &[HeadCandidate], last: usize) -> Result<HeadChoice> {
    let list = || {
        candidates
            .iter()
            .enumerate()
            .map(|(i, c)| format!("{i}={} ({})", c.pci, c.name))
            .collect::<Vec<_>>()
            .join(", ")
    };
    match spec {
        HeadSpec::Ordinal(n) => {
            if *n >= candidates.len() {
                bail!(
                    "--head-device {n} is out of range: this model opened {} device(s) — {} \
                     (the ordinal is the position in --devices, not a PCI address)",
                    candidates.len(),
                    list()
                );
            }
            Ok(HeadChoice {
                device: *n,
                reason: format!("explicit ordinal {n} into --devices"),
            })
        }
        HeadSpec::Pci(want) => {
            // Normalise here as well as in `parse`: `HeadSpec` is public, so a
            // hand-built `Pci("03:00.0")` must resolve the same as the CLI's.
            let want = breccia_vk::pci::normalize_address(want);
            let hit = candidates.iter().position(|c| {
                !c.pci.is_empty() && breccia_vk::pci::normalize_address(&c.pci) == want
            });
            match hit {
                Some(device) => Ok(HeadChoice {
                    device,
                    reason: format!("explicit PCI address {want}"),
                }),
                None => bail!("--head-device {want} matched no opened device — {}", list()),
            }
        }
        HeadSpec::Auto => {
            // 1. The profile's compute role. Prefer `streaming_compute`; among
            //    those (or, if there is none, among every measured card) the
            //    highest measured fp32 GFLOPs decides — that is "the compute
            //    device" in the profile's own vocabulary.
            let measured: Vec<(usize, &HeadCandidate)> = candidates
                .iter()
                .enumerate()
                .filter(|(_, c)| c.role.is_some())
                .collect();
            if measured.is_empty() {
                return Ok(HeadChoice {
                    device: last,
                    reason: format!(
                        "auto: no hardware profile covers these cards, so block {}'s device \
                         (the historical default)",
                        candidates.len()
                    ),
                });
            }
            let compute: Vec<(usize, &HeadCandidate)> = measured
                .iter()
                .copied()
                .filter(|(_, c)| c.role == Some(DeviceRole::StreamingCompute))
                .collect();
            let pool = if compute.is_empty() {
                measured.clone()
            } else {
                compute
            };
            let best = pool
                .iter()
                .copied()
                .max_by(|(ia, a), (ib, b)| {
                    a.fp32_gflops
                        .unwrap_or(0.0)
                        .partial_cmp(&b.fp32_gflops.unwrap_or(0.0))
                        .unwrap_or(std::cmp::Ordering::Equal)
                        // A tie must not depend on the order `--devices` was
                        // written in beyond the ordinal itself: lower index wins.
                        .then(ib.cmp(ia))
                })
                .expect("`pool` is non-empty");
            let (device, cand) = best;
            Ok(HeadChoice {
                device,
                reason: format!(
                    "auto: {} ({}, role {}) is the profile's compute device",
                    cand.pci,
                    cand.name,
                    cand.role
                        .map(|r| r.to_string())
                        .unwrap_or_else(|| "?".into())
                ),
            })
        }
    }
}

/// Refuse a head device that is not the device that owns the last block.
///
/// The crossing hand-off was **removed** (`docs/phase-4/01-forward-pass.md` §8aw):
/// with the head on the other card, 2 of 16 exclusive processes produced a wrong
/// greedy sequence, the residual copy itself was proven byte-exact (a read-back
/// compare), and the perturbation of the head's input was never localised. An
/// opt-in flag that occasionally returns wrong tokens is worse than no flag, so the
/// placement is now a consequence of `--split-blocks` / the `--devices` order: put
/// the card you want the head on **last**, or give it the last block.
///
/// `scripts/head-handoff-race.sh` is the harness that produced the 2/16 — it must
/// stay green (i.e. it must keep getting refused) for as long as this gate exists.
pub fn gate(device: usize, last: usize, candidates: &[HeadCandidate]) -> Result<()> {
    if device == last {
        return Ok(());
    }
    let name = |i: usize| {
        candidates
            .get(i)
            .map(|c| format!("{} ({})", c.pci, c.name))
            .unwrap_or_else(|| format!("device {i}"))
    };
    bail!(
        "the head can only run on the device that owns the last block ({}), not on {}: the \
         device-crossing hand-off is removed — it returned a wrong greedy sequence in 2 of 16 \
         exclusive processes and could not be made sound (docs/phase-4/01-forward-pass.md \
         §8aw). Either list `--devices` so the card you want owns the last block (the last \
         entry, since a contiguous split puts the tail there), or drop `--head-device`",
        name(last),
        name(device)
    )
}

/// Bytes the residual hand-off moves **each way** when the head lives on another
/// device: one `[hc, n_embd]` stack of f32. 40 KiB on this model.
///
/// This is the whole per-token cost of `--head-device`, so it is also the invariant
/// a regression has to break: the 416 MiB head never crosses the link per token
/// (the failure mode that was *suspected* of `--head-device 0` — it turned out to be
/// a contended window, but the guard is cheap and the suspicion was reasonable).
pub fn handoff_bytes(hc_dim: usize) -> u64 {
    (hc_dim * 4) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_crossing_head_device_is_refused() {
        let c = this_box();
        // The head may name block 47's device...
        assert!(gate(1, 1, &c).is_ok());
        assert!(gate(0, 0, &c).is_ok());
        // ...and nothing else, with a message that says what to do instead.
        let e = gate(0, 1, &c).unwrap_err();
        let msg = format!("{e:#}");
        assert!(msg.contains("removed"), "{msg}");
        assert!(msg.contains("0000:03:00.0"), "{msg}");
        assert!(
            msg.contains("--split-blocks") || msg.contains("--devices"),
            "{msg}"
        );
    }

    #[test]
    fn the_handoff_is_one_residual_stack_and_cannot_be_the_head() {
        let model = handoff_bytes(10_240); // hc 4 x n_embd 2560
        assert_eq!(model, 40 * 1024);
        // Three orders of magnitude below the 416 MiB head: a per-token weight
        // upload can never hide inside this path's budget.
        assert!(model * 100 < 416 << 20);
    }

    fn cand(pci: &str, name: &str, role: Option<DeviceRole>, gflops: Option<f64>) -> HeadCandidate {
        HeadCandidate {
            pci: pci.to_string(),
            name: name.to_string(),
            role,
            fp32_gflops: gflops,
        }
    }

    /// The device order the *user* wrote, not the cards' identity.
    fn this_box() -> Vec<HeadCandidate> {
        vec![
            cand(
                "0000:0e:00.0",
                "Radeon VII",
                Some(DeviceRole::StaticResident),
                Some(13.0),
            ),
            cand(
                "0000:03:00.0",
                "RX 6950 XT",
                Some(DeviceRole::StreamingCompute),
                Some(23.0),
            ),
        ]
    }

    #[test]
    fn the_ordinal_is_the_devices_order_and_says_so() {
        // The trap this test exists for: `--devices 0e,03` makes 0 the MI50.
        let c = this_box();
        let d = resolve(&HeadSpec::Ordinal(0), &c, 1).unwrap();
        assert_eq!(d.device, 0);
        assert!(d.reason.contains("ordinal"));
        assert_eq!(c[d.device].pci, "0000:0e:00.0");
    }

    #[test]
    fn a_pci_address_does_not_depend_on_the_devices_order() {
        let c = this_box();
        for spec in [
            HeadSpec::Pci("0000:03:00.0".into()),
            HeadSpec::Pci("03:00.0".into()),
        ] {
            assert_eq!(resolve(&spec, &c, 0).unwrap().device, 1);
        }
        // …and the same answer with the list written the other way round.
        let flipped: Vec<HeadCandidate> = c.into_iter().rev().collect();
        assert_eq!(
            resolve(&HeadSpec::Pci("0000:03:00.0".into()), &flipped, 1)
                .unwrap()
                .device,
            0
        );
    }

    #[test]
    fn auto_picks_the_profiles_compute_role() {
        // Both orders of `--devices` must give the same *card*.
        assert_eq!(resolve(&HeadSpec::Auto, &this_box(), 0).unwrap().device, 1);
        let flipped: Vec<HeadCandidate> = this_box().into_iter().rev().collect();
        assert_eq!(resolve(&HeadSpec::Auto, &flipped, 1).unwrap().device, 0);
    }

    #[test]
    fn auto_prefers_the_higher_measured_flops_between_compute_roles() {
        let c = vec![
            cand(
                "0000:03:00.0",
                "a",
                Some(DeviceRole::StreamingCompute),
                Some(11.0),
            ),
            cand(
                "0000:04:00.0",
                "b",
                Some(DeviceRole::StreamingCompute),
                Some(22.0),
            ),
        ];
        assert_eq!(resolve(&HeadSpec::Auto, &c, 0).unwrap().device, 1);
    }

    #[test]
    fn auto_falls_back_to_the_last_blocks_device_without_a_profile() {
        let c = vec![
            cand("0000:0e:00.0", "Radeon VII", None, None),
            cand("0000:03:00.0", "RX 6950 XT", None, None),
        ];
        let d = resolve(&HeadSpec::Auto, &c, 1).unwrap();
        assert_eq!(d.device, 1);
        assert!(d.reason.contains("no hardware profile"), "{}", d.reason);
        // …and to the last block even when the profile covers only *one* of them.
        let c = vec![
            cand("0000:0e:00.0", "Radeon VII", None, None),
            cand(
                "0000:03:00.0",
                "RX 6950 XT",
                Some(DeviceRole::StreamingCompute),
                None,
            ),
        ];
        assert_eq!(resolve(&HeadSpec::Auto, &c, 1).unwrap().device, 1);
    }

    #[test]
    fn an_out_of_range_ordinal_lists_the_devices() {
        let e = resolve(&HeadSpec::Ordinal(2), &this_box(), 1).unwrap_err();
        let msg = format!("{e:#}");
        assert!(msg.contains("out of range"), "{msg}");
        assert!(msg.contains("0000:0e:00.0"), "{msg}");
        assert!(msg.contains("0000:03:00.0"), "{msg}");
    }

    #[test]
    fn an_unknown_pci_address_is_an_error_not_a_fallback() {
        let e = resolve(&HeadSpec::Pci("0000:99:00.0".into()), &this_box(), 1).unwrap_err();
        assert!(format!("{e:#}").contains("matched no opened device"));
    }

    #[test]
    fn the_cli_value_parses_without_ambiguity() {
        assert_eq!(HeadSpec::parse("").unwrap(), HeadSpec::Auto);
        assert_eq!(HeadSpec::parse("auto").unwrap(), HeadSpec::Auto);
        assert_eq!(HeadSpec::parse("compute").unwrap(), HeadSpec::Auto);
        assert_eq!(HeadSpec::parse("0").unwrap(), HeadSpec::Ordinal(0));
        assert_eq!(HeadSpec::parse(" 1 ").unwrap(), HeadSpec::Ordinal(1));
        assert_eq!(
            HeadSpec::parse("0000:03:00.0").unwrap(),
            HeadSpec::Pci("0000:03:00.0".into())
        );
        assert_eq!(
            HeadSpec::parse("pci:03:00.0").unwrap(),
            HeadSpec::Pci("0000:03:00.0".into())
        );
        assert!(HeadSpec::parse("rx").is_err());
        // 0 and a PCI address can never be confused for each other.
        assert_ne!(
            HeadSpec::parse("0").unwrap(),
            HeadSpec::parse("03:00.0").unwrap()
        );
    }
}
