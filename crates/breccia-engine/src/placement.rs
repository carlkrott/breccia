//! Placement and the buffer plan: which block lives on which GPU, and how many
//! bytes that is before a single byte is uploaded.
//!
//! The plan follows `docs/specs/fork-flags-semantics.md`: the expert default is
//! **host** (`-cmoe`), so nothing in [`Placement`] budgets for expert weights on
//! a device. What a device holds is the *dense* set — attention/GDN projections,
//! hyper-connection projections, the router, the shared expert, the indexer —
//! plus, on the last device, the output head and the final mixer.
//!
//! `-ot`-style per-block expert overrides are **not** implemented here (M3):
//! the shipped plan is host-only experts, and the report says so.

use anyhow::{bail, Result};

/// How a device is addressed when it is opened.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceSpec {
    /// PCI address (`0000:03:00.0`), runtime index (`1`), or `None` for the
    /// first discrete GPU.
    pub selector: Option<String>,
    /// Human label for the banner (`"0000:03:00.0"` or `"first discrete"`).
    pub label: String,
}

impl DeviceSpec {
    pub fn pci(addr: &str) -> Self {
        DeviceSpec {
            selector: Some(addr.to_string()),
            label: addr.to_string(),
        }
    }
    pub fn first_discrete() -> Self {
        DeviceSpec {
            selector: None,
            label: "first discrete".to_string(),
        }
    }
    /// Parse a `--devices` item: a PCI address, a runtime index, or `default`.
    pub fn parse(s: &str) -> Self {
        let t = s.trim();
        if t.is_empty() || t.eq_ignore_ascii_case("default") || t.eq_ignore_ascii_case("auto") {
            Self::first_discrete()
        } else {
            Self {
                selector: Some(t.to_string()),
                label: t.to_string(),
            }
        }
    }
}

/// One contiguous run of blocks assigned to one device.
#[derive(Clone, Debug)]
pub struct LayerRun {
    pub device: usize,
    /// `[first, last)` block indices.
    pub blocks: (usize, usize),
    /// Dense bytes of those blocks.
    pub bytes: u64,
}

/// The whole placement: devices, the per-block assignment, the bytes per device.
#[derive(Clone, Debug)]
pub struct Placement {
    pub devices: Vec<DeviceSpec>,
    /// `layer_device[block]` → device ordinal.
    pub layer_device: Vec<usize>,
    /// Planned dense bytes per device (weights only).
    pub bytes: Vec<u64>,
    /// The contiguous runs, in block order (what the banner prints).
    pub runs: Vec<LayerRun>,
}

impl Placement {
    /// Explicit plan: `counts[i]` blocks on device `i`, in order. The last
    /// device also carries the tail (head + final mixer).
    ///
    /// This is what `--split-blocks` drives: the balanced split assumes the two
    /// cards are equally good at the *graph*, and on this host they are not (the
    /// MI50's synchronous-dispatch round trip is ~36 µs, the RX's ~64 µs), so the
    /// balance is a measurement, not an arithmetic mean of the dense bytes.
    pub fn explicit(
        devices: Vec<DeviceSpec>,
        layer_bytes: &[u64],
        tail_bytes: u64,
        counts: &[usize],
    ) -> Result<Self> {
        let n = layer_bytes.len();
        if devices.len() != counts.len() {
            bail!(
                "{} devices but {} block counts",
                devices.len(),
                counts.len()
            );
        }
        if counts.iter().sum::<usize>() != n {
            bail!(
                "block counts {counts:?} sum to {}, but the model has {n} blocks",
                counts.iter().sum::<usize>()
            );
        }
        let mut layer_device = vec![0usize; n];
        let mut bytes = vec![0u64; devices.len()];
        let mut runs = Vec::new();
        let mut at = 0usize;
        for (dev, &c) in counts.iter().enumerate() {
            let end = at + c;
            for (b, slot) in layer_device.iter_mut().enumerate().take(end).skip(at) {
                *slot = dev;
                bytes[dev] += layer_bytes[b];
            }
            let mut run_bytes = bytes[dev];
            if dev + 1 == devices.len() {
                run_bytes += tail_bytes;
                bytes[dev] = run_bytes;
            }
            runs.push(LayerRun {
                device: dev,
                blocks: (at, end),
                bytes: run_bytes,
            });
            at = end;
        }
        Ok(Placement {
            devices,
            layer_device,
            bytes,
            runs,
        })
    }

    /// Single-device plan: everything on device 0.
    pub fn single(dev: DeviceSpec, layer_bytes: &[u64]) -> Self {
        let total: u64 = layer_bytes.iter().sum();
        let n = layer_bytes.len();
        Placement {
            devices: vec![dev],
            layer_device: vec![0; n],
            bytes: vec![total],
            runs: vec![LayerRun {
                device: 0,
                blocks: (0, n),
                bytes: total,
            }],
        }
    }

    /// Balance `layer_bytes` (plus `tail_bytes`, which only the **last** device
    /// carries) over `devices` as contiguous block runs.
    ///
    /// Contiguity is a graph requirement, not an aesthetic one: moving the
    /// residual stack between GPUs is a PCIe round trip, so the forward pass
    /// wants exactly one hand-off per token per boundary. The MI50's PCIe link
    /// is gen3 ×4 (2.9 GB/s measured), so a hand-off must stay a handful of
    /// kilobytes — one `[hc, n_embd]` stack is 40 KiB.
    pub fn balanced(devices: Vec<DeviceSpec>, layer_bytes: &[u64], tail_bytes: u64) -> Self {
        let n = layer_bytes.len();
        let d = devices.len();
        if d <= 1 || n == 0 {
            let mut p = Self::single(
                devices
                    .into_iter()
                    .next()
                    .unwrap_or_else(DeviceSpec::first_discrete),
                layer_bytes,
            );
            p.bytes[0] += tail_bytes;
            return p;
        }
        // Greedy contiguous split: walk the blocks and cut when the running
        // device total plus this block's bytes would exceed the fair share of
        // what is left. `tail_bytes` is folded into the last device's share.
        let remain_total = |from: usize, last: bool| -> u64 {
            let s: u64 = layer_bytes[from..].iter().sum();
            if last {
                s + tail_bytes
            } else {
                s
            }
        };
        let mut layer_device = vec![0usize; n];
        let mut bytes = vec![0u64; d];
        let mut runs = Vec::new();
        let mut start = 0usize;
        for (dev, dev_bytes) in bytes.iter_mut().enumerate() {
            let last = dev + 1 == d;
            let remaining = remain_total(start, true);
            let share = remaining / (d - dev) as u64;
            let mut acc = 0u64;
            let mut end = start;
            while end < n {
                let b = layer_bytes[end];
                let extra = if last && end + 1 == n { tail_bytes } else { 0 };
                // Keep at least one block per device, and never overshoot the
                // share by more than half a block.
                if end > start && acc + b + extra > share + b / 2 {
                    break;
                }
                acc += b + extra;
                end += 1;
            }
            if end == start {
                end = (start + 1).min(n);
                acc = layer_bytes[start] + if last && end == n { tail_bytes } else { 0 };
            }
            for slot in layer_device.iter_mut().take(end).skip(start) {
                *slot = dev;
            }
            *dev_bytes = acc;
            runs.push(LayerRun {
                device: dev,
                blocks: (start, end),
                bytes: acc,
            });
            start = end;
            if start >= n {
                break;
            }
        }
        // Any blocks left over (a device that ended early) join the last run.
        if start < n {
            if let Some(last_run) = runs.last_mut() {
                for slot in layer_device.iter_mut().take(n).skip(start) {
                    *slot = last_run.device;
                }
                let extra: u64 = layer_bytes[start..].iter().sum::<u64>()
                    + if last_run.device + 1 == d {
                        tail_bytes
                    } else {
                        0
                    };
                last_run.bytes += extra;
                last_run.blocks.1 = n;
                bytes[last_run.device] += extra;
            }
        }
        // `bytes` for devices with no run must still count the tail on the last.
        if d > 1 && runs.len() < d {
            bytes[d - 1] += tail_bytes;
        }
        Placement {
            devices,
            layer_device,
            bytes,
            runs,
        }
    }

    /// Fail the plan if any device's dense bytes exceed its capacity.
    pub fn check_capacity(&self, capacity: &[u64]) -> Result<()> {
        self.check_capacity_of(&self.bytes, capacity)
    }

    /// [`Placement::check_capacity`] against an explicit per-device byte plan — used
    /// when the tail does not ride the last block's device (`--head-device`).
    pub fn check_capacity_of(&self, planned: &[u64], capacity: &[u64]) -> Result<()> {
        for (i, (&want, &have)) in planned.iter().zip(capacity.iter()).enumerate() {
            if want == 0 {
                continue;
            }
            // Leave 8 % + 256 MiB for state, KV, activations and the driver.
            let budget = have.saturating_sub(have / 12 + (256 << 20));
            if want > budget {
                bail!(
                    "device {i} ({}) would need {} MiB of dense weights but only {} MiB of its \
                     {} MiB is usable after reserving state/KV/activations",
                    self.devices[i].label,
                    want >> 20,
                    budget >> 20,
                    have >> 20
                );
            }
        }
        Ok(())
    }

    /// True when the block runs are contiguous per device and cover every block.
    pub fn is_contiguous(&self) -> bool {
        let mut expect = 0usize;
        for run in &self.runs {
            if run.blocks.0 != expect {
                return false;
            }
            expect = run.blocks.1;
        }
        expect == self.layer_device.len()
    }

    /// The device a block runs on.
    pub fn device_of(&self, block: usize) -> usize {
        self.layer_device[block]
    }
}

/// Estimated per-token state/KV bytes a device must also hold.
pub fn state_bytes(
    cfg: &crate::config::ModelConfig,
    layer_device: &[usize],
    dev: usize,
    capacity_cells: usize,
) -> u64 {
    let mut total = 0u64;
    for (block, &d) in layer_device.iter().enumerate() {
        if d != dev {
            continue;
        }
        match cfg.layer_kind(block) {
            crate::config::LayerKind::Gdn => {
                // 3 MiB recurrent state + 120 KiB conv history.
                total += (128 * 48 * 128 * 4 + 3 * 10240 * 4) as u64;
            }
            crate::config::LayerKind::Qsa => {
                // f16 K/V per cell (2048 B) + pooled indexer key (~128 B/cell).
                total += capacity_cells as u64 * (2048 + 132);
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn specs(n: usize) -> Vec<DeviceSpec> {
        (0..n)
            .map(|i| DeviceSpec::pci(&format!("0000:0{i}:00.0")))
            .collect()
    }

    #[test]
    fn single_device_gets_everything() {
        let lb = vec![100u64; 48];
        let p = Placement::single(DeviceSpec::first_discrete(), &lb);
        assert_eq!(p.layer_device.len(), 48);
        assert!(p.layer_device.iter().all(|&d| d == 0));
        assert_eq!(p.bytes[0], 4800);
    }

    #[test]
    fn two_devices_split_contiguously_and_cover_every_block() {
        let lb = vec![100u64; 48];
        let p = Placement::balanced(specs(2), &lb, 0);
        assert!(p.is_contiguous(), "runs must tile 0..48");
        assert_eq!(p.layer_device.len(), 48);
        // 24 blocks each, since every block costs the same.
        assert_eq!(p.runs.len(), 2);
        assert_eq!(p.runs[0].blocks, (0, 24));
        assert_eq!(p.runs[1].blocks, (24, 48));
        assert_eq!(p.bytes, vec![2400, 2400]);
    }

    #[test]
    fn the_tail_lands_on_the_last_device() {
        let lb = vec![100u64; 8];
        let p = Placement::balanced(specs(2), &lb, 640);
        assert!(p.is_contiguous());
        assert_eq!(p.bytes[0] + p.bytes[1], 8 * 100 + 640);
        assert_eq!(p.layer_device[7], 1);
    }

    #[test]
    fn unbalanced_blocks_still_produce_a_contiguous_plan() {
        let mut lb = vec![100u64; 8];
        lb[0] = 5000;
        let p = Placement::balanced(specs(2), &lb, 0);
        assert!(p.is_contiguous());
        assert_eq!(p.layer_device[0], 0);
        assert_eq!(*p.layer_device.last().unwrap(), 1);
    }

    #[test]
    fn capacity_check_rejects_an_impossible_plan() {
        let lb = vec![3u64 << 30; 8]; // 24 GiB of dense
        let p = Placement::single(DeviceSpec::first_discrete(), &lb);
        assert!(p.check_capacity(&[16u64 << 30]).is_err());
        assert!(p.check_capacity(&[64u64 << 30]).is_ok());
    }
}
