//! The expert arena as a **rule**, not a block list.
//!
//! Until this module existed the resident-expert set was a hand-tuned list
//! (`--gpu-experts "0..15,28..41"`), chosen by trial on one machine. It works, and
//! it is host-specific in two ways that both bite:
//!
//! 1. it assumes which card is bigger / has the display attached, and
//! 2. it assumes that both cards had the same free VRAM the day it was tuned.
//!
//! On this box the hand-tuned 30-block set leaves **2.6 GiB of the RX unused**
//! while the MI50 is packed to 265 MiB — the list was cut by hand, not by the
//! budget.
//!
//! # The rule
//!
//! * **Probe** the free VRAM of every device the model spans, *after* the dense
//!   weights and the head are uploaded (they are part of the model, not the arena).
//!   The source is `VK_EXT_memory_budget`'s `heapBudget - heapUsage` — the driver's
//!   own statement of what this process may still allocate. When the extension is
//!   absent the fallback is amdgpu's `mem_info_vram_{total,used}` from sysfs, which
//!   is system-wide, so it is *reported as the source* rather than silently
//!   treated the same.
//! * **Reserve** an explicit amount: a *display reserve* on any device that
//!   currently drives a connected output (detected from sysfs, never hard-coded —
//!   on this machine the connected outputs are on `0000:03:00.0`, not the MI50 the
//!   plan text assumes), and a smaller *headroom* everywhere else (the session's
//!   own scratch is allocated after the weights, ~120 MiB measured).
//! * **Select** blocks by measured routed-expert bytes: every resident block saves
//!   `n_expert_used / n_expert` of its arena per token, so the per-byte efficiency
//!   is the same for every block and the optimal set is the one that holds the
//!   **most bytes** inside the budget. First-fit-decreasing (largest arena first,
//!   lowest block index on a tie) reaches that in one pass and re-tries the smaller
//!   blocks afterwards, so a single oversized block cannot strand the rest.
//! * **Refuse or degrade**: a block that does not fit is simply not resident (it
//!   stays on the host lane, which is the default path), and an allocation that
//!   fails despite the arithmetic drops that block instead of aborting the load.
//!   `VK_ERROR_OUT_OF_DEVICE_MEMORY` is never the caller's problem.

use serde::{Deserialize, Serialize};

/// Where an explicit block list comes from (`--gpu-experts`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum ArenaSpec {
    /// No resident experts: every routed expert streams through the host lane.
    #[default]
    Off,
    /// An explicit list — the hand-tuned override, kept working verbatim.
    List(Vec<usize>),
    /// The measured-free-VRAM rule.
    Auto(ArenaOpts),
}

impl ArenaSpec {
    /// Parse the CLI value: `auto` or a comma-separated block list.
    pub fn parse(s: &str) -> Self {
        let t = s.trim();
        if t.eq_ignore_ascii_case("auto") {
            return ArenaSpec::Auto(ArenaOpts::default());
        }
        let blocks: Vec<usize> = t
            .split(',')
            .filter_map(|x| x.trim().parse::<usize>().ok())
            .collect();
        if blocks.is_empty() {
            ArenaSpec::Off
        } else {
            ArenaSpec::List(blocks)
        }
    }
}

/// The two reserves the rule keeps out of the measured free memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaOpts {
    /// Bytes left free on a device that currently drives a display.
    pub display_reserve: u64,
    /// Bytes left free on a headless device (the session's scratch and allocator
    /// slack; measured at ~120 MiB on this model).
    pub headroom: u64,
}

impl Default for ArenaOpts {
    fn default() -> Self {
        ArenaOpts {
            display_reserve: 768 << 20,
            headroom: 256 << 20,
        }
    }
}

/// One candidate: a block's whole routed-expert arena and the device that owns it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaBlock {
    pub block: usize,
    pub device: usize,
    /// `gate + up + down` uploaded for all 512 experts (912 MiB is typical here).
    pub bytes: u64,
}

/// Where a device's free-VRAM number came from — printed, so a different number
/// between two runs can be traced to a different source rather than to noise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetSource {
    /// `VK_EXT_memory_budget`: `heapBudget - heapUsage`, this process only.
    MemoryBudget,
    /// amdgpu sysfs `mem_info_vram_total - mem_info_vram_used`: system-wide.
    DriverSysfs,
}

impl BudgetSource {
    pub fn as_str(self) -> &'static str {
        match self {
            BudgetSource::MemoryBudget => "VK_EXT_memory_budget",
            BudgetSource::DriverSysfs => "sysfs mem_info_vram (system-wide)",
        }
    }
}

/// What one device may spend on the arena.
#[derive(Clone, Debug)]
pub struct DeviceBudget {
    pub device: usize,
    pub label: String,
    /// The device currently drives a connected display (sysfs).
    pub display: bool,
    /// Free bytes before the reserve.
    pub free: u64,
    /// The reserve subtracted from `free`.
    pub reserve: u64,
    pub source: BudgetSource,
}

impl DeviceBudget {
    /// Bytes the arena may use on this device.
    pub fn budget(&self) -> u64 {
        self.free.saturating_sub(self.reserve)
    }
}

/// Why a block is not resident.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Skip {
    /// Its arena alone is larger than the device's whole budget.
    TooBig,
    /// The device's budget was already spent on larger blocks.
    BudgetSpent,
    /// The device is not in the plan.
    NoDevice,
    /// The upload failed even though the arithmetic said it fit (degrade path).
    UploadFailed,
}

impl Skip {
    pub fn as_str(self) -> &'static str {
        match self {
            Skip::TooBig => "arena exceeds the whole device budget",
            Skip::BudgetSpent => "device budget spent",
            Skip::NoDevice => "device not in the plan",
            Skip::UploadFailed => "vkAllocateMemory failed at upload",
        }
    }
}

/// The rule's answer.
#[derive(Clone, Debug, Default)]
pub struct ArenaChoice {
    /// Blocks that become resident, in block order.
    pub blocks: Vec<ArenaBlock>,
    /// Per device: budget and consumption.
    pub devices: Vec<DeviceSpend>,
    /// Blocks left on the host lane, with the reason.
    pub skipped: Vec<(ArenaBlock, Skip)>,
}

impl ArenaChoice {
    pub fn bytes(&self) -> u64 {
        self.blocks.iter().map(|b| b.bytes).sum()
    }

    /// Blocks whose experts stay on the host lane (`None` = all of them, i.e. the
    /// arena is off).
    pub fn host_blocks(&self, n_layers: usize) -> usize {
        n_layers.saturating_sub(self.blocks.len())
    }
}

/// One device's share of the decision, for the banner and the JSON report.
#[derive(Clone, Debug)]
pub struct DeviceSpend {
    pub budget: DeviceBudget,
    pub chosen: usize,
    pub bytes: u64,
}

impl DeviceSpend {
    /// Bytes left in the budget after the rule's choice.
    pub fn left(&self) -> u64 {
        self.budget.budget().saturating_sub(self.bytes)
    }
}

/// Pick the resident set: **first-fit decreasing** per device, largest arena first.
///
/// The metric is deliberately *bytes held*, not block count: a resident block
/// removes `n_expert_used / n_expert` (10/512 here) of its own arena from the
/// host lane every token, so two blocks of the same byte size are worth the same
/// however their quant types differ. Ties break on the lower block index, which is
/// what makes the rule reproduce the hand-tuned list when the arenas are uniform.
///
/// One ordered pass visits every block largest-first and **skips** (rather than
/// stops at) a block that no longer fits, so an oversized arena cannot strand the
/// smaller blocks behind it.
pub fn select(blocks: &[ArenaBlock], budgets: &[DeviceBudget]) -> ArenaChoice {
    let mut out = ArenaChoice::default();
    for b in budgets {
        out.devices.push(DeviceSpend {
            budget: b.clone(),
            chosen: 0,
            bytes: 0,
        });
    }
    let mut order: Vec<usize> = (0..blocks.len()).collect();
    order.sort_by(|&a, &b| {
        blocks[b]
            .bytes
            .cmp(&blocks[a].bytes)
            .then(blocks[a].block.cmp(&blocks[b].block))
    });
    // `None` = still a candidate; `Some(why)` = rejected before/at the pass.
    let mut rejected: Vec<Option<Skip>> = vec![None; blocks.len()];
    for &i in &order {
        let b = blocks[i];
        match out.devices.iter().find(|s| s.budget.device == b.device) {
            None => rejected[i] = Some(Skip::NoDevice),
            Some(s) if b.bytes > s.budget.budget() => rejected[i] = Some(Skip::TooBig),
            Some(_) => {}
        }
    }
    for &i in &order {
        if rejected[i].is_some() {
            continue;
        }
        let b = blocks[i];
        let spend = out
            .devices
            .iter_mut()
            .find(|s| s.budget.device == b.device)
            .expect("checked above");
        if spend.bytes + b.bytes <= spend.budget.budget() {
            spend.bytes += b.bytes;
            spend.chosen += 1;
            out.blocks.push(b);
        } else {
            rejected[i] = Some(Skip::BudgetSpent);
        }
    }
    out.blocks.sort_by_key(|b| b.block);
    for &i in &order {
        if let Some(why) = rejected[i] {
            out.skipped.push((blocks[i], why));
        }
    }
    out
}

/// Mark a chosen block as failed (the degrade path) and give its bytes back.
pub fn drop_block(choice: &mut ArenaChoice, block: usize, why: Skip) -> bool {
    let Some(i) = choice.blocks.iter().position(|b| b.block == block) else {
        return false;
    };
    let b = choice.blocks.remove(i);
    if let Some(s) = choice
        .devices
        .iter_mut()
        .find(|s| s.budget.device == b.device)
    {
        s.bytes = s.bytes.saturating_sub(b.bytes);
        s.chosen = s.chosen.saturating_sub(1);
    }
    choice.skipped.push((b, why));
    true
}

/// The serialisable summary that goes into `LoadStats` (and so into the JSON
/// report): enough to explain the choice without re-running the rule.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ArenaReport {
    /// `auto` or `list`.
    pub rule: String,
    /// The resident blocks, in block order.
    pub blocks: Vec<usize>,
    pub bytes: u64,
    pub devices: Vec<DeviceArenaRow>,
    /// Blocks left on the host lane: `block: reason`.
    pub skipped: Vec<String>,
}

/// One device's line in [`ArenaReport`].
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DeviceArenaRow {
    pub device: usize,
    pub label: String,
    pub name: String,
    pub display: bool,
    pub source: String,
    pub free_bytes: u64,
    pub reserve_bytes: u64,
    pub budget_bytes: u64,
    pub chosen: usize,
    pub bytes: u64,
    pub left_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    fn budgets(free: &[u64], reserve: u64) -> Vec<DeviceBudget> {
        free.iter()
            .enumerate()
            .map(|(i, &f)| DeviceBudget {
                device: i,
                label: format!("dev{i}"),
                display: false,
                free: f,
                reserve,
                source: BudgetSource::MemoryBudget,
            })
            .collect()
    }

    #[test]
    fn the_rule_fills_the_budget_with_the_largest_arenas() {
        // Two blocks of 900, three of 600, a 100 MiB budget of 2000.
        let blocks = vec![
            ArenaBlock {
                block: 0,
                device: 0,
                bytes: 900,
            },
            ArenaBlock {
                block: 1,
                device: 0,
                bytes: 600,
            },
            ArenaBlock {
                block: 2,
                device: 0,
                bytes: 900,
            },
            ArenaBlock {
                block: 3,
                device: 0,
                bytes: 600,
            },
            ArenaBlock {
                block: 4,
                device: 0,
                bytes: 600,
            },
        ];
        let b = budgets(&[2100], 100);
        let c = select(&blocks, &b);
        assert_eq!(c.devices[0].budget.budget(), 2000);
        // 900 + 900 = 1800; the next 600 does not fit, and 200 MiB stays free.
        assert_eq!(
            c.blocks.iter().map(|x| x.block).collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert_eq!(c.bytes(), 1800);
        assert_eq!(c.devices[0].left(), 200);
    }

    #[test]
    fn a_skipped_big_block_does_not_hide_the_smaller_ones() {
        // A block bigger than the whole budget must not hide the two 480s behind it.
        let blocks = vec![
            ArenaBlock {
                block: 0,
                device: 0,
                bytes: 1200,
            },
            ArenaBlock {
                block: 1,
                device: 0,
                bytes: 480,
            },
            ArenaBlock {
                block: 2,
                device: 0,
                bytes: 480,
            },
        ];
        let c = select(&blocks, &budgets(&[1200], 200));
        assert_eq!(
            c.blocks.iter().map(|x| x.block).collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(c.bytes(), 960);
        assert_eq!(c.skipped, vec![(blocks[0], Skip::TooBig)]);
    }

    #[test]
    fn ties_break_on_the_lowest_block_index() {
        // This is what makes the rule reproduce the hand-tuned `0..15,28..41`
        // when every arena is the same size.
        let blocks: Vec<ArenaBlock> = (0..6)
            .map(|block| ArenaBlock {
                block,
                device: 0,
                bytes: 100,
            })
            .collect();
        let c = select(&blocks, &budgets(&[400], 100));
        assert_eq!(
            c.blocks.iter().map(|x| x.block).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }

    #[test]
    fn a_block_too_big_for_its_device_is_reported_not_attempted() {
        let blocks = vec![ArenaBlock {
            block: 7,
            device: 0,
            bytes: 1200,
        }];
        let c = select(&blocks, &budgets(&[1300], 200));
        assert!(c.blocks.is_empty());
        assert_eq!(c.skipped, vec![(blocks[0], Skip::TooBig)]);
    }

    #[test]
    fn blocks_land_on_the_device_that_owns_them() {
        let blocks = vec![
            ArenaBlock {
                block: 0,
                device: 0,
                bytes: 500,
            },
            ArenaBlock {
                block: 1,
                device: 1,
                bytes: 500,
            },
            ArenaBlock {
                block: 2,
                device: 0,
                bytes: 500,
            },
        ];
        let c = select(&blocks, &budgets(&[600, 600], 100));
        assert_eq!(
            c.blocks.iter().map(|x| x.block).collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(c.devices[0].chosen, 1);
        assert_eq!(c.devices[1].chosen, 1);
        assert_eq!(c.host_blocks(48), 46);
    }

    #[test]
    fn a_dropped_block_returns_its_bytes() {
        let blocks: Vec<ArenaBlock> = (0..3)
            .map(|block| ArenaBlock {
                block,
                device: 0,
                bytes: 100,
            })
            .collect();
        let mut c = select(&blocks, &budgets(&[400], 100));
        assert_eq!(c.bytes(), 300);
        assert!(drop_block(&mut c, 1, Skip::UploadFailed));
        assert_eq!(c.bytes(), 200);
        assert_eq!(c.devices[0].chosen, 2);
    }

    #[test]
    fn the_cli_value_parses_either_way() {
        assert_eq!(
            ArenaSpec::parse("auto"),
            ArenaSpec::Auto(ArenaOpts::default())
        );
        assert_eq!(ArenaSpec::parse("0,1, 4"), ArenaSpec::List(vec![0, 1, 4]));
        assert_eq!(ArenaSpec::parse(""), ArenaSpec::Off);
    }

    #[test]
    fn the_default_reserves_leave_a_display_card_alone() {
        let o = ArenaOpts::default();
        assert!(o.display_reserve > o.headroom);
        assert_eq!(o.display_reserve, 768 * MIB);
    }
}
