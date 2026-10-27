# 16 — SGLang: radix prefix reuse for a hybrid GDN model, HiCache tiers, and chunked prefill

Round 16. Deep-dive for README row 10 (SGLang, previously survey-only); indexed as row 22. Section A is the upstream anatomy. Section B answers the four questions. Section C ranks the three recommendations. Sections D–F cover what not to copy, the conflicts, and what was not verified.

**Pinned references.** SGLang at `4a58dc983825e1b150d2f375e6885a9f2c7f52ec` (upstream main at fetch time). Upstream paths are relative to `python/sglang/srt/` unless shown otherwise; `blob` is the git blob id of the file at that ref. The pin in `docs/reference-engines/02-kv-and-context.md` (`8252e3a`) is stale: the hybrid and mamba prefix path has moved into the unified radix cache and `unified_cache/components/`, and the Rust radix tree lives in `rust/sglang-radix-tree/`. Line numbers from the old pin were not re-verified and are not used here. The paths and blob ids in section A were checked against the git tree at the pin; the load-back test directory was listed, not read.

**Label note.** Nothing was built or run on a GPU; no cargo or build step was run. Anything that needs the card is marked `UNVERIFIED - requires the GPU qualification card`. Numbers taken from earlier rounds carry their round.

**Headline.**

1. **For a hybrid model, the reusable unit is the deepest GDN state on the matched path, not the KV hit.** A few-hundred-token shared system prompt gets no reuse from periodic 16K snapshots. Upstream plants a state at the deepest grid-aligned point of a request's KV match that the restorable state does not already cover. The second request that shares a prefix plants it, so the third request restores it. This is R1's "admit on second hit" with a mechanism behind it.
2. **The snapshot grid is `lcm(mamba chunk, page)`, and upstream floors the chunk at 64.** The floor is a correctness guard, not a speed setting: a grid of 1 lets the radix tree adopt another request's KV at tiny shared prefixes, and the kernel rounding then changes decode logits. For Breccia the grid is `lcm(C, 16)` with `C >= 64` on any snapshot-bearing run.
3. **Upstream persists mamba state write-through only.** Breccia's GDN state is mutated in place every token, so its snapshot must be captured with a fence and a staging copy, not a deferred write.
4. **The 256-token prefetch threshold is the wrong number for our tier.** At about 88 ms per prefill token per stream at chunk 64 (round 07's full-chunk figure) and about 25 GB/s host-to-device, restoring a 64-token boundary costs about 5 ms against about 5.6 s of prefill for the same 64 tokens. The threshold should be one grid.
5. **Restore overlap is not a top-three item.** Layer-wise restore events save at most about 0.1 s per 100K-token hit. The mechanism matters for expert streams, which is round 14 R4, and that is where it ranks.

---

## A. Upstream anatomy (pinned at `4a58dc98`)

| Concern | Upstream symbol | File (blob) |
|---|---|---|
| Restorable boundary and branch point | `MambaComponent.finalize_match_result_in_tree_core`, L164–194 | `mem_cache/unified_cache/components/mamba.py` (`a69394e6`) |
| Branch point carried to the request | `req.mamba_branching_seqlen = match_result.mamba_branching_seqlen` | `managers/schedule_batch.py` (`0c7e55c0`); `mem_cache/common.py` (`4a2f0fec`) |
| Prefill plants the branch-point state | plant when `prefix_len < branching_len <= prefix_len + new_token_count` (MLX backend only was read) | `hardware_backend/mlx/model_runner.py` (`90d74690`) |
| Snapshot grid | `mamba_checkpoint_grid(tree_page) = lcm(mamba_cache_chunk_size(), tree_page)` | `runtime_context.py` (`054e46f3`) |
| Chunk | `mamba_cache_chunk_size(server_args)` = `max(model mamba chunk or FLA CHUNK_SIZE, page_size)`, asserts one divides the other | `arg_groups/overrides.py` (`d29583d1`) |
| Decode-donated grid | `mamba_track_grid(tree_page) = lcm(mamba_checkpoint_grid, mamba_track_interval)` | `runtime_context.py` (`054e46f3`) |
| Rust mirror of the match rule | `least_common_multiple(mamba_cache_chunk_size, page_size)`; `mamba_branching_seqlen = (aligned > boundary).then_some(aligned)` | `rust/sglang-radix-tree/src/components/mamba.rs` (`a6f5e33b`) |
| Eviction | `_evict_excess_path_states` (keeps tail, forks, locked nodes, device leaves; soft cap `mamba_max_states_per_path`); `refresh_lru` touches only the matched node's state | `mem_cache/unified_cache/components/mamba.py` (`a69394e6`) |
| Prefill budget | `PrefillAdder.__init__` (decode debit), `floor_chunked_tokens`, `add_chunked_req`, `mamba_prefill_chunk_grid` | `managers/schedule_policy.py` (`debb5756`) |
| HiCache write policy | `write_policy` default `write_through_selective`; mamba incremental persistence is write-through only (comment at L176–179) | `mem_cache/hybrid_cache/hybrid_cache_controller.py` (`585e536a`); `managers/cache_controller.py` (`08932ece`); `mem_cache/unified_cache/components/mamba.py` (`a69394e6`) |
| Storage write threshold | `write_through_threshold = 1 if write_through else 2` (write after the Nth hit) | `mem_cache/unified_cache/storage_attachment.py` (`1d768766`) |
| HiCache prefetch | `prefetch_threshold` default 256, floored at `page_size`; prefetch when storage hit count is at least the threshold | `mem_cache/hybrid_cache/hybrid_cache_controller.py` (`585e536a`); `managers/cache_controller.py` (`08932ece`) |
| Layer-wise load overlap | `LayerLoadingEvent` (per-layer `load_events`, `wait`, `complete`); `start_loading` L1028–1060 | `managers/cache_controller.py` (`08932ece`) |
| Write fencing | `write_fence_stream` gates D2H in `start_writing` L885–926 | `managers/cache_controller.py` (`08932ece`) |
| Quantised checkpoint pool | `Int8CheckpointStore`, int8 with per-(head, k-channel) scale, QMAX 127, gated by `--enable-int8-mamba-checkpoint` | `mem_cache/mamba_checkpoint_pool.py` (`e1d1237b`) |
| Prefill chunk sizing (PP only) | quadratic fit f(l) = a·l² + b·l + c from up to 128 synthetic prefills | `managers/scheduler_components/dynamic_chunk_sizer.py` (`858ada87`) |

**Test evidence for the branch rule.** The load-back test file under `test/registered/unit/mem_cache/unified_radix_cache/` covers the cases that matter here (ref `4a58dc98`, no blob read):

- `test_mamba_branching_seqlen_disabled_under_hicache`: with no HiCache, dropping the leaf's state gives a branch point of one chunk. Under HiCache, with the leaf's state only on host, the match returns no branch point. The test name suggests HiCache disables the feature. The assertion is narrower: the host-restorable boundary already covers the aligned prefix, so no new point is needed.
- `test_mamba_branching_seqlen_uses_device_full_hit_under_hicache`: device prefix of one chunk, full KV hit of two chunks plus one token, state dropped from the leaf. Branch point = `2 * chunk`.
- `test_mamba_branching_from_host_full_is_reusable_after_insert`: the same case with the state on host. After the branch-point prefix is inserted and flushed, a second match has device prefix `2 * chunk` and no branch point. One new host mamba slot is allocated and the full-KV host slot is not reallocated.

---

## B. The four questions

### Q1. Prefix reuse when GDN state is not positionally sliceable (mamba radix semantics, against R1)

**What upstream does.** The restorable length is bounded by the mamba boundary, `mamba_boundary_len = device_prefix_len + host_hit_length`. The KV hit can extend past that boundary. The branch point is the grid-aligned floor of the full-KV hit, used only if it lies beyond the boundary:

```
branching = aligned(full_kv_hit) if aligned(full_kv_hit) > boundary else None
```

The request carries that value, and prefill plants a state there when the extend crosses it. Tokens past the boundary are recomputed.

Two consequences matter for Breccia:

- **The fork rule is the admission rule.** The first request has an empty KV hit and plants nothing. A second request that shares a prefix `P` matches `P` in the KV tree, so its branch point is `aligned(P)`, and it plants there while prefilling. A third request restores at `aligned(P)`. Admission on the second hit is therefore a consequence of the match, not a separate policy. R1 left this as "admit on second hit (UNVERIFIED)"; upstream supplies the mechanism.
- **Upstream also has a periodic class.** `mamba_track_interval` gives decode-donated states on `mamba_track_grid`. R1 corresponds to this periodic class. The fork-point class is what R1 lacks.

**What this means for Breccia.** The planner needs three numbers: the full-KV hit, the device-restorable prefix, and the host-restorable prefix. The branch point is then the floor of the full-KV hit on the grid `G`, if it lies beyond the restorable boundary. Two details differ from upstream:

- **Tokens between the deepest snapshot and the KV hit must be recomputed in full, KV writes included.** The layers are sequential: a QSA layer's input depends on the GDN layers below it, so attention cannot be skipped for a span while the GDN stack runs. A snapshot is worth exactly the prefix it covers, and the recompute beyond it is at most `G - 1` tokens plus the new suffix.
- **Periodic density is the expensive choice.** Periodic snapshots at grid `G` cost `117.7 MB × N / G` per stream. At `G = 64` that is about 184 GB per 100K tokens, which the 30 GB host tier cannot hold. At `I = 16K` it is 673 MiB per 100K (R1). Fork-point snapshots cost one 117.7 MB snapshot per distinct fork.

The per-stream GDN state is 112.2 MiB (36 × (3 MiB delta-rule + 120 KiB conv); `crates/breccia-kernels/src/gdn/mod.rs:79–102`). The snapshot is 117,669,888 bytes.

**Answer.** Use the fork-point rule on top of R1's periodic snapshots. A shared prefix of `N` tokens saves about `N × 88 ms` per hit from the third request onward. For `N = 512` that is about 45 s, `UNVERIFIED - requires the GPU qualification card`. The shared-prefix projection in `docs/phase-4/02-untried-levers.md` (4 agents sharing 200 of 300 prefix tokens: prompt seconds 95.6 → about 35, `UNVERIFIED - requires the GPU qualification card`) is an unmeasured prior, not a check.

**Eviction.** `_evict_excess_path_states` keeps the tail, forks, locked nodes, and device leaves, and enforces `mamba_max_states_per_path` as a soft cap. `refresh_lru` touches only the matched node's state, so ancestors that are not forks age out first. R1 should adopt the same rule: non-fork periodic ancestors are evicted before forks.

### Q2. Chunked prefill sizing and interleaving with decode on two GPUs

**What upstream does.**

- The per-step chunk budget is debited by decode tokens: `rem_chunk_tokens -= num_mixed_decode_tokens` (`managers/schedule_policy.py`, `PrefillAdder.__init__`).
- `add_chunked_req` handles a request whose prefill is already chunked (name confirmed in `managers/schedule_policy.py`; admission order was not read).
- A truncated chunk is rounded down to the mamba grid only when the cut is real (`floor_chunked_tokens`). The grid is 1 unless DCP is enabled (`mamba_prefill_chunk_grid`). The docstring says that without DCP, prefix-relative snapshots make rounding the cut worth at most one grid of budget.
- `DynamicChunkSizer` is PP only.

**What this means for Breccia.**

- **The chunk `C` sets the grid.** `G = lcm(C, 16)`. The chunk candidates `{32, 64, 128, 256}` (each divides 16K) are legal for runs without snapshots. With snapshots, `C = 32` gives `G = 32`, below the floor of 64 explained in the headline. Treat `C = 32` as ineligible for snapshot-bearing runs until the exactness test passes. `C = 128` or `256` coarsens the grid, which costs up to `G - 1` recomputed tokens per hit, because fork snapshots sit at the fork's own floor. At `C = 64` the recompute beyond a fork is at most 63 tokens, about 5.5 s at 88 ms per token. Round 15 R1's exact-restore requirement also means a restored run and a cold run must share the same chunk partition, `UNVERIFIED - requires the GPU qualification card`.
- **Interleaving needs per-stream positions first.** `Multi` steps every stream at one shared position (`crates/breccia-engine/src/multi.rs`, `generate`, about L225–250, via `token_all(&vec![t; ns], pos)`). Round 13 item 2 (`multi.rs:234`) is the prerequisite. Once positions are per stream, one step is one prefill chunk of `C` tokens for stream A plus one decode token for each other stream. Upstream debits the decode tokens from the chunk budget (`C_eff = C - n_decode`) and floors only a real cut to `G`. Breccia does not copy that debit. It keeps the prefill chunk at exactly `C` and adds the decode tokens on top of the step. This is a deliberate deviation. Every chunk candidate is a multiple of 16, so `G = C`: a debited chunk is shorter than `G`, so flooring it would empty the step, and leaving it unfloored would move the partition off multiples of `C`, which breaks the grid rule in R-1 that every branch point sits on a cold-run chunk boundary.
- **Upstream does not fix the ragged tail.** Its floor applies only to truncated chunks. The prompt's final chunk is not truncated, so it is not rounded. Round 07 measured the tail chunk at 658 ms per token against 88 ms per token for a full chunk, so that lever stands.
- **Sizing is a measurement, not a policy.** There is no analogue for a layer split across two cards. `DynamicChunkSizer`'s method (fit the per-chunk wall against length on synthetic prefills) is a usable measurement method for a `C` table. Check it with the `sum · wall · residual` invariant.

**Answer.** Keep `C = 64`. Sweep `{32, 64, 128, 256}` only on the no-snapshot path until the exactness test passes. Interleave after per-stream positions land. Interleave throughput gains `UNVERIFIED - requires the GPU qualification card`.

### Q3. HiCache write policy and prefetch for a 30 GB host tier

**What upstream does.**

- Both controllers default to `write_through_selective`.
- The storage attachment writes after the Nth hit. For selective policies `N = 2`.
- Mamba incremental persistence is write-through only. The comment at `mamba.py` L176–179 says write-back eviction may discard device-only state.
- Prefetch threshold defaults to 256 tokens and is floored at the page size. Prefetch policies are `best_effort`, `wait_complete`, and `timeout`.
- `load()` allocates full device rows, not just the missing span.

**What this means for Breccia.**

- **Capacity.** At 117.7 MB per snapshot, 30 GB holds about 255 per-stream GDN snapshots (about 273 at 30 GiB). At 24 KiB per token of f16 KV it holds about 1.2 M tokens. How the 30 GB is split between classes is not known (`UNVERIFIED - requires the GPU qualification card` for the split and its effect).
- **Write policy per class.** GDN snapshots: write-through at capture, admitted only on a second fork hit (selective, threshold 2), matching section B's Q1 rule. QSA pages: write back only when a page is full, because QSA pages are append-only and never rewritten in place, so a deferred write is safe (UNVERIFIED: QSA page mutation was not checked against the engine).
- **Prefetch.** Restoring a 64-token boundary moves 117.7 MB of GDN state plus 1.5 MiB of KV. At 25.4 GB/s that is about 5 ms, against about 5.6 s of prefill for the same 64 tokens. Prefetch any hit that covers a snapshot; the threshold is one grid, not 256 tokens. The 256 default is a storage-IO guard and does not transfer.

**Answer.** Write-through with a second-hit gate for GDN, write-back on full pages for QSA, prefetch at one grid. Per-hit restore times are `UNVERIFIED - requires the GPU qualification card`.

### Q4. Overlapping host-to-device copies with compute on Vulkan

**What upstream does.** H2D runs on `host_to_device_stream`, which waits on `load_fence_stream` (set to the forward stream) before writing, so a load cannot overwrite memory the forward still reads (`managers/cache_controller.py`, `start_loading` L1028–1060). Each layer has a `LayerLoadingEvent`. The forward waits on `load_events[i]` before layer `i`, and the loader records it after layer `i` arrives (`complete`). Writes are gated by `write_fence_stream` (`start_writing` L885–926).

**Vulkan mapping.** The CUDA stream and event pair maps to two timeline semaphores (core in Vulkan 1.2; Breccia's device support for them was not checked this round). The copy queue signals a per-block value when that block's data is resident. Compute waits on that value before the block. The copy queue waits on the compute timeline before overwriting a slot that compute is still reading (write-after-read). Two slots is the minimum (double buffering). Use bounded waits only, never `vkWaitForFences(u64::MAX)`.

**Queue family.** On the RX, a dedicated transfer family appears only under `RADV_EXPERIMENTAL=transfer_queue` (round 09). Round 02 measured the transfer family at 8.6× slower for on-card copies on the MI50. Use a second compute-family queue. The transfer ring is described as defaulting to that mode; which mode the code selects was not checked this round (`crates/breccia-vk/src/`, `TransferRing`).

**Cost of host fences.** A submit with fence costs a fixed 56 µs (round 07). Forty-eight per-block host waits would cost about 2.7 ms per token. Timeline values avoid that.

**What this means for Breccia.**

- **Restore overlap is not worth ranking.** The 100K-token restore set (2.46 GB KV plus 117.7 MB GDN) takes about 0.1 s at 25 GB/s, roughly one 88 ms prefill token. Per-layer restore events hide at most that time per hit on the RX link. On the MI50 path, round 15 estimates at the BRIEF link figure of about 2.9 GB/s (not measured) about 0.66 s for the q8 KV set (1.92 GB) and about 0.85 s for the f16 set (2.46 GB) per 100K tokens, `UNVERIFIED - requires the GPU qualification card`. Even that is small against the prefill it replaces, so Q4 is not a ranking input.
- **Expert-stream overlap is the one that matters.** The 18 blocks on the CPU lane move 301 MiB per token. If that traffic were streamed over PCIe instead, it would take about 12 ms per token at 25.4 GB/s (`UNVERIFIED` for the current code path). Overlap hides `min(copy, GEMM)` per block. Round 14 R4 gates this on R1 of round 14: copy and GEMM must each be at least 10% of a chunk. That share is not measured yet, `UNVERIFIED - requires the GPU qualification card`.

**Answer.** Restore overlap: no. Expert-stream overlap with timeline events on a second compute-family queue: yes, ranked third, behind round 14 R4's gate.

---

## C. Ranked recommendations

### R-1 (rank 1). Fork-point GDN snapshots on the chunk grid, layered on R1

**(a) Upstream.** `MambaComponent.finalize_match_result_in_tree_core` (`mamba.py`, blob `a69394e6`); `mamba_checkpoint_grid` and `mamba_track_grid` (`runtime_context.py`, blob `054e46f3`); `mamba_cache_chunk_size` (`arg_groups/overrides.py`, blob `d29583d1`); `mamba_prefill_chunk_grid` and `floor_chunked_tokens` (`managers/schedule_policy.py`, blob `debb5756`); the plant rule (`hardware_backend/mlx/model_runner.py`, blob `90d74690`).

**(b) Breccia adaptation.** A pure planner in the planned `crates/breccia-kv` takes `(full_kv_hit, device_restorable, host_restorable, G)` and returns the branch point or none. `G = lcm(C, 16)` with `C >= 64`. Because `G` is a multiple of `C`, every branch point is a chunk boundary of the cold run's schedule, which is the property that keeps a restore exact. The prefill loop plants a GDN snapshot at the branch point. Eviction follows the upstream rule: tails, forks and leaves are kept, and non-fork periodic ancestors go first. The snapshot is captured by R-2.

**(c) Expected effect.** For a shared prefix of `N` tokens, each hit from the third request onward saves `N × 88 ms`. Periodic 16K snapshots give none for `N < 16K`. Fork snapshots cost 117.7 MB per distinct fork, against about 184 GB per 100K tokens for periodic snapshots at `G = 64`. Hit rate and saving: `UNVERIFIED - requires the GPU qualification card`.

**(d) Measurement plan.**
- Cheap-offline: planner unit tests. The branch point is always a multiple of `G`, always beyond the restorable boundary, and absent when the host hit covers the aligned prefix. Mirror the three upstream load-back cases in section A. Add a property test over random hit lengths.
- GPU-window: a restored run against a cold run. Greedy 32/32 token parity, and bitwise equality of the restored GDN state against the cold snapshot at the branch point (checksum on both).
- GPU-window: a workload with a 512-token shared system prompt and four requests. Count prefill tokens saved from the third request on. `UNVERIFIED - requires the GPU qualification card`.

### R-2 (rank 2). Staged fork-point snapshot capture, write-through, second-hit admission

**(a) Upstream.** `start_writing` (`managers/cache_controller.py`, blob `08932ece`, L885–926) waits on `write_fence_stream` before D2H; storage write threshold (`mem_cache/unified_cache/storage_attachment.py`, blob `1d768766`); mamba write-through only (`mem_cache/unified_cache/components/mamba.py`, blob `a69394e6`, L176–179).

**(b) Breccia adaptation.** At a branch point, the last GDN dispatch of the boundary token signals a timeline value. A copy on the compute queue moves the 112.2 MiB state into a VRAM staging slot. A D2H on a second compute-family queue waits for the staging copy's value, then signals completion. Compute continues in the meantime. Only second-hit forks go to host. The staging slot's VRAM headroom on the RX is not measured (round 12: 3.66 GiB dense plus about 8 GiB arena on the RX, against 16 GiB), `UNVERIFIED - requires the GPU qualification card`.

**(c) Expected effect.** Removes a host-synchronous stall of about 6.6 ms per snapshot (117.7 MB at 17.9 GB/s down, round 09) and a 56 µs fence (round 07). At one snapshot per 16K tokens that is about 0.4 µs per token, so this is a stall and correctness item rather than a TPS item. That is why it ranks second. Stall per snapshot: `UNVERIFIED - requires the GPU qualification card`.

**(d) Measurement plan.**
- GPU-window: stall per snapshot with and without staging.
- GPU-window: host copy against staging copy, checksum equality.
- GPU-window: a fault-injection run with the staging signal deliberately withheld. It must end in a bounded timeout, not a hang. This tests the no-unbounded-wait rule.

### R-3 (rank 3). Per-block timeline events for expert-stream copies on a second compute-family queue

**(a) Upstream.** `LayerLoadingEvent` and `start_loading` (`managers/cache_controller.py`, blob `08932ece`, L64–80 and L1028–1060). The loader records `complete(layer)`, and the forward waits on `load_events[i]`.

**(b) Breccia adaptation.** This is round 14 R4 with the mechanism named. The copy queue signals block `b`'s expert-slice value. Compute for block `b` waits on it. The copy for block `b + 1` waits on the compute value for block `b - 1` (slot reuse). Two slots minimum. Copies run on the second compute-family queue, not the transfer-only family (round 09 and round 02). Round 14 R4 is gated on R1 of round 14 showing copy and GEMM each at 10% or more of a chunk.

**(c) Expected effect.** Per block, overlap saves `min(copy, GEMM)`. The streamed-equivalent copy of the CPU lane's 301 MiB per token is about 12 ms per token at 25.4 GB/s (`UNVERIFIED` for the current path). The realised saving depends on the round 14 gate: `UNVERIFIED - requires the GPU qualification card`.

**(d) Measurement plan.**
- Cheap-offline: a host-side simulator of the timeline protocol (no GPU). Assert that no slot is written before its last reader signals, for every interleaving of a bounded set of blocks.
- GPU-window: the round 13 copy/compute overlap ratio η (`UNVERIFIED` in round 13), after round 14's gate is met.
- GPU-window: parity 32/32 with overlap on against overlap off. Bounded waits only.

---

## D. What not to copy

- **Int8 checkpoint pool** (`mem_cache/mamba_checkpoint_pool.py`, `Int8CheckpointStore`, blob `e1d1237b`). Lossy: int8 with a per-(head, k-channel) scale, QMAX 127. It conflicts with R1's exact-restore rule. If it is ever used, it must be a labelled non-exact tier behind the parity gate. It is not ranked.
- **Prefetch threshold of 256 tokens.** A storage-IO guard. For Breccia's host tier the threshold is one grid (section B, Q3).
- **Grid of 1.** The Inkling config (`configs/inkling.py`, blob `6047744e`) floors the mamba chunk at 64 for this reason: a smaller grid lets the radix tree adopt another request's KV at tiny shared prefixes, and kernel rounding then perturbs decode logits. Do not allow grids below 64 on snapshot-bearing runs.
- **Mamba write-back.** Upstream does not use it for mamba state, and neither should Breccia for GDN.
- **`DynamicChunkSizer`** (`managers/scheduler_components/dynamic_chunk_sizer.py`, blob `858ada87`). PP only, with no analogue for a layer split across two cards. Its fitting method is a measurement method, not a policy.
- **Test name `test_mamba_branching_seqlen_disabled_under_hicache`.** The assertion does not disable anything under HiCache; it checks that a host-covered boundary needs no new point. Do not copy the name.
- **CUDA stream names** (`device_to_host_stream`, `host_to_device_stream`) and CUDA event waits. The Vulkan mapping is in section B, Q4.

---

## E. Conflicts and superseded numbers

1. **Barrier pruning.** `docs/BRIEF.md` L178 and `10-dispatch-reduction-and-exllamav2.md` L55–60 measured 19.6% fewer barriers and zero gain. Earlier projections (`06-vulkan-gemm-shapes-and-ncnn.md` L190–191, L314, L350: −1.0 to −4.5 ms per token; `04-vulkan-launchcount-and-lmdeploy.md`: −0.7 to −3.2 ms per token) are superseded by that measurement. None of R-1 to R-3 depends on barrier pruning. R-2 and R-3 use the round 07 fence cost and the round 06a barrier law.
2. **Exact restore against int8 checkpoints.** R1 requires exact restore; upstream's int8 pool is lossy. Resolved by leaving int8 out of the ranked list.
3. **Stale pin.** `docs/reference-engines/02-kv-and-context.md` §7 cites `8252e3a` line numbers that no longer match the tree. This round does not edit that file.
4. **Write policy in the HiCache test.** The third load-back case flushes with `writing_check(write_back=True)`, while the mamba component comment says new branch persistence is write-through only. Which policy the fixture uses was not checked (`UNVERIFIED`). The write-back flush cannot be read as settled.

---

## F. Unverified and skipped

- The `scheduler.py` overlap loop (`event_loop_overlap`) was not read in full.
- The branch-point plant was read only in the MLX model runner. The CUDA path was not read.
- `disaggregation/prefill.py` was not read.
- The HiCache load and write streams are CUDA-named. The Vulkan mapping in this doc is a proposal.
- `8252e3a` line numbers were not re-verified.
- `multi.rs` was re-checked this round: `token_all` at L178, `generate` at L225, prompt loop at L234–235; the decode loop was not read in full.
- VRAM headroom for the 112 MiB staging slot on the RX was not measured.
- Whether the 30 GB host tier is shared with the expert lane's pinned memory is not known.
- The 88 ms per prefill token is round 07's full-chunk per-stream figure at chunk 64; it is not re-measured. The 155–166 ms agent-token figure is a denominator artefact and is not used.
- Every hit-rate, saving, stall, and overlap number is `UNVERIFIED - requires the GPU qualification card`. No GPU work was done.
