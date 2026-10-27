# 21 — Speculative decoding (MTP, suffix drafter, layer skip) and llama.cpp RPC for Breccia

Round 21 of the paired rotation. Research only: no code was changed, and no GPU, binary or benchmark was run. Every speed number is cited, marked `derived` (arithmetic on cited inputs), or marked `UNVERIFIED - requires the GPU qualification card`. The tailnet mesh is deferred and is not covered; only the llama.cpp RPC mechanism is.

## 1. Summary

- Breccia has no speculative decoding. `crates/breccia-format/src/mtp.rs` loads the MTP head, no crate runs its forward pass, and `crates/breccia-draft/src/lib.rs:1` is a stub. The "0.60×" is a projection in `docs/phase-4/02-untried-levers.md` §6, not a measurement. The 10–12 → 1.6 tok/s collapse came from the earlier HIP stack (`docs/phase-0/01-baseline-hip.md:48`), not from Breccia.
- Verifying k drafts reads the union of the routed experts of k+1 rows. Under uniform routing that union is 3.9× one token's experts at k=3 (derived). Whether speculation pays depends on s, the share of a token's time that scales with expert reads. The host-lane share puts s at 0.23, which is a floor. Doc 02's 0.60× implicitly assumes s≈1.
- At s=1 there is no break-even at any k once a draft cost is added. At s=0.23 the break-even acceptance is 0.30–0.56 for k=1..4 with an MTP drafter (d=0.07), and 0.23–0.48 with a suffix drafter (d=0). Both are derived (§3). They are optimistic lower bounds, because s=0.23 is a floor; at s=0.5 the MTP range is 0.56–0.77.
- Gate F (85 tok/s, `docs/plan/BRECCIA-PLAN.md:50`; Phase 5 at `:46`) is 5.0× the 16.94 tok/s ladder row. Since r ≥ 1 and Y < 1/(1 − p), speedup S < 1/(1 − p), so 5× needs acceptance above 0.8 at any k (derived). The recorded acceptances are 0.53–0.65 (secondhand, doc 02), so speculation does not close Gate F under this model.
- Prior art: llama.cpp merged an MTP path (PR #22673). Its named model families at the pinned SHA are step35 (chain heads) and qwen35 / qwen35moe (single head). Qwen3-Next and GLM-4-MoE are registered architectures with generic NextN tensors, but no driver names them. No primary source gives Qwen3-Next MTP acceptance numbers.

## 2. Breccia state

- **Tried.** Nothing on the Breccia engine. The earlier HIP stack's naive MTP on the RX dropped decode from 10–12 to 1.6 tok/s "by doubling expert reads" (`docs/phase-0/01-baseline-hip.md:48`). `docs/plan/BRECCIA-PLAN.md` Phase 5 calls it the known trap.
- **Present.** Loader only: `crates/breccia-format/src/mtp.rs:22` (`MtpDenseKind`), `:81` (`MtpGeometry`; default is the Qwen3.8 draft layer at `:87`), `:119` (`MtpRt`), `:136` (`open_with_geometry`), `:285` (`expert_blob`), `:301` (`draft_vocab_len`). The draft vocabulary is 106,299 ids (`mtp.rs:11`). The MTP expert set is 512 × 1,382,400 B = 708 MB (`crates/breccia-format/src/mtp.rs:9–10` and `:99`; doc 02 §6 quotes the same 708 MB for Basalt's MTP set).
- **The 0.60× (doc 02 §6).** `docs/phase-4/02-untried-levers.md:281` and `:286`: verify cost 1 + 3 × 0.95 = 3.85 at K=3, yield 2.31, and 2.31 / 3.85 = 0.60×. The text says p=0.6, but Y(3, 0.6) = 1 + 0.6 + 0.36 + 0.216 = 2.176, which gives 0.565×. A yield of 2.31 needs p≈0.64. The verdict holds: at m=0.95 with zero draft cost, break-even needs p≈0.97 at k=3 (derived). The label is inconsistent; the conclusion is not.
- **"Cache fixed."** Not defined in one place. Read with doc 02 §6 and `docs/BRIEF.md:287` ("needs the cache fixed first"), it means the expert-residency cache serves the drafter and the verify rows, so drafting adds no host-lane reads. Basalt keeps its 512 MTP experts in VRAM (`core/mtp.hpp:13–17`, as cited in doc 02 §6). Whether Breccia's MTP set fits in VRAM beside its 30-block residency is UNVERIFIED - requires the GPU qualification card.
- **Acceptance.** Breccia has no MTP acceptance number. Other engines' figures are secondhand, as quoted in doc 02 lines 274–275 (the source files are not in this worktree): Basalt 61.1–64.5% at K=7–8 (cited there as `docs/precision/features.md:217`); capy T24 at K=3, acceptance 0.533 and yield 2.59 (same table).
- **Missing pieces.** No multi-row verify path, and no GDN snapshot or rollback. `docs/phase-3/06-attention-gdn.md:504–507` says so, and `docs/phase-3/04-op-kernels.md:684` lists snapshot/rollback as M4. The recurrent state is `gdn_state` (`crates/breccia-engine/src/graph.rs:490`), reset at `:1613`.

## 3. MoE economics and break-even (derived)

Union. A token's block reads 10 of 512 experts. For n rows in one verify, uniform random routing gives

D(n) = 512 × (1 − (502/512)^n)

D(1) = 10.00, D(2) = 19.80, D(3) = 29.42, D(4) = 38.84, D(5) = 48.08. D(4) = 38.84 matches doc 02's "38 of 40" (secondhand). Real routing is correlated across consecutive tokens, so the true union is smaller; it is not measured here. A 4-row verify reads 3.88 × 799 MiB ≈ 3,100 MiB (about 3.0 GiB) of expert bytes per step (derived). The 799 MiB per token is ladder row 0 (zero resident, `docs/phase-4/01-forward-pass.md:2962`). At the row-30 placement the host bytes are 301 MiB, so the host-lane verify there is 3.88 × 301 ≈ 1,170 MiB (derived).

Time model, in units of T1 = 59.0 ms (ladder row 30, `docs/phase-4/01-forward-pass.md:2965`):

- Verify of k drafts: r = (1 − s) + s × D(k+1)/10 + k × d. The (1 − s) term is held flat across rows. That is fair for the DRAM-bound RX. The ALU-bound MI50 share would grow with rows, which makes r larger and break-even harder; the block split is `--split-blocks 28,20` (`docs/phase-4/01-forward-pass.md:2956`), and which device carries the 20-block share is not checked here. UNVERIFIED - requires the GPU qualification card.
- Yield: Y(k,p) = Σ from i=0 to k of p^i (i.i.d. acceptance p; Y ≤ k+1).
- Speedup S = Y / r. Break-even p* solves Y(k,p*) = r. If r ≥ k+1, no p works.
- d = 0.070: one average block (59.0 − 7.1) / 48 = 1.08 ms, plus the draft-vocab head 7.1 × 106,299 / 248,320 = 3.04 ms. The 7.1 ms full-vocab head is `docs/phase-4/01-forward-pass.md:2996`. With T1 = 53.78 ms (`docs/BRIEF.md:73`), d = 0.075. A suffix or n-gram drafter has d ≈ 0.
- s: the host-lane term is 13.5 ms of 59.0 ms (row 30, `docs/phase-4/01-forward-pass.md:2965`; 13.4 ms at `:3144`), so s = 0.23. That counts only the host lane. GPU-resident expert reads also scale with the union, so 0.23 is a floor. s = 1 is the worst case.

Break-even acceptance p* (derived). The last column is S at p=0.6, k=3. The s=0.23 rows are optimistic lower bounds on p*, since s is a floor.

| Cost model | k=1 | k=2 | k=3 | k=4 | S at p=0.6, k=3 |
|---|---|---|---|---|---|
| m=0.95 (doc 02 as written), d=0.07 | none | none | none | none | 0.54× |
| m=0.40 (no union), d=0.07 | 0.47 | 0.59 | 0.67 | 0.72 | 0.90× |
| m=0.25 (no union), d=0.07 | 0.32 | 0.44 | 0.53 | 0.59 | 1.11× |
| s=1 (union), d=0.07 | none | none | none | none | 0.53× |
| s=0.5 (union), d=0.07 | 0.56 | 0.67 | 0.73 | 0.77 | 0.82× |
| s=0.23 (union), d=0.07 | 0.30 | 0.41 | 0.50 | 0.56 | 1.16× |
| s=0.23 (union), d=0 (suffix) | 0.23 | 0.34 | 0.42 | 0.48 | 1.31× |
| s=1 (union), d=0 (suffix) | 0.98 | 0.98 | 0.98 | 0.98 | 0.56× |

The "m" rows fix the marginal row cost, so r = 1 + k(m + d) (doc 02 §6's table uses m=0.40 and 0.25 with no draft cost; here d is added). The "s" rows let the marginal cost follow the union; the m=0.95 row matches s≈1, since D(4)/10 = 3.88 ≈ 1 + 3 × 0.96. Perfect acceptance (p=1, d=0) gives S = (k+1)/r: 2.41× at k=3 and 2.67× at k=4 for s=0.23 (derived). The k=4 figure is an illustration, not a cap. At k=7, r = 2.49 and S = 3.2 (derived), and at k=30 S is 5.03 (derived). The bound that holds at any k is S < 1/(1 − p), since r ≥ 1, so 5× needs p > 0.8. The recorded acceptances are 0.53–0.65 (doc 02, secondhand). At p=0.65 the best suffix-drafter k is 2, at 1.43× (Y = 2.07, r = 1.45; derived, optimistic). At s=1 the cap at p=1 is about 1.03–1.04× (k=3, 4). The model is a bound, not a predictor: arXiv 2506.20675 §8.3 says analytic MoE timing "would not work".

## 4. GDN state rollback

- Per GDN block: 128 × 48 × 128 × 4 B recurrent state + 3 × 10,240 × 4 B conv history = 3,268,608 B (`crates/breccia-engine/src/placement.rs:309–310`; `crates/breccia-engine/src/config.rs:39` gives the 10,240 conv channels). Over 36 GDN blocks: 117,669,888 B = 112.2 MiB (derived).
- Full-state copy: 117.67 MB at 576 GB/s = 0.204 ms, or 0.0035 T1 (derived). `docs/BRIEF.md:123` gives 810 GB/s for BF16 GEMV, which would make it 0.145 ms. The two DRAM figures conflict within the same doc; this doc uses the lower one.
- Snapshots: k+1 per-token snapshots cost 224, 337, 449 and 561 MiB for k=1..4 (derived). The memory per slot is the main cost, not the copy time.
- Options: (a) a per-token snapshot ring; (b) snapshot before verify, then replay the accepted tokens (replay cost not measured; UNVERIFIED); (c) keep only the state at the accepted prefix. Breccia's layout is "deliberately snapshot-friendly" (`docs/phase-3/06-attention-gdn.md:504–507`).
- llama.cpp at the pinned SHA: `src/llama-memory-recurrent.cpp:186–187` says recurrent models "can't have a state partially erased at the end". `n_rs_seq` sizes the recurrent cells (`:28`, `:36`, `:102`, `:126–127`). An unpinned master read describes a per-token snapshot index bounded by n_rs_seq (about `:197–212`); UNVERIFIED at this SHA. `common/speculative.cpp:941` has `need_boundary_stash()`; its body was not read.
- vLLM: the Qwen3-Next recipe runs `qwen3_next_mtp` with 2 tokens. No rollback mechanism was found; UNVERIFIED.
- SGLang: `--mamba-radix-cache-strategy` (Qwen3-Next cookbook §3.2). `extra_buffer` "trades higher mamba state memory for better throughput". That is prefix caching of states, not speculative rollback. NEXTN flags are in §4.2.4.
- A TensorRT-Edge-LLM replay-of-accepted-tokens approach appeared in a search snippet and was not fetched. Secondary only.

## 5. Prior art (primary sources; numbers from abstracts unless marked)

- **llama.cpp MTP.** PR #22673 "llama + spec: MTP Support" (created 2026-05-04, merged 2026-05-16). At SHA d05706b4f32a545814856e98ca518292585443b9: `common/speculative.cpp:78` registers `draft-mtp`; `:1505–1506` name "step35" chain heads and "a single trained MTP head" for qwen35 and qwen35moe. The file has no `glm` or `qwen3next` string. Architectures: `src/llama-arch.cpp:41` (qwen3next), `:94` (glm4moe), generic NextN key `:240`, tensors `:675–683`; `src/llama-model.cpp:1411` has a generic NextN read. A Qwen3-Next or GLM GGUF with NextN tensors may load, but no driver mode names it. UNVERIFIED whether it runs as MTP. The flag is `--spec-type` (`common/arg.cpp:4330`), whose list includes `draft-mtp`; `--draft` (`:4468`) is the draft-length alias.
- **Qwen3-Next model card** (MTP paragraph): MTP is "not generally available in Hugging Face Transformers". SGLang NEXTN takes 3 steps, top-k 1, 4 draft tokens; vLLM `qwen3_next_mtp` takes 2 tokens. No acceptance or speedup figure.
- **MoE speculation.** arXiv 2506.20675 (Utility-Driven Speculative Decoding for Mixture-of-Experts). §3: vLLM v0.6.6, n-gram speculation on five MoE models, EAGLE only on Mixtral. Abstract: verification "increases data movement and verification time by 2–3x"; slowdowns up to 1.5x (the Figure 4 caption says 2x). §2.4: uniform random-routing count (K=7 Mixtral ≈ 3.5× theoretical data movement). Abstract: Cascade "improves throughput by 7–14%" (§5.6, Figure 13: 7–15%).
- **Layer skip (self-speculation).** arXiv 2404.16710 LayerSkip: up to 2.16× (CNN/DM), 1.82× (coding), 2.0× (TOPv2). arXiv 2410.06916 SWIFT (on-the-fly layer skip, no separate draft model): 1.3×–1.6×. For Breccia a skip-draft runs a subset of the 48 blocks, so it reads only those blocks' experts and needs no drafter weights. The cost is acceptance; no MoE acceptance figure appears in these abstracts.
- **EAGLE.** arXiv 2401.15077: 2.7×–3.5× latency on LLaMA2-Chat 70B (abstract). Not MoE.
- **Offload plus speculation.** arXiv 2505.10259 SpecOffload: 2.54× inference throughput and 4.49× GPU utilization over the best baseline (abstract). Not MoE-specific per the abstract.
- **Pipelining.** arXiv 2411.11217 MoE-Lightning: CGOPipe CPU-GPU-I/O pipelining with paged weights; 10.3× peak throughput over offloading systems, Mixtral 8x7B on one 16 GB T4 (abstract). The closest prior art to a "phase-shifted" CPU/GPU schedule. The phrase "phase-shifted" appears in no fetched source; UNVERIFIED as a named technique.
- **Dynamic routing.** arXiv 2403.07652 (Harder Tasks Need More Experts): +0.7% over top-2 with under 90% activated parameters (abstract; models not named there). arXiv 2402.14800 (Not All Experts are Equal): no numbers in the abstract.

## 6. Adaptive top-k and a CPU-only gate

- **Accuracy risk.** Fewer experts changes every block's output. The only aggregate figure here is arXiv 2403.07652's +0.7% (other routing, other models). Breccia has no accuracy measurement. Its harnesses measure speed only: `harnesses/bench_attrib.sh` (header lines 1–12), `harnesses/bench_decode.py`, `harnesses/moe_cache_ab.sh` (lines 2–4).
- **Cost of one expert less.** Each token loses 10% of its expert bytes per expert removed. At s=0.23 that is 2.3% of T1 per expert removed (derived).
- **Gate from existing pieces.** `tools/perplexity/perplexity.cpp:175` (`kl_divergence_result`) and `:191` (`log_softmax` against base log-probabilities) compute KL against saved base logits. `common/arg.cpp:2957` (`--override-kv`) and the `expert_used_count` key at `src/llama-arch.cpp:227` allow a CPU run at reduced k. UNVERIFIED: that Breccia's model loads in llama.cpp at this SHA, and that the MoE graph honors the override. Breccia has no CPU reference forward (UNVERIFIED).
- **Route trace.** `crates/breccia-engine/src/graph.rs:2432` (`BRECCIA_TRACE_ROUTE`) prints distinct experts per block in the prefill merged-read path. It measures union, not accuracy, and needs a GPU run.

## 7. RPC and the deferred mesh

- llama.cpp `tools/rpc/README.md` (master; not pinned to the SHA): the server is "insecure" and must not run on an open network (line 5); it exposes every accelerator device on the host (line 36); local cache with `-c` (line 88); RDMA with TCP fallback (line 98).
- An RPC split puts a socket round trip at each cross-host block boundary. Breccia's 48-block chain pays no such cost on one host. The size of that cost is UNVERIFIED - requires measurement. Nothing in this round supports RPC as a speedup; it is a capacity option.
- The tailnet mesh is deferred. This doc names no hosts or addresses.

## 8. Sources, unverified items and skipped reads

- **Sources (public).** llama.cpp at SHA d05706b4f32a545814856e98ca518292585443b9; PR #22673 (github.com/ggml-org/llama.cpp/pull/22673); tools/rpc/README.md at master; Qwen3-Next model card (huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct); vLLM Qwen3-Next recipe (docs.vllm.ai, projects/recipes, Qwen/Qwen3-Next); SGLang Qwen3-Next cookbook (docs.sglang.io); arXiv IDs as cited above.
- **UNVERIFIED.** Breccia GDN rollback replay cost; whether Qwen3-Next or GLM NextN tensors run as MTP at the pinned SHA; whether `--spec-type draft-mtp` takes a Qwen3-Next or GLM GGUF (§5); whether Breccia's GGUF loads in llama.cpp and honors `--override-kv`; whether the MTP expert set fits in VRAM; s and D(k) on hardware; the MI50 per-row cost behind the flat remainder (§3); the KL bound; "phase-shifted" pipelining as a named technique; the vLLM rollback mechanism; the RPC round-trip cost.
- **Corrections.** Doc 02 §6 yield label (2.31 implies p≈0.64, text says 0.6). `docs/BRIEF.md` gives 576 GB/s (line 16) and 810 GB/s (line 123) for the same DRAM.
- **Not read.** Basalt and capy sources (their figures are quoted from doc 02); the `draft-mtp` implementation body beyond the cited lines; `harnesses/vk-probe`; LayerSkip, SWIFT, EAGLE, SpecOffload and MoE-Lightning full texts (abstracts only); the RPC README at the pinned SHA; vLLM and SGLang source (docs only); research docs 02, 09, 18 and 19 beyond the grep hits above; TensorRT-Edge-LLM.


## 9. Top 3

1. **Pin the expert union on the card.** What: 4-token windows through `BRECCIA_TRACE_ROUTE` (`crates/breccia-engine/src/graph.rs:2432`) over long text; no MTP forward is needed. Moves: s and D(4), the two inputs §3 cannot fix. Evidence: §3; the trace hook exists. Gate: measured per-block distinct experts over 4-token windows, fed into the §3 table. A drafter is in scope only if its measured acceptance clears the p* for the measured s.
2. **Suffix or n-gram drafter first, with a real multi-row verify and GDN rollback.** What: draft from the token history, with no drafter weights and no draft expert reads. Moves: d (0.07 → 0) and the drafter's reads; at s=0.23 and k=3 the threshold is p*=0.42 (§3; an optimistic lower bound, since s=0.23 is a floor). Evidence: §3; doc 02 §6 names the suffix drafter; arXiv 2506.20675 ran n-gram drafts on MoE models (acceptance numbers not extracted). Gate: measured acceptance on Breccia workloads ≥ p* at the measured s, and speedup > 1 after verify and rollback cost. Multi-row verify and rollback do not exist, so the build is the real cost.
3. **CPU-only accuracy gate before any adaptive-k decode path.** What: KL to base logits at reduced expert count on a held-out set, using the perplexity tool (§6). Moves: expert reads per token; at s=0.23 each removed expert is 2.3% of T1 (derived). Evidence: §6; arXiv 2403.07652 (+0.7%, abstract). Gate: a KL bound set on the GPU qualification card before any decode change. The bound value is UNVERIFIED; the repo has none. Needs the arch to load on CPU, or a Breccia CPU reference, which does not exist.

## 10. Overlaps with other docs

- **Doc 02** (`docs/phase-4/02-untried-levers.md` §6 and the §8 lane numbers). This round re-derives §6's break-even and finds its 0.60× yield label inconsistent (2.31 implies p≈0.64; the text says 0.6). It adds s and the union. It does not reorder §6's ranking.
- **Doc 09** (`docs/research/09-zero-copy-host-memory-and-ggml-vulkan.md`): host memory and transfer. A grep found no speculative content. The overlap is the host-lane copy cost, which this doc does not re-cost.
- **Doc 18** (`docs/research/18-flashinfer-and-ft-moe.md`): top-k primitives for sampling and routing. Overlaps §6 on the top-k topic only; its numbers are not reused.
- **Doc 19** (`docs/research/19-mlx-and-llm-in-a-flash.md`): expert offload and locality. Overlaps §3, since the union is a locality question. Its locality numbers were not re-read and are not reused here.
- **Doc 20** (`docs/research/20-deepseek-stack.md`): carries MTP as "doc 02 #6" with no new figure (its row 13). This round supplies the break-even that doc 20 defers. Its top-k note (QSA) is unrelated to expert routing.
- Also `docs/research/02-vulkan-followups-and-powerinfer.md:475` names discrete top-k routing as Breccia's sparsity source. That is relevant to §6; the rest of that doc was not read.
