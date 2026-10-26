# Forward-pass integration design (M2 target)

Blueprint for the first end-to-end breccia token. Inputs: `docs/specs/qwen4exp-architecture.md` (op semantics, shapes), `docs/phase-3/01–04` (kernels + measured rates), `docs/phase-2/02` (loader APIs), `docs/specs/fork-flags-semantics.md` (placement parity).

## 1. What a token costs today (parity baseline)

From the measured production configuration (`docs/phase-0/01`): 20.79 tok/s = 48.1 ms/token, split as
- **GPU lane** (RX 6950 XT + Radeon VII): 13,577 MiB of dense/attention weights + 13 expert blocks; decode GEMV at 271–470 GB/s per type (measured)
- **CPU lane**: ~35 blocks of host-mapped experts, but only the **top-10 of 512 per layer** are touched per token: 42.9 GB of expert weights / 48 layers / 512 experts × 10 × 48 ≈ **0.84 GB read per token** (the 52 GB figure is the whole model, not the per-token working set), plus PLE row gathers from the 28.8 GB lazily-mapped table
- MoE: 10 of 512 experts per layer per token (top-10), shared expert always

The engine does not have to beat 48.1 ms by cleverness in M2 — it has to **match it first** (parity), then apply the measured levers (expert batching 3.4×, route-tail skipping, cache).

## 2. Layer graph (all 48 blocks)

```
token ids ──▶ emb gather (token_embd IQ3_S) ──▶ R [2560, 4, T]   (4 hyper-connection copies)
                                                 │
for L in 0..48:                                  ▼
  HC_READ ──▶ sublayer:                          (mixer: low-rank 320)
    if L % 4 != 3:  GDN/SSM  (36 layers)         state (S=128, h_v=48), conv hist 3
    else:           QSA      (12 layers)         KV (2 heads × 256) + lightning indexer (top-k 2048, compress 4)
  HC_WRITE ──▶ HC_READ ──▶ MoE:
    router GEMV (BF16 [2560,512]) → softmax → top-10 (2^-14 renorm clamp)
    ──▶ batched expert GEMV ×10 (IQ2/IQ3/IQ4 mix) + shared expert (SwiGLU)
  HC_WRITE
                                                 │
output HC mixer (= final norm) ──▶ lm_head (Q5_K / IQ3_S) ──▶ logits [320001536]
PLE (blk.1 only): hash(tokens) → row gather from IQ4_NL table → signed-sqrt gate → conv(3) → add
```

Every arrow is a kernel that already exists in `breccia-kernels` (1=GEMV, 2=batch, 04=ops) or is still pending (GDN step, QSA attention + indexer). The graph crate does orchestration only — no math.

## 3. Buffer plan (per device)

| Buffer | Where | Size @ T=1 | Notes |
|---|---|---|---|
| Dense weights (attn/qkv/o, ssm_*, hc_*, indexer, norms, router, shared expert, emb, head) | GPU, split by profile plan | ~13.6 GB (measured placement) | static, uploaded once from mmap |
| Expert weights | host (default) + per-block GPU overrides | 52 GB host-mapped | fork parity: `-cmoe` default, `-ot` pins blocks to GPU |
| PLE table | host lazy-mapped | 28.8 GB (never resident) | row gather per token; the lazy-mode equivalent |
| KV (12 QSA layers) | VRAM | 12 × 2 heads × 256 × K+V × 2 B ≈ 24 KiB/token | f16 first; quantised body later (M5) |
| GDN state (36 layers) | VRAM | 3.0 MiB/layer ≈ 110 MiB total | ctx-free; conv history 3 |
| Activations | VRAM | 2560 × 4 × 4 copies ≈ 41 KiB + logits scratch | tiny; the head dominates |
| Expert slot ring | VRAM (RX) | e.g. 8 × 16 MiB = 128 MiB | `breccia_vk::transfer::TransferRing`; DMA hidden under compute (measured 90%+ on RX) |

## 4. Dispatch order per token (decode)

1. emb gather (1 dispatch) → `R`
2. per layer: HC_READ (1) → attention (GDN: conv+delta-rule ~6 dispatches; QSA: qkv GEMV + RoPE + indexer + FA + gate ~8) → HC_WRITE (1) → MoE: router GEMV (1) + softmax/top-k (1) + **batched expert GEMV ×10 in one dispatch** (measured 3.4× vs loop) + shared SwiGLU (1) → HC_WRITE (1)
3. output HC mixer (1) → lm_head GEMV (1) → sample

Sync policy: one command buffer per layer, timeline-semaphore handoff between the two GPUs only at the layer-split boundary; host never round-trips mid-token. Expert uploads (if a GPU override block misses) go through the ring on the transfer queue concurrently with the previous layer's compute.

## 5. Validation strategy (definition of done for M2)

1. **Numerical parity, not vibes**: fixed prompt (e.g. 32 tokens) → compare logits against the production HIP server (`qwen-dual`) for the same tokens; require top-1 agreement on ≥95% of steps and max |Δlogit| documented per step. Any mismatch is traced to the specific kernel (per-op conformance already exists).
2. **Per-op golden vectors**: every kernel already has CPU-reference conformance; the graph adds only routing/ordering — keep a debug mode that dumps per-layer activations for diffing against a reference implementation (`strata` or the fork's debug output if available).
3. **Speed measurement**: tokens/sec on the same prompt and context as the 20.79 baseline; report P50 decode over ≥3 runs of 256 tokens.
4. Safety: every run `timeout`, MI50-first for new kernels (AGENTS.md).

## 6. Remaining kernel work before the graph

| Kernel | Status |
|---|---|
| GEMV + dequant, 14 types | ✅ measured |
| Batched expert GEMV | ✅ 3.4× (top-10 shape) |
| RMSNorm / IMRoPE / SwiGLU / softmax / router top-k / emb & PLE gather / HC mixer | in progress (`docs/phase-3/04`) |
| GDN step (conv + delta rule + gating) | pending — next kernel block |
| QSA attention (FA + lightning indexer scoring + output gate) | pending — next kernel block |
| lm_head + sampler | after graph skeleton (GEMV exists; add argmax/sampling) |

## 7. Open risks

- **KV layout for QSA** must match the 12-layer paged scheme chosen for M5 (24 KiB/token f16) — decide now, not later.
- **HC mixer algebra** has documented ambiguity in the spec (§ open questions); the ops report records the chosen interpretation — verify against reference activations before trusting end-to-end numbers.
- **CPU expert lane**: ~0.84 GB/token (top-10 × 48 layers), i.e. the 20.79 tok/s baseline sustains ~17.5 GB/s of host-side expert reads. The graph needs a vectorised host GEMV over mmap'd quants (AVX2), which should comfortably exceed that; `--moe-cache-mib > 0` on the production side remains a cheap external A/B.
