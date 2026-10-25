# Model artifacts inventory — Qwen3.8-Flash-Next GSQ-RCO (this machine)

Inventory of every weight artifact a from-scratch inference engine has to parse, with the exact
byte-level structures as measured on 2026-10-25. Everything below was read directly from the files
(GGUF headers parsed with `gguf-py` from `/home/korphaus/llama.cpp-qwen4exp-mmap/gguf-py`, plus
hand-rolled `struct` parses for cross-checks); nothing here is copied from a README without
verification. Where a number is a claim of a *different* release (Q2_0, Coder) it is marked as such.

Re-verification recipe used throughout:

```bash
PYTHONPATH=/home/korphaus/llama.cpp-qwen4exp-mmap/gguf-py python3 -c "
from gguf import GGUFReader
r = GGUFReader('<shard>')
print(len(r.tensors), r.data_offset, r.fields.keys())
for t in r.tensors[:5]: print(t.name, t.shape, t.tensor_type.name, t.data_offset, t.n_bytes)"
```

---

## 1. Where everything lives

| Path | What it is |
| --- | --- |
| `/srv/strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000{1,2}-of-00002.gguf` | The model: 2 split GGUF shards (43.81 + 26.82 GiB) |
| `/srv/strata-data/models/IQ3_XXS/*.gguf.done` | 16-byte download-completion markers, content is an ISO timestamp (`2026-10-11 01:42`) |
| `/srv/strata-data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf` | Vision projector (`clip` arch), BF16/F32 |
| `/srv/strata-data/models/Q2_0` | **symlink** → `/mnt/agent-storage-b/ai-dev/ai-runtime/home/korphaus/models/qwen-dual/Q2_0` (see §7) |
| `/srv/strata-data/models/tensor-allocation/*.rco-allocation.txt` | Per-tensor RCO quant allocation record (Q2_0 release) — see §3 |
| `/srv/strata-data/packs/iq3_xxs/` | Strata "native pack": `index.txt`, `dense.bin`, `experts.bin`, `native_experts.txt`, JSON sidecars, `tokenizer/` — see §4 |
| `/srv/strata-data/mtp/` | Multi-Token-Prediction (draft layer) artifacts — see §5 |
| `/srv/strata-data/mtp-unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf` | A second, unsloth-derived MTP GGUF (4,137,429,120 B), not wired into the config |
| `/home/korphaus/projects/strata` | Reference engine (C++/HIP) that consumes the pack + MTP files |
| `/home/korphaus/llama.cpp-qwen4exp-mmap` | llama.cpp fork with the `qwen4exp` architecture (source tree *and* `build-hip-mmap/` build dir in one place) |

### Disk sizes

| File / dir | Bytes | GiB / MiB |
| --- | ---: | ---: |
| `IQ3_XXS/…-00001-of-00002.gguf` | 47,039,860,096 | 43.81 GiB |
| `IQ3_XXS/…-00002-of-00002.gguf` | 28,800,138,432 | 26.82 GiB |
| `IQ3_XXS/` dir total (incl. 2× `.done`) | 75,839,998,560 | 70.63 GiB |
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf` | 907,543,008 | 865.5 MiB |
| `tensor-allocation/Qwen3.8-…-Q2_0-00001-of-00002.rco-allocation.txt` | 39,062 | 38 KiB |
| `packs/iq3_xxs/dense.bin` | 1,538,035,200 | 1.43 GiB |
| `packs/iq3_xxs/experts.bin` | 42,912,972,800 | 39.97 GiB |
| `packs/iq3_xxs/index.txt` | 93,687 | 91 KiB |
| `packs/iq3_xxs/native_experts.txt` | 3,296 | 3 KiB |
| `packs/iq3_xxs/conversions.json` | 310 | — |
| `packs/iq3_xxs/experts.bin.src.json` | 377 | — |
| `packs/iq3_xxs/tokenizer/` (vocab/merges/types/template/meta) | 9,845,757 | 9.4 MiB |
| `packs/iq3_xxs/` dir total | 44,460,951,427 | 41.41 GiB |
| `mtp/mtp-q2_0.gguf` | 889,017,216 | 847.8 MiB |
| `mtp/mtp-manifest.json` | 11,643 | — |
| `mtp/mtp-inventory.json` / `.md` | 7,514 / 2,437 | — |
| `mtp/mtp-q2_0.gguf.report.json` | 3,589 | — |
| `mtp/tensors/` (31 raw `.bin`) | 5,214,301,696 | 4.86 GiB |
| `mtp/rt/dense.bin` | 116,099,072 | 110.7 MiB |
| `mtp/rt/experts.bin` | 707,788,800 | 675.0 MiB |
| `mtp/rt/draft_vocab.bin` | 425,196 | 415 KiB |
| `mtp/rt/dense.txt` | 1,851 | — |
| `mtp/` dir total | 6,927,659,014 | 6.45 GiB |
| Q2_0 shard 1 (on `/mnt/agent-storage-b`) | 37,623,740,192 | 35.04 GiB |
| `mtp-unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf` | 4,137,429,120 | 3.85 GiB |
| **Everything above, total** | ~1.70 × 10¹¹ | ~158 GiB |

---

## 2. The GGUF shards (IQ3_XXS)

### 2.1 Container facts (measured)

| | shard 1 (`-00001-of-00002`) | shard 2 (`-00002-of-00002`) | mmproj |
| --- | ---: | ---: | ---: |
| magic / GGUF version | `GGUF` / 3 | `GGUF` / 3 | `GGUF` / 3 |
| KV pairs | **67** | 3 | 31 |
| tensors | **1223** | **1** | 334 |
| `data_offset` (first data byte) | 11,024,256 | 192 | 20,000 |
| max(`data_offset + n_bytes`) | 47,039,860,096 = **file size** | 28,800,138,432 = **file size** | 907,543,008 = file size |
| trailing bytes | 0 | 0 | 0 |
| alignment | 32 B (GGUF default; 11,024,256 % 32 == 0) | 32 B (header ends 173, padded to 192) | 32 B |

* Every KV pair, every tensor name/type/shape/offset below was read out of the file itself.
* The tensor payload of shard 1 ends **exactly** at EOF (no padding, no truncation), and the sum of
  the 144 expert tensors' bytes is **42,912,972,800** — byte-for-byte the size of the pack's
  `experts.bin` (§4).

### 2.2 Split rules

```
split.no            = 0        (shard 1) / 1 (shard 2)
split.count         = 2        (both shards)
split.tensors.count = 1224     (both shards: 1223 in shard 1 + 1 in shard 2)
```

Shard 1 carries **all** model metadata (architecture, hparams, tokenizer, chat template). Shard 2
carries only `split.*` and a single tensor: `per_layer_token_embd.weight` (the PLE table). A loader
must open shard 1 for KV and both shards for the tensor table.

### 2.3 KV pairs (all 67, verified)

`general.*`

| key | value |
| --- | --- |
| `general.architecture` | **`qwen4exp`** |
| `general.type` | `model` |
| `general.name` | `Qwen3.8-Flash-Next GSQ-RCO` |
| `general.description` | `A Preview of the Qwen4 Architecture` |
| `general.size_label` | `512x56B` |
| `general.file_type` | 23 (= `MOSTLY_IQ3_XXS`) |
| `general.quantization_version` | 2 |
| `general.quantized_by` | `ISTA DASLab` |
| `general.repo_url` | `https://huggingface.co/ISTA-DASLab` |
| `general.tags` | `["gsq-rco", "ista-daslab"]` |
| `general.sampling.{top_k,top_p,temp}` | 20 / 0.95 / 1.0 |

`qwen4exp.*` (every key present — **all items from the prior investigation are confirmed**, marked ✓):

| key | value | |
| --- | --- | --- |
| `block_count` | 48 | ✓ |
| `context_length` | 262144 | ✓ |
| `embedding_length` | 2560 | ✓ |
| `attention.head_count` | 24 | ✓ |
| `attention.head_count_kv` | 2 | ✓ |
| `attention.key_length` | 256 | ✓ (`klen 256`) |
| `attention.value_length` | 256 | ✓ (`vlen 256`) |
| `attention.layer_norm_rms_epsilon` | 9.999999974752427e-07 | ✓ |
| `attention.indexer.head_count` | 4 | ✓ |
| `attention.indexer.key_length` | 128 | ✓ |
| `attention.indexer.top_k` | 2048 | ✓ |
| `attention.compress_ratios` | int32[48]: `0,0,0,4` repeating (4 at indices 3,7,…,47 → the 12 full-attention layers) | ✓ (new) |
| `rope.dimension_sections` | int32[4] = `[11, 11, 10, 0]` | (new) |
| `rope.dimension_count` | 64 | (new) |
| `rope.freq_base` | 10,000,000.0 | (new) |
| `expert_count` | 512 | ✓ |
| `expert_used_count` | 10 | ✓ |
| `expert_feed_forward_length` | 640 | (new) |
| `expert_shared_feed_forward_length` | 640 | (new) |
| `ssm.conv_kernel` | 4 | ✓ |
| `ssm.state_size` | 128 | ✓ |
| `ssm.group_count` | 16 | ✓ |
| `ssm.time_step_rank` | 48 | ✓ |
| `ssm.inner_size` | 6144 | ✓ |
| `full_attention_interval` | 4 | ✓ |
| `hyper_connection.count` | 4 | ✓ |
| `hyper_connection.low_rank` | 320 | ✓ |
| `ple.layers` | int32[1] = `[1]` | (new — PLE applies at layer 1 only) |
| `ple.ngram_size` | 3 | ✓ |
| `ple.heads_per_ngram` | 8 | ✓ |
| `ple.conv_kernel` | 4 | (new) |
| `ple.eos_token_id` | 248044 | (new) |
| `ple.image_token_id` | 248056 | (new) |
| `ple.layer_multipliers` | uint64[3] = `[23703573157769, 20109073645365, 8052911324071]` (bit-packed) | (new) |
| `ple.head_offsets` | uint64[16] = `0, 20000003, 40000026, …, 300001275` (cumulative) | (new) |
| `ple.head_vocab_sizes` | uint64[16] = `20000003, 20000023, …, 20000171`; sums to **320,001,446** | (new) |
| `embedding_length_per_layer_input` | 160 | (new — = PLE table `ne0`) |

Note: `sum(ple.head_vocab_sizes) = 320,001,446`, while the shard-2 PLE tensor has `ne1 =
320,001,536` (90 more). Use the tensor's own shape as authoritative; the offsets/sizes describe
head slicing inside it.

`tokenizer.*`: `model=gpt2`, `pre=qwen35`, 248,320 tokens, 247,587 merges, `eos=248046`,
`bos=248044`, `padding=248044`, `add_bos_token=false`, `chat_template` = 10,279-char Jinja2
(Qwen3-style, Unsloth fixes, image/video placeholders, `reasoning_effort` xhigh/medium/low).

`quantize.*`: `imatrix.file=imatrix-qwen3.8-flash-next.gguf`, `dataset=calib_qwen3.8-flash-next.tokids`,
`entries_count=926`, `chunks_count=1000`.

### 2.4 Tensor naming scheme (llama.cpp-style renames)

GGUF names (this is what a loader sees; the HF/strata names in parentheses):

```
token_embd.weight                          (model.embed_tokens)
output.weight                              (lm_head)
output_hc_{down,norm,up}.weight            (hyper-connection on the output/residual)
blk.N.attn_qkv.weight                      fused Q, K, V — 36 SSM blocks   (self_attn.q/k/v_proj)
blk.N.attn_q.weight / attn_k.weight / attn_v.weight / attn_output.weight
                                           separate projections — the 12 full-attention blocks
blk.N.attn_{q,k}_norm.weight               (self_attn.q/k_norm)
blk.N.attn_gate.weight                     attention gate              (36 SSM blocks only)
blk.N.ssm_{a, dt.bias, alpha.weight, beta.weight, conv1d.weight, norm.weight, out.weight}
                                           hybrid Mamba/SSM params      (36 blocks only)
blk.N.ffn_gate_exps.weight                 routed experts, gate  [ne0=n_embd, ne1=n_ff, ne2=512]
blk.N.ffn_up_exps.weight                   routed experts, up
blk.N.ffn_down_exps.weight                 routed experts, down  [ne0=n_ff, ne1=n_embd, ne2=512]
blk.N.ffn_gate_inp.weight                  router / gate input   [2560, 512]
blk.N.ffn_{gate,up,down}_shexp.weight      shared expert projections   (shared_expert.*)
blk.N.ffn_gate_inp_shexp.weight            shared-expert router scalar [2560]
blk.N.hc_attn_{down,up,inject,norm}.weight hyper-connection, attention branch
blk.N.hc_ffn_{down,up,inject,norm}.weight  hyper-connection, FFN branch
blk.N.indexer_{...} → blk.N.indexer.{q,k}_{proj,norm}.weight + blk.N.indexer.q_proj/k_proj
                                           sparse-attention indexer (12 full-attention blocks only)
blk.N.ple_key.weight / ple_value.weight / ple_norm_{key,query,conv}.weight / ple_conv1d.weight
                                           PLE module — exists ONLY in blk.1
per_layer_token_embd.weight                the PLE token table — shard 2 only, [160, 320001536]
```

Two block families (verified by tensor presence):

* **36 SSM/hybrid blocks** — all except 3,7,11,…,47: fused `attn_qkv`, `attn_gate`, full `ssm_*`
  set. Blocks: 0,1,2,4,5,6,8,…,46.
* **12 full-attention blocks** — 3,7,11,15,19,23,27,31,35,39,43,47: separate q/k/v/o projections,
  `attn_q/k_norm`, and the `indexer.*` tensors (`full_attention_interval = 4`).

### 2.5 Quantization per tensor group (measured histogram, shard 1, 1223 tensors)

| type (ggml enum id) | count |
| --- | ---: |
| BF16 (30) | 484 |
| F32 (0) | 292 |
| Q6_K (14) | 71 |
| IQ4_XS (23) | 67 |
| IQ4_NL (20) | 58 |
| Q4_K (12) | 45 |
| Q2_0 (42) | 38 |
| IQ2_XS (17) | 20 |
| IQ2_S (22) | 20 |
| IQ2_XXS (16) | 18 |
| Q5_K (13) | 15 |
| IQ3_XXS (18) | 12 |
| IQ3_S (21) | 82 |
| F16 (1) | 1 |

Group by group (quant set is **mixed per tensor** — this is the RCO allocation; the release name
"IQ3_XXS" is only the whole-file `general.file_type`, *not* a per-tensor guarantee):

| group | n | shapes (ne0, ne1[, ne2]) | quant types |
| --- | ---: | --- | --- |
| `output.weight` | 1 | 2560 × 248320 | Q5_K |
| `token_embd.weight` | 1 | 2560 × 248320 | IQ3_S |
| `hc_{attn,ffn}_{down,up,inject}` + `output_hc_{down,up}` (290 total) | 6×48 + 2 | 10240×320, 320×10240, 10240×4 | BF16 (always) |
| `output_hc_norm`, `hc_{attn,ffn}_norm` (97 total) | 2×48 + 1 | 10240 | F32 (always) |
| `attn_qkv` (36 SSM blocks) | 36 | 2560 × 10240 | IQ4_XS 11, Q6_K 14, Q4_K 8, Q5_K 3 |
| `attn_gate` (36) | 36 | 2560 × 6144 | IQ3_S 10, Q6_K 5, IQ4_XS 11, Q4_K 7, Q5_K 3 |
| `attn_q` (12 full) | 12 | 2560 × 12288 | IQ4_XS 4, Q4_K 3, Q6_K 4, IQ3_S 1 |
| `attn_k` / `attn_v` (12) | 12 each | 2560 × 512 | mixed IQ4_XS/Q4_K/Q6_K/Q5_K/IQ3_S |
| `attn_output` (12) | 12 | 6144 × 2560 | Q6_K 10, Q5_K 1, Q4_K 1 |
| `attn_{q,k}_norm` (12) | 12 each | 256 | F32 |
| `ssm_out` (36) | 36 | 6144 × 2560 | Q6_K 19, IQ4_XS 6, Q4_K 4, Q5_K 4, IQ3_S 3 |
| `ssm_{conv1d, dt.bias, a, norm}` (36) | 36 each | 4×10240 / 48 / 128 | F32 (always) |
| `ssm_{alpha,beta}` (36) | 36 each | 2560 × 48 | BF16 (always) |
| `ffn_gate_exps` / `ffn_up_exps` | 48 each | 2560 × 640 × 512 | per layer one of IQ2_XXS 9, IQ2_XS 10, IQ3_XXS 6, IQ3_S 13, IQ2_S 10 |
| `ffn_down_exps` | 48 | 640 × 2560 × 512 | per layer: IQ4_NL 18 layers, Q2_0 30 layers |
| `ffn_gate_inp` (router) | 48 | 2560 × 512 | BF16 |
| `ffn_gate_inp_shexp` | 48 | 2560 | BF16 |
| `ffn_{gate,up,down}_shexp` (shared expert) | 48 each | 2560×640 / 640×2560 | IQ3_S/Q4_K/Q5_K/Q6_K/IQ4_XS; down: IQ4_NL 40, Q2_0 8 |
| `indexer.q_proj` / `k_proj` (12) | 12 each | 2560×512 / 2560×128 | BF16 (always) |
| `indexer.{q,k}_norm` (12) | 12 each | 128 | F32 |
| `ple_key`, `ple_value` (blk.1) | 1 each | 2560×10240, 2560×2560 | BF16 |
| `ple_conv1d` (blk.1) | 1 | 4 × 10240 | **F16** |
| `ple_norm_{key,query,conv}` (blk.1) | 3 | 10240 | F32 |
| `per_layer_token_embd` (shard 2) | 1 | 160 × 320001536 | **IQ4_NL** (28,800,138,240 B = 4.5 bpw) |

Practical rule: *float tensors are F32 (norms, SSM scalars), BF16 (hyper-connection, routers,
indexer projections, `ssm_alpha/beta`), F16 (`ple_conv1d`) — quantized tensors are per-tensor
allocated and must be looked up by name, never assumed from the release name.*

Block geometry of every type present (block = *elements, bytes*; byte counts cross-checked against
actual offset deltas in these files — the fork's enum lives at `ggml/include/ggml.h:390-433`, and
strata ships its own reference table at `include/strata/artifact/gguf_reader.hpp:122`
(`block_geometry()`, with the `Q2_0 = 64/18` proof at `:120`) plus a per-tensor size check in
`src/artifact/gguf_reader.cpp:30-58`):

| id | name | block (elems, bytes) | B/elem | bpw | tensors in shard 1 |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | F32 | (1, 4) | 4.000 | 32 | 292 |
| 1 | F16 | (1, 2) | 2.000 | 16 | 1 |
| 12 | Q4_K | (256, 144) | 0.5625 | 4.5 | 45 |
| 13 | Q5_K | (256, 176) | 0.6875 | 5.5 | 15 |
| 14 | Q6_K | (256, 210) | 0.8203 | 6.5625 | 71 |
| 16 | IQ2_XXS | (256, 66) | 0.2578 | 2.0625 | 18 |
| 17 | IQ2_XS | (256, 74) | 0.2891 | 2.3125 | 20 |
| 18 | IQ3_XXS | (256, 98) | 0.3828 | 3.0625 | 12 |
| 20 | IQ4_NL | (32, 18) | 0.5625 | 4.5 | 58 |
| 21 | IQ3_S | (256, 110) | 0.4297 | 3.4375 | 82 |
| 22 | IQ2_S | (256, 82) | 0.3203 | 2.5625 | 20 |
| 23 | IQ4_XS | (256, 136) | 0.5312 | 4.25 | 67 |
| 30 | BF16 | (1, 2) | 2.000 | 16 | 484 |
| 42 | **Q2_0** (fork-only) | **(64, 18)** | 0.28125 | **2.25** | 38 |

The Q2_0 *model* (§3) additionally uses `Q4_0` (2, 32/18), `Q5_0` (6, 32/22), `Q8_0` (8, 32/34) and
`Q3_K` (11, 256/110) — see the allocation file's type counts.

Type-code → name table (from `gguf-py/gguf/constants.py`, `GGMLQuantizationType`, in the fork —
note this fork adds `Q1_0=41`, `Q2_0=42`):

```
F32=0  F16=1  Q4_0=2  Q4_1=3  Q5_0=6  Q5_1=7  Q8_0=8  Q8_1=9
Q2_K=10 Q3_K=11 Q4_K=12 Q5_K=13 Q6_K=14 Q8_K=15
IQ2_XXS=16 IQ2_XS=17 IQ3_XXS=18 IQ1_S=19 IQ4_NL=20 IQ3_S=21 IQ2_S=22 IQ4_XS=23
I8=24 I16=25 I32=26 I64=27 F64=28 IQ1_M=29 BF16=30 TQ1_0=34 TQ2_0=35
MXFP4=39 NVFP4=40 Q1_0=41 Q2_0=42
```

`Q2_0` (42) is **not** ggml's legacy Q2_0: measured at 18 bytes per 64 values = **2.25 bpw**
(16 B 2-bit codes + 2 B fp16 scale per 64 elements). `IQ4_NL` here is 18 B per 32 values = 4.5 bpw.
An engine must implement the fork's block layouts, not upstream's, for type 42.

### 2.6 Shard 2

Single tensor `per_layer_token_embd.weight`, `ne = [160, 320001536]`, type IQ4_NL, `data_offset=192`,
`n_bytes = 28,800,138,240`, ends exactly at EOF. `160 = qwen4exp.embedding_length_per_layer_input`
(PLE row width), `320001536 ≈ Σ ple.head_vocab_sizes` (all 16 heads' token rows concatenated).
Strata passes this shard separately as `--ple-gguf`; `setup.py` locates it by scanning shards for a
tensor named `per_layer_token_embd.weight`.

---

## 3. `models/tensor-allocation/*.rco-allocation.txt`

Single file: `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.rco-allocation.txt` (39,062 B,
1,235 lines = 12 comment lines + 1,223 data lines).

**What it is:** the RCO (Rank/Cluster/Ordering — ISTA-DASLab's per-tensor quantization planner)
*allocation record* published with the release: it states, for every tensor of the **Q2_0** GGUF
shard 1, which ggml storage type that tensor was written with, plus enough header facts to prove
the file's own consistency. It is provenance/reproducibility documentation, not a runtime input —
see §7 for the (absence of) parser code.

Header (verbatim, all 12 lines):

```
# RCO per-tensor GGUF allocation
# Repository: ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
# GGUF: Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
# Source: https://huggingface.co/ISTA-DASLab/.../Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
# Target whole-file bpw: 2.40
# GGUF version: 3
# GGUF metadata entries: 73
# Tensor count: 1223
# Quant-type counts: BF16=483, F16=1, F32=292, IQ4_NL=7, IQ4_XS=56, Q2_0=202, Q3_K=91,
#                    Q4_0=16, Q4_K=38, Q5_0=7, Q5_K=16, Q6_K=12, Q8_0=2
# Format: <GGUF tensor name>: <GGML quantization/storage type>
# Includes fixed non-RCO tensors (e.g. F32/BF16) so the released GGUF is fully reproducible.
#
```

Body: one line per tensor, `name: TYPE`, in GGUF tensor order, e.g.

```
output.weight: Q5_K
token_embd.weight: Q3_K
blk.0.attn_qkv.weight: IQ4_XS
blk.0.ffn_down_exps.weight: Q2_0
blk.0.ffn_gate_exps.weight: Q2_0
blk.0.ffn_gate_inp.weight: BF16
blk.47.indexer.q_proj.weight: BF16
```

Verified: the quant-type counts sum to 1,223; 1,223 data lines + 12 comments = 1,235 = `wc -l`.
This file describes the **Q2_0** model (target 2.40 bpw, 73 KV entries) — a *different* allocation
from the IQ3_XXS shards in §2 (those have 67 KV entries and its own per-tensor mix). The `Coder`
release's copy additionally has a "section 2" that maps original→kept expert indices
(`bench/results/2026-09-28-coder/README.md:27`); **our copy has only the per-tensor section**
(12 comment lines total, no section markers).

---

## 4. The strata pack — `/srv/strata-data/packs/iq3_xxs/`

Produced by `tools/iq_pack.py` (the header line of `index.txt` says exactly which tool and source
shard). Purpose: give the C++ engine a flat, JSON-free, mmap-friendly view of the model.

### 4.1 `index.txt` (93,687 B, 1,081 lines)

```
# strata pack index v3 -- generated by tools/iq_pack.py (native experts) from Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
# align 64 pool 1538035200 tensors 1079
```

Then 1,079 rows, whitespace-separated, **no quoting, exactly 19 fields** (the engine refuses any
row that does not parse into 19 fields — `src/core/weights.cpp:177`):

```
name file kind src_off src_bytes dst_off dst_bytes ne0 ne1
     code_bits code_bias group_elems codebook has_offset codes_bytes scales_bytes
     offset_bytes scales_fp16 act_kind
```

Observed values in this pack (`file` is always `0` = `dense.bin`):

| kind | rows | meaning |
| ---: | ---: | --- |
| 0 | 302 | **native**: `src_* == 0`, `dst_bytes == 0`, `code_bits=8 group_elems=32` are placeholders — the tensor is served straight from the GGUF given by `--native`; row carries shape only (`dst_off` still tracks the arena cursor) |
| 4 | 484 | BF16 stored raw in `dense.bin` (engine takes the high 16 bits of an f32) |
| 2 | 292 | F32 copied verbatim |
| 5 | 1 | F16 stored raw (`blk.1.ple_conv1d.weight`) |
| 1 / 3 | 0 | not used in this pack (`BF16_IN_F32` promoted, `F16_IN_F32` narrowed by loader — reserved) |

* 302 + 484 + 292 + 1 = **1,079** rows = 1,224 model tensors − 144 expert tensors − 1 PLE table.
* `dst_bytes` of the 777 non-native rows sums to **1,538,035,200** = `# align … pool …` = exactly
  `dense.bin`'s size.
* Rows appear in GGUF tensor order (first `output.weight`, last `blk.47.indexer.q_proj.weight`),
  each row's `dst_off` is the running 64-byte-aligned arena cursor, `src_off` the offset inside
  `dense.bin` (identical here because `file=0`).
* `ne0`/`ne1` are the GGUF shape with `ne0` the contiguous axis; expert rows are absent.
* `act_kind` (col 19) and `codebook`/`has_offset`/`*_bytes`/`scales_fp16` (cols 13-18) are written
  as `0`/`1` defaults by `iq_pack.py`; the fully-populated variant comes from `tools/pack_index.py`
  (the `pack/full` family). The engine still parses and checks them.

Column semantics are documented in `tools/pack_index.py` (docstring) and implemented by
`src/core/weights.cpp` (`pool_bytes`, `index_code_bits`, `load`).

### 4.2 `native_experts.txt` (3,296 B)

```
# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard] (n_expert 512, total 42912972800; absolute offsets in Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf, or in the named shard beside it)
0 17 20 0 1868800 1227833216 1473813376 755052416
1 16 42 956825600 1305600 2016152832 2235918592 1779301632
…
47 18 20 41798860800 2176000 46363501952 46688125312 45890721152
```

48 rows (one per layer), space-separated:

| field | meaning (verified against the GGUF) |
| --- | --- |
| `layer` | 0…47 |
| `gu_type` | ggml type id of **gate and up** for that layer: 16 IQ2_XXS (9 layers), 17 IQ2_XS (10), 18 IQ3_XXS (6), 21 IQ3_S (13), 22 IQ2_S (10) |
| `d_type` | ggml type id of **down**: 42 Q2_0 (30 layers), 20 IQ4_NL (18 layers) |
| `offset` | cumulative offset of this layer's 512 blobs inside `experts.bin` (layer 0 = 0, layer 1 = 956,825,600 = 512 × 1,868,800 …) |
| `blob_bytes` | **bytes per expert blob** (not per layer): `gate rows | up rows | down rows` for one expert. `blob_bytes × 512 × Σ = 42,912,972,800` |
| `gate_off` / `up_off` / `down_off` | absolute byte offset of `blk.L.ffn_{gate,up,down}_exps.weight` inside shard 1's file (== GGUF `data_offset` of those tensors — verified for layers 0 and 1) |
| `[shard]` | optional 9th field, only in v4 packs (shard boundary inside a layer) — absent here |

Verified: for every layer, `blob_bytes × 512 == GGUF(gate) + GGUF(up) + GGUF(down)`, and
`Σ blob_bytes × 512 == experts.bin size` exactly.

### 4.3 `experts.bin` (42,912,972,800 B)

Raw, unquantized-*layout* copy of the GGUF expert tensors, re-sliced **per layer, per expert**:

```
for layer in 0..47:                       # layer offsets from native_experts.txt `offset`
    for expert in 0..511:                 # blob_bytes = layer's per-expert size
        [ gate rows for this expert ]     # GGUF blocks copied verbatim (gate 640 rows × 2560)
        [ up rows for this expert ]       # then up (640 × 2560)
        [ down rows for this expert ]     # then down (2560 × 640)
```

Same bytes as the GGUF, only the *interleaving* (tensor-major → expert-major) changes. Layer blob
sizes differ (e.g. layer 0 = 1,868,800 B/expert, layer 1 = 1,305,600 B/expert) because each layer
has its own `gu_type`/`d_type`. Engine side: `src/core/expert_source.cpp` (`open` → `experts.bin`,
`layer_offsets_`/`layer_blob_bytes_`, or `open_gguf()` when the file is absent and the GGUF shards
are mapped instead).

### 4.4 `dense.bin` (1,538,035,200 B)

The 777 non-native, non-expert tensors concatenated in `index.txt` order, each padded to **64 bytes**,
in the *engine's* form (BF16 as stored, F32 as stored, F16 as stored). No header — offsets come from
`index.txt`.

### 4.5 Sidecars and `tokenizer/`

`conversions.json` (310 B):

```json
{ "schema": 1, "tool": "tools/iq_pack.py", "compat_bf16": false,
  "source_shards": [ {"name": "…-00001-of-00002.gguf", "size": 47039860096},
                     {"name": "…-00002-of-00002.gguf", "size": 28800138432} ],
  "tensors": [] }
```

`tensors: []` ⇒ **no tensor needed conversion** for this model (the GSQ-RCO floats are already in the
engine's form); for other releases this array records `{name, shard, source_type, dst_type, method,
exact, max_abs_err, source_sha256}`.

`experts.bin.src.json` (377 B):

```json
{ "schema": 1,
  "shards": [ {"name": "…-00001-of-00002.gguf", "size": 47039860096},
              {"name": "…-00002-of-00002.gguf", "size": 28800138432} ],
  "native_experts_sha256": "ae28cf7a2a1c9063c4b55d6c8cf86d4d079dcddcf597f9e8697c9157d462c0ee",
  "bytes": 42912972800 }
```

— the provenance check: an `experts.bin` is only reused when the source shard names/sizes and the
hash of `native_experts.txt` match.

`tokenizer/` (exported by `tools/strata_tokenizer.py`):

| file | bytes | content |
| --- | ---: | --- |
| `vocab.json` | 5,737,005 | token → id map (248,320 entries) |
| `merges.txt` | 3,353,258 | 247,587 BPE merges |
| `token_type.json` | 744,960 | 248,320 token types |
| `tokenizer.json` | 541 | metadata only: `{"model":"gpt2","pre":"qwen35","vocab_size":248320,"n_merges":247587,"special_ids":{eos:248046,padding:248044,bos:248044},"add_bos_token":false,"pre_pattern":"(?:'[sS]|'…)","pre_pattern_source":".ref/llama.cpp src/llama-vocab.cpp L396 (LLAMA_VOCAB_PRE_TYPE_QWEN35)"}` |
| `chat_template.jinja` | 9,993 | same template as the GGUF KV `tokenizer.chat_template` |

Note `tokenizer.json` is **not** a HuggingFace tokenizer config — it is a small pointer file; the
real data is vocab/merges/token_type. The split regex must be implemented as `LLAMA_VOCAB_PRE_TYPE_QWEN35`.

---

## 5. `/srv/strata-data/mtp/` — the draft (MTP) layer

### 5.1 Provenance files

* `mtp-manifest.json` (11,643 B) — **array of 31 entries**, one per MTP tensor, pulled from the
  upstream HF safetensors:
  `{name, shard, dtype, shape[], start, end, bytes, file, sha256}`
  e.g. `{"name":"mtp.fc_embedding.weight","shard":"model-00120-of-00131.safetensors","dtype":"BF16",
  "shape":[2560,2560],"start":1935412640,"end":1948519839,"bytes":13107200,
  "file":"tensors/mtp.fc_embedding.weight.bin","sha256":"241d6e87…"}`.
* `mtp-inventory.json` (7,514 B) — **object**: `{repo, total_bytes: 5214301696, tensors: [31 ×
  {name, shard, dtype, shape, start, end, bytes}]}` (no `file`/`sha256`).
* `mtp-inventory.md` (2,437 B) — human table: 31 tensors, 5.214 GB, 28 shards, dtype/shape/MB.
  Note: the upstream names include `mtp.layers.0.*` for the one draft layer.
* `tensors/` — the 31 raw BF16/F32 blobs, file name = tensor name + `.bin`
  (`mtp.layers.0.mlp.experts.gate_up_proj.bin` 3,355,443,200 B … `mtp.layers.0.self_attn.q_norm.weight.bin`
  512 B). Total 5,214,301,696 B.
* `mtp-q2_0.gguf` (889,017,216 B) + `mtp-q2_0.gguf.report.json` — an intermediate GGUF built from
  those blobs by `tools/mtp_pack.py`; the report lists the 31 tensors with their stored format
  (BF16/F32 for floats, `q2_0` for the two expert tensors, plus measured relative RMS error and
  conversion seconds). Not read by the engine at inference; it is the pack's source.

### 5.2 `rt/` — the runtime files the engine actually loads

Written by `tools/mtp_rt.py`; consumed by `src/core/mtp.cpp`.

The four runtime files total **824,314,919 B**.

**`rt/dense.txt`** (1,851 B, 29 lines; kinds `q8_0` 10, `bf16` 10, `f32` 9) — one line per
non-expert tensor:

```
name kind rows cols offset bytes
fc_embedding.weight q8_0 2560 2560 0 6963200
fc_hidden.weight q8_0 2560 2560 6963200 6963200
hyper_connection_mixer.hc_norm.weight f32 1 10240 13926400 40960
…
pre_fc_norm_hidden.weight f32 1 10240 116058112 40960
```

* `kind ∈ {q8_0, bf16, f32}`; offsets are absolute inside `dense.bin`, contiguous, no padding
  (last offset 116,058,112 + 40,960 = 116,099,072 = file size ✓).
* Names drop the `mtp.` prefix (top level) or `mtp.layers.0.` (draft layer): manifest
  `mtp.layers.0.self_attn.q_proj.weight` → `self_attn.q_proj.weight`.
* Semantics from `mtp_rt.py`: large projections re-quantized to **Q8_0** (ggml reference rounding,
  34 B / 32 values) so the multi-column MMVQ path runs; hyper-connection and router weights kept
  **BF16**; RMSNorm weights stored **F32 with the Gemma "+1" already applied** (vLLM GemmaRMSNorm
  scales by `1 + w` — a loader must not add it again).

**`rt/dense.bin`** (116,099,072 B) — the packed bytes of those 29 rows.

**`rt/experts.bin`** (707,788,800 B) — 512 blobs × `BLOB = 3 × (2560 × 640 × 18 / 64) = 1,382,400 B`
(verified: 512 × 1,382,400 = 707,788,800). Layout is the *engine's* expert layout, **not** the GGUF
one (`include/strata/kernels/cpu/expert.hpp`, `tools/mtp_rt.py::blob_of`):

```
gate/up rows interleaved (row 2r = gate r, 2r+1 = up r), down rows after
   part 1: gate/up codes  (16 B per 64 values, rows interleaved)
   part 2: down codes
   part 3: gate/up fp16 scales (2 B per 64)
   part 4: down fp16 scales
```

i.e. a *plane-split* Q2_0 (2 bits + fp16 scale per 64 values) — same bytes as the GGUF, re-laid out.

**`rt/draft_vocab.bin`** (425,196 B) — a bare little-endian `int32` array: **106,299 token ids**,
no header (first values 0,1,2,…; max 248,319 < vocab 248,320; unique; base order preserved with
script additions appended in id order). Written by `tools/draft_vocab.py` (variants
`data/draft_vocab{,_en,_cyrillic,_fr}.bin` copied in by `setup.py::refresh_draft_vocab`). The draft
head computes logits over these rows only; a token outside can never be drafted.

Consumers in `/home/korphaus/projects/strata` (line numbers verified against the source):

| file | reader |
| --- | --- |
| `rt/dense.txt` | `src/core/mtp.cpp:270-278` — `is >> t.name >> t.kind >> t.rows >> t.cols >> t.off >> t.bytes`; a malformed line is fatal |
| `rt/dense.bin` | `src/core/mtp.cpp:296` — whole-file read, uploaded to VRAM |
| `rt/experts.bin` | `src/core/mtp.cpp:311-331` — `n_expert × BLOB` bytes, 64 MiB chunks |
| `rt/draft_vocab.bin` | `src/core/mtp.cpp:637-660` — `size/4` tokens; default path `<mtp>/draft_vocab.bin` (`include/strata/core/mtp.hpp:207`), override `--mtp-draft-vocab` (`src/program/generate.cpp:665, 2011`) |
| presence checks | `src/program/generate.cpp:7644-7645` requires `dense.txt`, `experts.bin`, `draft_vocab.bin` |

---

## 6. Vision: `mmproj-Qwen3.8-Flash-Next-BF16.gguf`

907,543,008 B, GGUF v3, `general.architecture = clip`, `general.type = mmproj`,
`general.file_type = 32`, `general.size_label = 449M`, tags `["gsq-rco","ista-daslab","image-text-to-text"]`.
334 tensors, `data_offset = 20,000`, ends exactly at EOF. Types: **BF16 110, F32 224** (no quantized
tensors).

KV: `clip.vision.projection_dim=2560, image_size=768, patch_size=16, embedding_length=1152,
feed_forward_length=4304, block_count=27, attention.head_count=16, image_mean/std=[0.5]*3,
projector_type=qwen3vl_merger, use_gelu=true, spatial_merge_size=2,
attention.layer_norm_epsilon=1e-6, is_deepstack_layers=[false × 27]`.

Tensor groups: `v.patch_embd.{weight,bias}` (4-D `16×16×3×1152`, duplicated as `.weight.1`),
`v.position_embd.weight` (1152×2304), 27 × `v.blk.N.{attn_qkv,attn_out,ffn_up,ffn_down}.{weight,bias}`
+ `v.blk.N.ln{1,2}.{weight,bias}` (all weights BF16, all biases/layernorms F32), `v.post_ln.*`,
and 2 × `mm.N.weight` (4608×4608) + 2 × `mm.N.bias` (the merger). Loaded by the engine via
`--vision` (`setup.py:5648`) and by the fork as `--mmproj`.

---

## 7. `models/Q2_0` symlink

```
/srv/strata-data/models/Q2_0
  -> /mnt/agent-storage-b/ai-dev/ai-runtime/home/korphaus/models/qwen-dual/Q2_0/
       Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf          37,623,740,192 B  (regular file)
       Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf  ->  /srv/strata-data/models/IQ3_XXS/
                                                            ...-IQ3_XXS-00002-of-00002.gguf  (86-byte symlink)
```

So the Q2_0 variant shares the **IQ3_XXS shard 2** — i.e. the same IQ4_NL PLE table for both
releases (confirmed by `docs/EXCHANGE_ROTATION.md:86` and `docs/AMD_HIP.md:163`: "Original GSQ-RCO
IQ3_XXS uses shard 2 for PLE"). The symlink lives *outside* `/srv/strata-data` (on
`/mnt/agent-storage-b`), so a hard-disk-space plan must account for it separately.

---

## 8. Reference-engine source map (read-only citations)

### strata (`/home/korphaus/projects/strata`)

| file | what it defines |
| --- | --- |
| `tools/iq_pack.py` | writes this pack: `index.txt` (v3, 19-field rows, kinds 4/5/2/3), `dense.bin` (64-B align), `native_experts.txt` (written last = completion marker), `conversions.json`, `experts.bin`, `tokenizer/`; `FORM` table (which float form each tensor must be in), `KIND = {BF16:4, F16:5, F32:2}` |
| `tools/pack_index.py` | the alternate/older index generator, column doc-comment and the `FILE_ID`/`VERBATIM/BF16_IN_F32/F32_COPY/F16_IN_F32` kinds, `act_kind` (Q8_0 vs Q8_K activation) |
| `tools/pack_layer.py`, `tools/strata_pack.py`, `tools/embd_bf16_pack.py`, `tools/ple_fp8_pack.py` | other pack variants |
| `src/core/weights.cpp:92-215` | **the `index.txt` loader**: `pool_bytes` (parses `# align %d pool %llu tensors %d`), `index_code_bits`, `load` (`sscanf` of exactly 19 fields, refuses otherwise; kind 4/5 → `raw16`; skip-set arena compaction) |
| `src/core/expert_source.cpp:841` | opens `pack_dir + "/experts.bin"`; `layer_offsets_`/`layer_blob_bytes_` |
| `src/core/expert_source.cpp:1066` | `FileExpertSource::open_gguf` — the fallback that maps the GGUF shards and slices `[gate|up|down]` per expert instead of `experts.bin` |
| `src/core/expert_source.cpp:4062-4140` | `expert_gguf_file` + `check_experts_gguf`: validates `native_experts.txt` (`gu_type`, `d_type`, `gguf_off`, per-role shape/type) against the actual GGUF; error text distinguishes v3 (per layer) and v4 (per role) |
| `src/core/expert_source.cpp:4032` | fingerprint of `{"manifest.json","index.txt","native_experts.txt"}` |
| `src/core/mtp.cpp:270` | parses `rt/dense.txt`, then reads `rt/dense.bin` and `rt/experts.bin` (`n_expert * BLOB`) |
| `tools/gguf_reader.py`, `include/strata/artifact/gguf_reader.hpp:120-122`, `src/artifact/gguf_reader.cpp:30-58` | strata's own GGUF reader + `block_geometry()` table (incl. the fork-only `Q2_0 = 64/18`) and `gguf_reader <file> --check` |
| `tools/mtp_rt.py`, `tools/mtp_pack.py`, `tools/draft_vocab.py` | writers of `rt/dense.{bin,txt}`, `rt/experts.bin`, `mtp-q2_0.gguf`, `draft_vocab*.bin` |
| `setup.py:5571-5574` | the real command line: `--pack <dir> --native <shard1> --ple-gguf <shard2> --expert-profile … --mtp <rt> …` |
| `docs/DETAILS.md:354, 367-374` | GSQ-RCO provenance; **only textual mention of `rco-allocation.txt`** (Coder's kept-expert mapping) |
| `bench/results/2026-09-28-coder/README.md:27` | "section 2 of the release's `tensor-allocation/*.rco-allocation.txt`" |
| `docs/AMD_HIP.md:163, 177-183` | `--native` / `--ple-gguf` usage for the original GSQ-RCO shards |

**No code in `strata` parses `*.rco-allocation.txt`** — `grep -rniI 'rco-allocation'` over the whole
tree returns only the three doc/bench hits above. It is a provenance artifact, not an engine input.

### llama.cpp fork (`/home/korphaus/llama.cpp-qwen4exp-mmap`, source tree + `build-hip-mmap/` in the same dir)

| file | what it defines |
| --- | --- |
| `src/llama-arch.cpp:45` | `LLM_ARCH_QWEN4EXP ↔ "qwen4exp"`; `:504` `LLM_TENSOR_ATTN_QKV → "blk.%d.attn_qkv"`; `:542` `ssm_out`; `:597` `ple_key`; `:313-317` `hyper_connection.*` KV names |
| `src/llama-hparams.h:13-14` | `LLAMA_MAX_PLE_NGRAM 8`, `LLAMA_MAX_PLE_HEADS 64` |
| `src/llama-model.cpp` | tensor creation for `qwen4exp` (incl. `ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps`, optional expert `scale`/`input_scale`, indexer cache selection, `arch == LLM_ARCH_QWEN4EXP` branches at `:3115`) |
| `src/models/qwen4exp.cpp` | the forward pass (SSM + full attention + indexer + PLE + hyper-connection) |
| `src/llama-memory-hybrid-idx.*`, `src/llama-kv-cache.*` | hybrid/recurrent + indexer KV-cache layout |
| `src/llama-quant.cpp` | quant tables incl. the fork's extra types (`Q1_0=41`, `Q2_0=42`) |
| `gguf-py/gguf/constants.py:5997-6029` | `GGMLQuantizationType` enum used above |

**No `RCO` / `rco-allocation` / `tensor-allocation` parsing exists in the fork either**
(`grep -rniI '\brco\b'` over `*.cpp,*.h,*.py,*.md` returns nothing). Neither engine consumes the
allocation file; both simply read whatever quant type each tensor declares in the GGUF header.

---

## 9. "What a new engine must parse" checklist

**A. GGUF container (both shards + mmproj)**
- [ ] `GGUF` magic, `u32 version == 3`, `u64 n_tensors`, `u64 n_kv`
- [ ] KV stream: `u64 len + bytes` key, `u32 type`, value; types 0-12 incl. type 9 arrays
      (array = `u32 elem_type` + `u64 count`, recursive) — tokenizer merges alone are 247,587 strings
- [ ] 32-byte alignment of the tensor-data section (`data_offset`); tensor record =
      `name(u64+bytes)`, `u32 n_dims`, `u64 dims[n_dims]`, `u32 type`, `u64 offset`
- [ ] Split handling: metadata only on shard 1; check `split.no/count/tensors.count` consistency
      (1224 total); open all shards; last tensor ends exactly at EOF (integrity check)
- [ ] Tensor lookup by name across shards (PLE table lives in shard 2; `--native` dense tensors live
      in shard 1; expert tensors may be read in place instead of `experts.bin`)

**B. Quant block formats (implement exactly the ids in §2.5)**
- [ ] Floats: F32, F16, **BF16**
- [ ] K-quants: Q4_K, Q5_K, Q6_K (plus Q3_K/Q4_0/Q5_0/Q8_0 needed for the Q2_0 model per §3)
- [ ] i-quants: IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS (IQ1_S/IQ1_M for other releases)
- [ ] **Fork-specific `Q2_0 = 42`: 64 values / 18 B (16 B codes + 2 B fp16 scale), 2.25 bpw** —
      not upstream ggml's Q2_0
- [ ] Note per-tensor allocation: type must come from the tensor header, never from the release name

**C. Model topology from KV (`qwen4exp.*`)**
- [ ] 48 blocks; 36 hybrid-SSM (fused `attn_qkv`, `ssm_*`) + 12 full-attention (blocks 3,7,…,47 with
      separate q/k/v/o + `indexer.*`), `full_attention_interval=4`, `attention.compress_ratios`
- [ ] 512 experts / 10 used, router `ffn_gate_inp`, shared expert, 4-way hyper-connection (low_rank 320)
- [ ] SSM: conv_kernel 4, state 128, groups 16, time_step_rank 48, inner 6144
- [ ] PLE at layer 1 only: `ple_*` tensors in `blk.1`, table `per_layer_token_embd.weight`
      `[160, 320001536]` in shard 2, `ple.head_offsets/head_vocab_sizes/layer_multipliers`
- [ ] Attention: 24 heads / 2 KV heads, `key_length=value_length=256`, indexer 4 heads / key 128 /
      top_k 2048, rope `dimension_sections [11,11,10,0]`, `freq_base 1e7`, ctx 262144, d_model 2560

**D. Tokenizer**
- [ ] BPE (`gpt2`), 248,320 tokens / 247,587 merges, pre-tokenizer type **QWEN35** (exact regex),
      special ids eos 248046, bos/pad 248044, `add_bos=false`, image token 248056
- [ ] Jinja chat template (10,279 chars) or an equivalent hand-written renderer

**E. Strata pack (if loading the pack instead of / alongside the GGUF)**
- [ ] `index.txt`: 2 header comment lines (`# strata pack index v3`, `# align <a> pool <p> tensors <n>`),
      then exactly 19 whitespace-separated fields per row; refuse on parse failure
- [ ] kinds 0 (native → resolve from the `--native` GGUF by name), 4 (BF16 raw), 2 (F32), 5 (F16 raw),
      1/3 (promoted/narrowed variants); arena = `pool` bytes, 64-byte alignment, `dst_off` per row
- [ ] `dense.bin`: headerless blob of exactly `pool` bytes
- [ ] `native_experts.txt`: header comment + 48 rows of `layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard]`
      with `blob_bytes` = per-expert size and offsets absolute inside the shard
- [ ] `experts.bin`: 48 layers × 512 blobs of `[gate rows | up rows | down rows]`, verbatim GGUF
      blocks; layer sizes from `native_experts.txt`
- [ ] `conversions.json` / `experts.bin.src.json`: validation (schema, source shard sizes, sha256)
- [ ] `tokenizer/`: `vocab.json` + `merges.txt` + `token_type.json` + the `tokenizer.json` pointer
      (not a HF tokenizer.json) + `chat_template.jinja`

**F. MTP draft layer (`--mtp <dir>`)**
- [ ] `rt/dense.txt`: 6 fields `name kind rows cols offset bytes`, kinds `q8_0|bf16|f32`;
      name mapping strips `mtp.` / `mtp.layers.0.`
- [ ] `rt/dense.bin`: 116,099,072 B, offsets contiguous, RMSNorm values already have `+1` applied
- [ ] `rt/experts.bin`: `n_expert × BLOB` with `BLOB = 3·(H·FF·18/64) = 1,382,400` B, plane-split
      layout (gu codes | down codes | gu scales | down scales), gate/up rows interleaved
- [ ] `rt/draft_vocab.bin`: bare `int32[]` of allowed token ids (106,299)
- [ ] `mtp-manifest.json` / `mtp-inventory.json` / `mtp-q2_0.gguf`: provenance only (not needed at
      inference)

**G. Vision**
- [ ] `mmproj` GGUF with `clip` architecture, 334 BF16/F32 tensors, qwen3vl_merger projector,
      27 blocks, patch 16 / image 768 / width 1152 → projection 2560

**H. Optional / provenance-only (no parser required)**
- [ ] `*.rco-allocation.txt` (12 comment lines + `name: TYPE` per tensor, Q2_0 release only)
- [ ] `*.gguf.done` timestamp markers, `models/.cache/`, `mtp-unsloth/`
- [ ] `mtp/mtp-q2_0.gguf.report.json`, `mtp-inventory.md`
