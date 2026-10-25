# Model artifacts inventory — Qwen3.8-Flash-Next GSQ-RCO (this machine)

**Purpose.** A from-scratch inference engine has to open these files and nothing else. This document
records what is actually on disk under `/srv/strata-data`, how each file is laid out byte for byte,
which reference implementation reads it, and what a new loader must parse.

**Method.** All GGUF headers were parsed with `python3` + `struct` (no `gguf-py` on this machine —
`python3 -c "import gguf"` fails; the pinned copy lives at
`/home/korphaus/llama.cpp-qwen4exp-mmap/gguf-py/` and
`/home/korphaus/projects/strata/tools/_paths.py:add_gguf_py`). Every byte-size claim below was
cross-checked against the block geometry of the types present (offset deltas sum to the file size
exactly). Large payload files were **not** read beyond the first 512 bytes; all numbers come from
headers, indices and metadata. Read-only inventory; nothing on disk was modified.

Date of capture: 2026-10-25.

---

## 0. Layout and disk sizes

`/srv/strata-data` contains four top-level directories: `models/`, `packs/`, `mtp/`, `mtp-unsloth/`.

| Path | Bytes | GiB | What it is |
|---|---:|---:|---|
| `models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` | 47,039,860,096 | 43.81 | model shard 1 (all metadata, 1223 tensors) |
| `models/IQ3_XXS/…-00002-of-00002.gguf` | 28,800,138,432 | 26.82 | model shard 2 (1 tensor: the PLE table) |
| `models/IQ3_XXS/*.gguf.done` (×2) | 16 each | — | downloader timestamp, e.g. `2026-10-11 02:25` |
| `models/mmproj-Qwen3.8-Flash-Next-BF16.gguf` | 907,543,008 | 0.845 | vision projector (`clip`, 334 tensors, unquantised) |
| `models/mmproj-….gguf.done` | 16 | — | timestamp |
| `models/tensor-allocation/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.rco-allocation.txt` | 39,062 | — | RCO per-tensor quant assignment (Q2_0 variant) |
| `models/Q2_0/` | — | — | **symlink directory** → `/mnt/agent-storage-b/ai-dev/ai-runtime/home/korphaus/models/qwen-dual/Q2_0` |
| `models/Q2_0/…Q2_0-00001-of-00002.gguf` | 37,623,740,192 | 35.04 | Q2_0 variant shard 1 |
| `models/Q2_0/…Q2_0-00002-of-00002.gguf` | 86 (symlink) | — | → `models/IQ3_XXS/…IQ3_XXS-00002-of-00002.gguf` (PLE shard shared verbatim) |
| `models/.cache/huggingface/download/tensor-allocation/` | — | — | HF download cache for the allocation file only |
| `packs/iq3_xxs/index.txt` | 93,687 | — | pack weight table (1079 rows) |
| `packs/iq3_xxs/native_experts.txt` | 3,296 | — | 48-layer expert locator (v3) |
| `packs/iq3_xxs/experts.bin` | 42,912,972,800 | 39.97 | pre-laid-out routed experts, all 48 layers |
| `packs/iq3_xxs/dense.bin` | 1,538,035,200 | 1.43 | float tensors in engine form |
| `packs/iq3_xxs/conversions.json` | 310 | — | conversion log (`"tensors": []` — nothing was converted) |
| `packs/iq3_xxs/experts.bin.src.json` | 377 | — | provenance/hash for `experts.bin` |
| `packs/iq3_xxs/tokenizer/` (5 files) | 9,845,757 | — | vocab/merges/template export |
| `packs/iq3_xxs/` total | 44,460,951,427 | 41.41 | |
| `mtp/mtp-manifest.json` | 11,643 | — | 31 MTP tensors: HF shard, byte range, sha256 |
| `mtp/mtp-inventory.json` / `.md` | 7,514 / 2,437 | — | same list, human/JSON form |
| `mtp/tensors/` (31 `.bin`) | 5,214,301,696 | 4.86 | raw BF16 MTP tensors as fetched |
| `mtp/mtp-q2_0.gguf` | 889,017,216 | 0.83 | MTP block packed as a GGUF |
| `mtp/mtp-q2_0.gguf.report.json` | 3,589 | — | per-tensor format + relative RMS error |
| `mtp/rt/dense.bin` | 116,099,072 | 0.108 | MTP dense weights, engine form |
| `mtp/rt/experts.bin` | 707,788,800 | 0.659 | 512 MTP expert blobs |
| `mtp/rt/draft_vocab.bin` | 425,196 | — | 106,299 × int32 token ids |
| `mtp/rt/dense.txt` | 1,851 | — | 29-row tensor index |
| `mtp/` total | 6,927,659,014 | 6.45 | |
| `mtp-unsloth/MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf` | 4,137,429,120 | 3.85 | Unsloth Q8_0 MTP GGUF (alt. draft head) |
| `mtp-unsloth/dl.log` | 253 | — | `hf` download log |
| `models/` total | 76,747,580,822 | 71.47 | |

Filesystem: `/dev/nvme0n1p6`, 778 G total, 639 G used, 121 G free at capture time.

---

## 1. The GGUF shards

### 1.1 Container mechanics (verified on these files)

```
offset 0   : 4 B magic  'GGUF' (LE 0x46554747)
offset 4   : u32 version = 3
offset 8   : u64 tensor_count
offset 16  : u64 metadata_kv_count
then       : metadata KV pairs   (key = u64 len + bytes; u32 value type; value)
then       : tensor infos        (name = u64 len + bytes; u32 n_dims; u64 dims[n_dims];
                                  u32 ggml type; u64 offset — relative to data_start)
data_start : align(end_of_tensor_infos, 32)          <- GGUF_DEFAULT_ALIGNMENT = 32
```

Value types: `0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32, 6 f32, 7 bool, 8 string, 9 array
(u32 elem_type + u64 count + values), 10 u64, 11 i64, 12 f64`.

Measured for shard 1 (`…IQ3_XXS-00001-of-00002.gguf`):

| Field | Value |
|---|---:|
| magic / version | `GGUF` / 3 |
| metadata KV count | 67 |
| end of metadata | 10,946,629 (tokenizer arrays dominate: 248,320 tokens + 247,587 merges) |
| end of tensor info (unaligned) | 11,024,251 |
| **data_start (aligned to 32)** | **11,024,256** |
| tensor count | 1,223 |
| data bytes | 47,028,835,840 |
| file size | 47,039,860,096 (= 11,024,256 + 47,028,835,840, exact) |

Shard 2 (`…-00002-of-00002.gguf`): 3 KVs, metadata ends at 106, tensor info ends at 173,
`data_start = 192`, 1 tensor, data = 28,800,138,240 B (exact: 160 × 320,001,536 elements × 18/32
for IQ4_NL).

Each shard repeats the split KVs; **all architecture/tokenizer metadata lives in shard 1 only**.
A loader must therefore open every shard's header (a shard cannot be interpreted standalone) but
only reads the KV block once.

### 1.2 Full metadata KV list — shard 1 (67 pairs)

Verified and extended from the earlier investigation. `type` is the GGUF value type id.

| Key | Type | Value |
|---|---|---|
| `general.architecture` | str | **`qwen4exp`** |
| `general.type` | str | `model` |
| `general.sampling.top_k` | i32 | 20 |
| `general.sampling.top_p` | f32 | 0.95 |
| `general.sampling.temp` | f32 | 1.0 |
| `general.name` | str | `Qwen3.8-Flash-Next GSQ-RCO` |
| `general.description` | str | `A Preview of the Qwen4 Architecture` |
| `general.size_label` | str | `512x56B` |
| `general.quantized_by` | str | `ISTA DASLab` |
| `general.repo_url` | str | `https://huggingface.co/ISTA-DASLab` |
| `general.tags` | arr[str] | `['gsq-rco', 'ista-daslab']` |
| `general.file_type` | u32 | 23 = `LLAMA_FTYPE_MOSTLY_IQ3_XXS` |
| `general.quantization_version` | u32 | 2 |
| `qwen4exp.block_count` | u32 | **48** |
| `qwen4exp.context_length` | u32 | **262144** |
| `qwen4exp.embedding_length` | u32 | **2560** |
| `qwen4exp.embedding_length_per_layer_input` | u32 | 160 (PLE row dim) |
| `qwen4exp.attention.head_count` | u32 | **24** |
| `qwen4exp.attention.head_count_kv` | u32 | **2** |
| `qwen4exp.attention.key_length` | u32 | **256** |
| `qwen4exp.attention.value_length` | u32 | **256** |
| `qwen4exp.attention.layer_norm_rms_epsilon` | f32 | 1e-6 |
| `qwen4exp.attention.compress_ratios` | arr[u32] ×48 | `0,0,0,4` repeated ×12 (ratio 4 exactly on layers ≡3 mod 4) |
| `qwen4exp.full_attention_interval` | u32 | **4** |
| `qwen4exp.rope.dimension_count` | u32 | 64 |
| `qwen4exp.rope.dimension_sections` | arr[u32] | `[11, 11, 10, 0]` |
| `qwen4exp.rope.freq_base` | f32 | 10,000,000.0 |
| `qwen4exp.expert_count` | u32 | **512** |
| `qwen4exp.expert_used_count` | u32 | **10** |
| `qwen4exp.expert_feed_forward_length` | u32 | 640 |
| `qwen4exp.expert_shared_feed_forward_length` | u32 | 640 |
| `qwen4exp.ssm.conv_kernel` | u32 | **4** |
| `qwen4exp.ssm.state_size` | u32 | **128** |
| `qwen4exp.ssm.group_count` | u32 | **16** |
| `qwen4exp.ssm.time_step_rank` | u32 | **48** |
| `qwen4exp.ssm.inner_size` | u32 | **6144** |
| `qwen4exp.hyper_connection.count` | u32 | **4** |
| `qwen4exp.hyper_connection.low_rank` | u32 | **320** |
| `qwen4exp.attention.indexer.head_count` | u32 | **4** |
| `qwen4exp.attention.indexer.key_length` | u32 | **128** |
| `qwen4exp.attention.indexer.top_k` | u32 | **2048** |
| `qwen4exp.ple.layers` | arr[u32] | `[1]` (PLE lives in block 1 only) |
| `qwen4exp.ple.ngram_size` | u32 | **3** |
| `qwen4exp.ple.heads_per_ngram` | u32 | **8** |
| `qwen4exp.ple.conv_kernel` | u32 | 4 |
| `qwen4exp.ple.eos_token_id` | u32 | 248044 |
| `qwen4exp.ple.image_token_id` | u32 | 248056 |
| `qwen4exp.ple.layer_multipliers` | arr[i64] ×3 | `23703573157769, 20109073645365, 8052911324071` |
| `qwen4exp.ple.head_offsets` | arr[i64] ×16 | `0, 20000003, 40000026, …, 300001275` |
| `qwen4exp.ple.head_vocab_sizes` | arr[i64] ×16 | `20000003, 20000023, …, 20000171` |
| `tokenizer.ggml.model` | str | **`gpt2`** |
| `tokenizer.ggml.pre` | str | **`qwen35`** |
| `tokenizer.ggml.tokens` | arr[str] | 248,320 entries |
| `tokenizer.ggml.token_type` | arr[i32] | 248,320 entries |
| `tokenizer.ggml.merges` | arr[str] | 247,587 entries |
| `tokenizer.ggml.eos_token_id` | u32 | 248046 |
| `tokenizer.ggml.bos_token_id` | u32 | 248044 |
| `tokenizer.ggml.padding_token_id` | u32 | 248044 |
| `tokenizer.ggml.add_bos_token` | bool | false |
| `tokenizer.chat_template` | str | Jinja (image/video aware; 9,993 B, identical to `tokenizer/chat_template.jinja`) |
| `quantize.imatrix.file` | str | `imatrix-qwen3.8-flash-next.gguf` |
| `quantize.imatrix.dataset` | str | `calib_qwen3.8-flash-next.tokids` |
| `quantize.imatrix.entries_count` | u32 | 926 |
| `quantize.imatrix.chunks_count` | u32 | 1000 |
| `split.no` | u16 | 0 |
| `split.count` | u16 | 2 |
| `split.tensors.count` | i32 | 1224 |

Notes on the *new* keys (everything after the prior investigation's list):

* `rope.dimension_sections` is the per-section RoPE split `[11,11,10,0]` alongside
  `rope.dimension_count=64`; `rope.freq_base=1e7`.
* `compress_ratios[i] == 4` exactly on layers 3, 7, 11, … 47 and 0 elsewhere. In the reference
  converter (`conversion/qwen4exp.py:86`) that is
  `ratio if layer_types[i] == "full_attention" else 0`, i.e. **12 full-attention/QSA layers
  (≡3 mod 4) and 36 linear-attention (SSM/GDN) layers**. `src/models/qwen4exp.cpp:148` states the
  same rule as "linear attention everywhere except every `full_attention_interval`-th layer".
  All 12 QSA layers must share one ratio (`qwen4exp.cpp:71`, `indexer_kpool`), and
  `indexer_top_k % ratio == 0` (2048 % 4 == 0).
* `ple.layers=[1]` → the PLE/`per_layer_token_embd` hash table belongs to block 1 (see §1.5).
* `ple.layer_multipliers` / `ple.head_offsets` / `ple.head_vocab_sizes` are **int64 hash constants
  copied verbatim from the checkpoint** (`conversion/qwen4exp.py:110-115` →
  `ple_embedding.layer_multipliers`, `ple_embedding.ngram_heads_offsets`,
  `ple_embedding.ngram_heads_vocab_sizes`). They are not tensor layout offsets; treat as opaque
  constants for the PLE hash.
* `general.file_type` is the `llama_ftype` enum, **not** a ggml type id: 23 = `MOSTLY_IQ3_XXS`
  here, 41 = `MOSTLY_Q2_0` on the Q2_0 file, 32 = `MOSTLY_BF16` on mmproj, 7 = `MOSTLY_Q8_0` on
  the Unsloth MTP file (`include/llama.h:119-160` in the llama.cpp fork).

Shard 2 carries only three KVs: `split.no = 1`, `split.count = 2`,
`split.tensors.count = 1224` (shard 1 has `split.no = 0`). No architecture, no tokenizer —
shard 2 cannot be interpreted on its own, it only has to be *located*.

### 1.3 Tensor inventory (1,224 total)

| Location | Tensors |
|---|---:|
| shard 1 | 1,223 |
| shard 2 | 1 (`per_layer_token_embd.weight`) |
| **total** | **1,224** (`split.tensors.count`) |

Five tensors are outside the block loop:

| Name | Shape (GGUF order, dim0 fastest) | Type | Bytes |
|---|---|---|---:|
| `token_embd.weight` | `[2560, 248320]` | IQ3_S | 273,152,000 |
| `output.weight` | `[2560, 248320]` | Q5_K | 437,043,200 |
| `output_hc_down.weight` | `[10240, 320]` | BF16 | 6,553,600 |
| `output_hc_norm.weight` | `[10240]` | F32 | 40,960 |
| `output_hc_up.weight` | `[320, 10240]` | BF16 | 6,553,600 |
| `per_layer_token_embd.weight` *(shard 2)* | `[160, 320001536]` | IQ4_NL | 28,800,138,240 |

`per_layer_token_embd` is 16 heads × 20,000,096 rows of 160 dims (the PLE n-gram table). It is
26.8 GB of the model and is **not** in the pack; the reference engine reads it straight from
shard 2 with `--ple-gguf` (`src/kernels/ngram.cpp:242`).

### 1.4 Block taxonomy (per-block tensor sets)

| Block role | Layers | Tensors/layer | Distinguishing tensors |
|---|---|---:|---|
| **SSM / linear-attention** | 0,1,2, 4,5,6, 8,9,10, … 44,45,46 (**36**) | 25 (25 + 6 in blk 1) | `attn_qkv`, `attn_gate`, `ssm_*` |
| **Full attention / QSA** | 3,7,11, … 47 (**12**) | 26 | `attn_q/k/v/output`, `attn_q_norm`, `attn_k_norm`, `indexer.*` |
| **with PLE** | 1 only | 31 | `ple_*` (+6 tensors) |

`compress_ratios[i]==4` and `attn_q/k/v` presence agree exactly on the same 12 layers.

### 1.5 Naming scheme and shapes

Names are `blk.<N>.<role>` (GGUF order = dim0 fastest throughout). Quants shown are the distinct
set across the 48 layers — **each tensor instance carries its own type** (§1.7).

**Common to all 48 layers**

| Pattern | Count | Shape | Type (varies per layer) |
|---|---:|---|---|
| `blk.N.ffn_gate_exps.weight` | 48 | `[2560, 640, 512]` | IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S |
| `blk.N.ffn_up_exps.weight` | 48 | `[2560, 640, 512]` | same family as gate |
| `blk.N.ffn_down_exps.weight` | 48 | `[640, 2560, 512]` | IQ4_NL / Q2_0 |
| `blk.N.ffn_gate_inp.weight` | 48 | `[2560, 512]` | BF16 (router, 512 experts) |
| `blk.N.ffn_gate_shexp.weight` | 48 | `[2560, 640]` | IQ3_S / IQ4_XS / Q4_K / Q5_K / Q6_K |
| `blk.N.ffn_up_shexp.weight` | 48 | `[2560, 640]` | same family |
| `blk.N.ffn_down_shexp.weight` | 48 | `[640, 2560]` | IQ4_NL / Q2_0 |
| `blk.N.ffn_gate_inp_shexp.weight` | 48 | `[2560]` | BF16 (shared-expert router) |
| `blk.N.hc_attn_down.weight` | 48 | `[10240, 320]` | BF16 |
| `blk.N.hc_attn_up.weight` | 48 | `[320, 10240]` | BF16 |
| `blk.N.hc_attn_inject.weight` | 48 | `[10240, 4]` | BF16 (`hyper_connection.count=4`) |
| `blk.N.hc_attn_norm.weight` | 48 | `[10240]` | F32 |
| `blk.N.hc_ffn_down/up/inject/norm` | 48×4 | as above | BF16 / F32 |

**SSM layers (36)**

| Pattern | Count | Shape | Type |
|---|---:|---|---|
| `blk.N.attn_qkv.weight` | 36 | `[2560, 10240]` | IQ4_XS / Q4_K / Q5_K / Q6_K |
| `blk.N.attn_gate.weight` | 36 | `[2560, 6144]` | IQ3_S / IQ4_XS / Q4_K / Q5_K / Q6_K |
| `blk.N.ssm_out.weight` | 36 | `[6144, 2560]` | IQ3_S / IQ4_XS / Q4_K / Q5_K / Q6_K |
| `blk.N.ssm_conv1d.weight` | 36 | `[4, 10240]` | F32 (`ssm.conv_kernel=4`) |
| `blk.N.ssm_alpha.weight` | 36 | `[2560, 48]` | BF16 (`ssm.time_step_rank=48`) |
| `blk.N.ssm_beta.weight` | 36 | `[2560, 48]` | BF16 |
| `blk.N.ssm_a` | 36 | `[48]` | F32 |
| `blk.N.ssm_dt.bias` | 36 | `[48]` | F32 |
| `blk.N.ssm_norm.weight` | 36 | `[128]` | F32 (`ssm.state_size=128`) |

**Full-attention layers (12)**

| Pattern | Count | Shape | Type |
|---|---:|---|---|
| `blk.N.attn_q.weight` | 12 | `[2560, 12288]` | IQ3_S / IQ4_XS / Q4_K / Q6_K |
| `blk.N.attn_k.weight` | 12 | `[2560, 512]` | IQ3_S / IQ4_XS / Q4_K / Q5_K / Q6_K |
| `blk.N.attn_v.weight` | 12 | `[2560, 512]` | IQ4_XS / Q4_K / Q6_K |
| `blk.N.attn_output.weight` | 12 | `[6144, 2560]` | Q4_K / Q5_K / Q6_K |
| `blk.N.attn_q_norm.weight` | 12 | `[256]` | F32 |
| `blk.N.attn_k_norm.weight` | 12 | `[256]` | F32 |
| `blk.N.indexer.q_proj.weight` | 12 | `[2560, 512]` | BF16 (4 heads × 128) |
| `blk.N.indexer.k_proj.weight` | 12 | `[2560, 128]` | BF16 |
| `blk.N.indexer.q_norm.weight` | 12 | `[128]` | F32 |
| `blk.N.indexer.k_norm.weight` | 12 | `[128]` | F32 |

**PLE (block 1 only, 6 tensors)**

| Name | Shape | Type |
|---|---|---|
| `blk.1.ple_key.weight` | `[2560, 10240]` | BF16 |
| `blk.1.ple_value.weight` | `[2560, 2560]` | BF16 |
| `blk.1.ple_conv1d.weight` | `[4, 10240]` | **F16** (the only F16 tensor) |
| `blk.1.ple_norm_query.weight` | `[10240]` | F32 |
| `blk.1.ple_norm_key.weight` | `[10240]` | F32 |
| `blk.1.ple_norm_conv.weight` | `[10240]` | F32 |

Naming vocabulary: `attn_*` attention, `ffn_*_exps` routed experts (512 along the last dim),
`ffn_*_shexp` shared expert, `ffn_gate_inp*` routers, `ssm_*` GDN/linear-attention state space,
`hc_*` hyper-connection (4 streams, low rank 320), `indexer.*` QSA/MoA index heads,
`ple_*` per-layer n-gram embedding, `output*`/`token_embd` embedding head. There is **no**
`blk.N.attn_*` prefix variant like `blk.N.self_attn` — the GGUF names are the short forms; the
MTP checkpoint names (`mtp.layers.0.self_attn.*`) are the HF-side names (§4).

### 1.6 Quantisation types present and their block geometry

Type ids are the `ggml_type` enum of the llama.cpp fork
(`ggml/include/ggml.h:390-433`). Byte counts were verified against actual offset deltas.

| id | Name | block (elems, bytes) | B/elem | bpw | # tensors in shard 1 |
|---:|---|---|---:|---:|---:|
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
| 42 | **Q2_0** | (64, 18) | 0.28125 | 2.25 | 38 |

**Type 42 = `Q2_0` is a fork-only type** (64-element blocks, 18 bytes: fp16 `d` + 16 B of 2-bit
codes). It is the GSQ-RCO routed-expert encoding, absent from gguf-py ≤0.19 — this is why
`tools/gguf_reader.py` exists and why `include/strata/artifact/gguf_reader.hpp:120` proves
`Q2_0 = 64/18` "from this artifact's bytes". Q2_0 additionally uses ids 2 (`Q4_0`, 32/18),
6 (`Q5_0`, 32/22), 8 (`Q8_0`, 32/34), 11 (`Q3_K`, 256/110).

A reference geometry table lives in `/home/korphaus/projects/strata/include/strata/artifact/gguf_reader.hpp:122`
(`block_geometry()`), and a per-tensor byte check in `src/artifact/gguf_reader.cpp:30-58`
(`gguf_reader <file> --check` brackets each tensor's size between its offset and the next one's).

### 1.7 Per-tensor (non-uniform) quantisation — the "RCO" part

Quantisation is **per tensor, chosen per layer**, not one type per tensor family. Examples
(layer → type):

```
blk.N.ffn_down_exps  : 0:IQ4_NL 1:Q2_0 2:IQ4_NL 3:Q2_0 4:IQ4_NL 5:Q2_0 6:Q2_0 … 47:IQ4_NL
blk.N.ffn_gate_exps  : 0:IQ2_XS 1:IQ2_XXS 2:IQ2_XS 3:IQ2_XXS 4:IQ2_XS 5:IQ2_XS …
blk.N.attn_qkv       : 0:IQ4_XS 1:IQ4_XS 2:IQ4_XS 4:Q6_K 5:Q6_K 6:IQ4_XS …
blk.N.ssm_out        : 0:IQ4_XS 1:IQ3_S 2:Q4_K 4:IQ4_XS 5:IQ4_XS 6:Q5_K …
blk.N.attn_q (QSA)   : 3:IQ4_XS 7:IQ4_XS 11:Q4_K 15:IQ4_XS … 47:IQ3_S
```

So the loader must read the type id **per tensor from the GGUF tensor info**; it cannot infer it
from the tensor name. Byte share of shard 1's data section:

| Type | # | Bytes | Share |
|---|---:|---:|---:|
| IQ3_S | 82 | 9,774,899,200 | 20.78 % |
| IQ4_NL | 58 | 8,530,329,600 | 18.14 % |
| Q2_0 | 38 | 7,081,574,400 | 15.06 % |
| IQ2_S | 20 | 5,373,952,000 | 11.43 % |
| IQ2_XS | 20 | 4,849,664,000 | 10.31 % |
| IQ2_XXS | 18 | 3,892,838,400 | 8.28 % |
| IQ3_XXS | 12 | 3,853,516,800 | 8.19 % |
| BF16 | 484 | 1,527,889,920 | 3.25 % |
| Q6_K | 71 | 864,460,800 | 1.84 % |
| Q5_K | 15 | 580,771,840 | 1.23 % |
| IQ4_XS | 67 | 391,854,080 | 0.83 % |
| Q4_K | 45 | 296,939,520 | 0.63 % |
| F32 | 292 | 10,063,360 | 0.02 % |
| F16 | 1 | 81,920 | 0.00 % |
| **total** | **1223** | **47,028,835,840** | 100 % |

Whole model (both shards) = 75,828,974,080 data bytes over 176,943,899,520 stored elements
→ 3.43 bits per stored element (this counts the 51.2 B-parameter PLE table).

Fixed float groups (never RCO'd): all `hc_*`, `*_norm*`, `ssm_a`, `ssm_dt.bias`,
`ssm_conv1d`, all `indexer.*_proj`, all `ffn_gate_inp*`, `ple_value/ple_key` (BF16) and the
norm/`ssm_norm`/`ple_norm_*` (F32) tensors.

### 1.8 Split-shard handling

* `split.no` / `split.count` / `split.tensors.count` are present in every shard (0/2/1224 and
  1/2/1224).
* Every shard has its own complete GGUF header and its own 32-byte-aligned data section.
* Tensor ownership is discovered by parsing each shard's tensor-info list (no cross-shard index).
* The Q2_0 variant re-uses IQ3_XXS's shard 2 by symlink, so **shard 2 is byte-identical across
  variants** (the float/PLE tensors are identical; confirmed by `packs/iq3_xxs/conversions.json`).

---

## 2. `models/tensor-allocation/*.rco-allocation.txt` (the RCO allocation file)

One file: `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.rco-allocation.txt`, 39,062 B,
1,235 lines = 12 comment lines + 1,223 allocation lines. It describes the **Q2_0** GGUF, not the
IQ3_XXS one.

Header (verbatim structure):

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

Then one line per tensor, in GGUF tensor order, `<name>: <TYPE>`, e.g.

```
output.weight: Q5_K
token_embd.weight: Q3_K
blk.0.attn_gate.weight: Q4_K
blk.0.ffn_down_exps.weight: Q2_0
blk.1.ple_conv1d.weight: F16
blk.47.indexer.q_proj.weight: BF16
```

Semantics, verified against the Q2_0 GGUF on disk:

* **1,223 lines ↔ 1,223 shard-1 tensors; zero missing, zero extra, zero type mismatches** (an
  independent parse of the Q2_0 shard-1 header produced exactly the same type per tensor and the
  same histogram as the header comment). The file is therefore a *complete, reproducible
  specification* of the variant's quantisation — a new engine can regenerate or verify the file
  from it, but does not need it if it reads the GGUF (the type is in the tensor info).
* The RCO (rate-control optimisation) step is what produced the **non-uniform** assignment: a
  target whole-file budget (`Target whole-file bpw: 2.40`) is met by choosing a type per tensor
  while fixed float tensors stay as they are.
* The allocation file covers **shard 1 only**; shard 2 (PLE) is fixed IQ4_NL.
* "Section 2" is mentioned in `bench/results/2026-09-28-coder/README.md:27` for the *Coder*
  variant's file (kept-expert mapping). The Q2_0 file on this machine has no numbered sections —
  it is the flat `name: TYPE` form above.

**Who parses it:** nothing in either engine. Grep results:

* `/home/korphaus/projects/strata` — no reader; only prose references:
  `docs/DETAILS.md:374`, `bench/results/2026-09-28-coder/README.md:27`,
  `bench/results/2026-10-06-llamacpp-0.6.0/prompt-32k.txt:318`.
* `/home/korphaus/llama.cpp-qwen4exp-mmap` — no reference at all.
* Breccia's read-only copy: `Reference Projects/Strata/tools/nvfp4_verify_gguf.py:142`
  accepts one as `--reference` ("an rco-allocation.txt listing the reference tensor names") — the
  only code-level consumer found, and it only reads the names.

So the allocation file is **authoritative documentation, not a runtime input**; the equivalent
information is in the GGUF tensor info.

---

## 3. `packs/iq3_xxs/` — the strata pack

Built by `/home/korphaus/projects/strata/tools/iq_pack.py` (docstring at the top of that file is
the format's specification) and consumed by
`src/core/weights.cpp` (dense), `src/core/expert_source.cpp` (experts),
`src/kernels/ngram.cpp` (PLE). Selected by the engine as `--pack /srv/strata-data/packs/iq3_xxs`
(`strata-iq3_xxs.json:4-5`).

### 3.1 `index.txt` — 93,687 B, 1,081 lines (2 headers + 1,079 rows)

```
# strata pack index v3 -- generated by tools/iq_pack.py (native experts) from Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf
# align 64 pool 1538035200 tensors 1079
```

Header fields: `align` = arena alignment (64 B), `pool` = total bytes the rows need (1,538,035,200
= `dense.bin` size exactly), `tensors` = row count.

Each row is **space-separated, no quoting, exactly 19 fields** (the engine refuses any row that
does not parse to 19 — `src/core/weights.cpp:177`). Column names come from
`tools/pack_index.py:242-243` and the struct in `src/core/weights.cpp:34-44`:

```
# name file kind src_off src_bytes dst_off dst_bytes ne0 ne1 code_bits code_bias
#   group_elems codebook has_offset codes_bytes scales_bytes offset_bytes scales_fp16 act_kind
```

| Field | Meaning |
|---|---|
| `name` | GGUF tensor name (`blk.N.…`) |
| `file` | which pack file the bytes live in: `0` = `dense.bin`, `1` = `embd.bin`, `2` = `experts.bin`, `3` = `extra.bin` (`src/core/weights.cpp:66-73`, `tools/pack_index.py:77`); `iq_pack.py` writes `0` for every row of this pack |
| `kind` | what the loader must DO: `0` VERBATIM, `1` BF16-in-F32 (take high 16 bits), `2` F32 (4 B/elem), `3` F16-in-F32 (round-to-nearest-even), `4` BF16 raw 2 B, `5` F16 raw 2 B. Kinds 4/5 are the native pack's 16-bit forms and are remapped to 1/3 at load (`weights.cpp:184-188`) |
| `src_off`, `src_bytes` | offset/length in the source file (dense.bin) |
| `dst_off`, `dst_bytes` | offset/length in the engine arena; `dst_off` is written by the packer (`write_index` re-packs rows in index order at `align`) |
| `ne0`, `ne1` | shape, dim0 contiguous |
| `code_bits`, `code_bias`, `group_elems`, `codebook`, `has_offset`, `codes_bytes`, `scales_bytes`, `offset_bytes`, `scales_fp16`, `act_kind` | quant plane description; for this pack: `0 0 1 0 0 0 0 0 0 0` (dense rows) or `8 0 32 0 0 0 0 0 0 0` (GGUF-served rows) |

Row kinds actually present:

| kind | rows | meaning here |
|---:|---:|---|
| 0 (+`code_bits=8`) | 302 | quantised; **shape only** — served from the GGUF via `--native`. The engine tests `code_bits != 0` (`src/core/native_dense.cpp:178-180`) to keep them in the "skip" set |
| 2 | 292 | F32 copied from `dense.bin` |
| 4 | 484 | BF16 copied from `dense.bin` |
| 5 | 1 | F16 (`blk.1.ple_conv1d`) copied from `dense.bin` |
| **total** | **1079** | |

Coverage: 1,224 GGUF tensors − 144 routed-expert tensors (`ffn_gate/up/down_exps` ×48) −
`per_layer_token_embd` = **1,079**. The pack never duplicates the experts or the PLE table.

Example rows:

```
output.weight 0 0 0 0 0 0 2560 248320 8 0 32 0 0 0 0 0 0 0        <- GGUF-served
output_hc_norm.weight 0 2 6553600 40960 6553600 40960 10240 0 0 0 1 0 0 0 0 0 0 0
output_hc_down.weight 0 4 0 6553600 0 6553600 10240 320 0 0 1 0 0 0 0 0 0 0
```

### 3.2 `native_experts.txt` — 3,296 B, 1 header + 48 rows (v3)

Header (verbatim):

```
# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard]
#   (n_expert 512, total 42912972800; absolute offsets in
#    Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf, or in the named shard beside it)
0 17 20 0 1868800 1227833216 1473813376 755052416
1 16 42 956825600 1305600 2016152832 2235918592 1779301632
…
```

| Column | Meaning (verified) |
|---|---|
| `layer` | 0…47 |
| `gu_type` | ggml type id of `blk.L.ffn_gate_exps` / `ffn_up_exps` (both share a type) |
| `d_type` | ggml type id of `blk.L.ffn_down_exps` |
| `offset` | byte offset of this layer's block inside `experts.bin` (`offset[1] = 956,825,600 = 512 × 1,868,800` ✓) |
| `blob_bytes` | **per-expert** blob size = 1,638,400 × (2·bpe(gu_type) + bpe(d_type)); checked for layers 0 (1,868,800), 1 (1,305,600), 5 (1,408,000) ✓ |
| `gate_off`, `up_off`, `down_off` | **absolute file offsets** of the three expert tensors in the GGUF shard (data_start 11,024,256 + tensor offset; e.g. `blk.0.ffn_gate_exps` = 11,024,256 + 1,216,808,960 = 1,227,833,216 ✓) |
| `[shard]` | optional; absent here (all experts in shard 1). v4 uses a per-role `gate,up,down` field when a shard boundary falls inside a layer |

Global consistency: `Σ_layers 512 × blob_bytes = 42,912,972,800` = `experts.bin` size **exactly**.

### 3.3 `experts.bin` — 42,912,972,800 B (39.97 GiB)

Headerless raw blob stream (first 64 bytes are IQ code/scale bytes, no magic):

```
for layer in 0..47:
    for expert in 0..511:            # one blob = 512 × [gate rows | up rows | down rows]
        gate block-plane, up block-plane, down block-plane
```

* Blob size is **per layer** (types differ per layer); layout is the engine's expert blob:
  `include/strata/kernels/cpu/expert.hpp` — gate/up rows interleaved (`2r`=gate, `2r+1`=up) then
  down rows for the pure engine layout, or the raw GGUF slices for this "native" pack
  (`tools/iq_pack.py:1-20`, `src/core/expert_source.cpp:1066-1070`).
* `experts.bin.src.json` binds it to its source:

  ```json
  {"schema":1,
   "shards":[{"name":"…IQ3_XXS-00001-of-00002.gguf","size":47039860096},
             {"name":"…IQ3_XXS-00002-of-00002.gguf","size":28800138432}],
   "native_experts_sha256":"ae28cf7a2a1c9063c4b55d6c8cf86d4d079dcddcf597f9e8697c9157d462c0ee",
   "bytes":42912972800}
  ```
  An `experts.bin` is only reused when shard names/sizes and the `native_experts.txt` hash match.

### 3.4 `dense.bin` — 1,538,035,200 B (1.43 GiB)

Headerless; every tensor is placed by `index.txt` (`dst_off`, 64-byte aligned). Contents are the
292 F32 + 484 BF16 + 1 F16 tensors in **engine form** (first bytes are BF16 values, as expected for
the first non-native row `output_hc_down.weight`). `pool` in the header equals the file size.

### 3.5 `conversions.json` — 310 B

```json
{"schema":1, "tool":"tools/iq_pack.py", "compat_bf16":false,
 "source_shards":[{name,size} ×2], "tensors":[]}
```
`tensors: []` → nothing needed converting for this variant: the GSQ-RCO file already stores the
routers/hyper-connections/norms in the forms the engine wants (see `tools/iq_pack.py:60-66`).

### 3.6 `tokenizer/` — 5 files, 9,845,757 B

Exported from the GGUF by `tools/strata_tokenizer.py`; configured as
`"tokenizer": "/srv/strata-data/packs/iq3_xxs/tokenizer"` (`strata-iq3_xxs.json:31`).

| File | Bytes | Content |
|---|---:|---|
| `tokenizer.json` | 541 | manifest, **not** a HF tokenizer.json: `{"model":"gpt2","pre":"qwen35","vocab_size":248320,"n_merges":247587,"special_ids":{eos:248046,padding:248044,bos:248044},"add_bos_token":false,"pre_pattern":"(?:'[sS]|…)` (Qwen3 pre-tokenizer regex), "pre_pattern_source":"…llama-vocab.cpp L396 (LLAMA_VOCAB_PRE_TYPE_QWEN35)"}` |
| `vocab.json` | 5,737,005 | HF-style `{token: id}` map, 248,320 entries (`"!":0 …`) |
| `merges.txt` | 3,353,258 | 247,587 BPE merges, `Ġ Ġ` first (same order as `tokenizer.ggml.merges`) |
| `token_type.json` | 744,960 | JSON array of 248,320 token types (identical to `tokenizer.ggml.token_type`) |
| `chat_template.jinja` | 9,993 | byte-identical to `tokenizer.chat_template` in the GGUF |

---

## 4. `mtp/` — the multi-token-prediction (draft) head

The GSQ-RCO GGUF ships no MTP head; it was fetched from the BF16 checkpoint
(`tools/mtp_fetch.py:3`), packed (`tools/mtp_pack.py`) and reduced to runtime files
(`tools/mtp_rt.py`). Selected by `--mtp /srv/strata-data/mtp/rt`
(`strata-iq3_xxs.json:20-21`, plus `--batch-mtp`).

### 4.1 `mtp-inventory.md` / `mtp-inventory.json` / `mtp-manifest.json`

* `mtp-inventory.md` — human table of the 31 MTP tensors (5.214 GB total, 28 safetensors shards of
  `Qwen/Qwen3.8-Flash-Next@de4b8e4d…`), with dtype/shape/MB.
* `mtp-inventory.json` — same as JSON: `{repo, total_bytes: 5214301696, tensors:[{name, shard,
  dtype, shape, start, end, bytes}]}`.
* `mtp-manifest.json` — a **JSON array of 31 objects** (not an object): inventory fields plus
  `"file": "tensors/<name>.bin"` and `"sha256"`. This is the fetch manifest: byte ranges in the
  HF safetensors shard + the extracted file + its hash.

Key shapes: `mtp.fc_embedding.weight` `[2560,2560]`, `mtp.fc_hidden.weight` `[2560,2560]`,
`mtp.layers.0.mlp.experts.gate_up_proj` `[512,1280,2560]`,
`mtp.layers.0.mlp.experts.down_proj` `[512,2560,640]`,
`mtp.layers.0.self_attn.q_proj` `[12288,2560]`, `o_proj` `[2560,6144]`,
`mtp.layers.0.self_attn.indexer.index_qk_proj` `[640,2560]`, plus hyper-connection and shared
expert tensors — i.e. one extra decoder layer + fc/projection glue (HF naming).

### 4.2 `tensors/` — 31 raw BF16 `.bin`, 5,214,301,696 B

Exactly the bytes described by the manifest, no header. These are inputs to `mtp_pack.py` /
`mtp_rt.py`, not runtime files.

### 4.3 `mtp-q2_0.gguf` — 889,017,216 B (+ `.report.json`)

The MTP block re-emitted as a Strata GGUF (`tools/mtp_pack.py`): dense tensors BF16/F32, routed
experts (`gate_up_proj`, `down_proj`) in **Q2_0** (64-element blocks, grid `{-1,0,1,2}×d`,
`q2_0()` in `tools/mtp_pack.py`), same encoding as the main model. `.report.json` records, per
tensor, `{tensor, format, bytes}` plus `relative_rms_error_first_experts` for the expert tensors
(0.4236 / 0.4291) and `total_bytes: 889014272`.

### 4.4 `rt/` — the runtime files actually loaded (824,314,919 B)

| File | Bytes | Format |
|---|---:|---|
| `dense.txt` | 1,851 | 29 lines: `name kind rows cols offset bytes`, kind ∈ `q8_0 \| bf16 \| f32` (10 / 10 / 9) |
| `dense.bin` | 116,099,072 | concatenation of those rows at `offset` (last row ends exactly at EOF: 116,058,112 + 40,960) |
| `experts.bin` | 707,788,800 | 512 fixed blobs of 1,382,400 B = `3 × (2560×640×18/64)` (gate/up interleaved codes, down codes, then the two fp16 scale planes — `mtp_rt.py:blob_of`) |
| `draft_vocab.bin` | 425,196 | raw little-endian **int32 token ids**, 106,299 tokens, sorted (`0,1,2,3,…` at offset 0) |

`dense.txt` kinds and contents: the big projections (`fc_embedding`, `fc_hidden`, `q/k/v/o_proj`,
`shared_expert.*`, `indexer.index_qk_proj`) are **Q8_0** (ggml reference rounding, 34 B/32-elem
block, so `2560×2560 → 6,963,200` B ✓); hyper-connection and `mlp.gate` stay BF16; norms are F32
with the Gemma "+1" applied (`mtp_rt.py` docstring).

Consumption in `/home/korphaus/projects/strata`:

| File | Reader |
|---|---|
| `rt/dense.txt` | `src/core/mtp.cpp:270-278` — `is >> t.name >> t.kind >> t.rows >> t.cols >> t.off >> t.bytes`; malformed line is fatal |
| `rt/dense.bin` | `src/core/mtp.cpp:296` — whole-file read, uploaded to VRAM |
| `rt/experts.bin` | `src/core/mtp.cpp:311-331` — `n_expert × BLOB` bytes, 64 MiB chunks, `madvise(WILLNEED)` |
| `rt/draft_vocab.bin` | `src/core/mtp.cpp:637-660` — `size/4` tokens, rows gathered from the main head; default path `<mtp>/draft_vocab.bin` (`include/strata/core/mtp.hpp:207`), overridable with `--mtp-draft-vocab` (`src/program/generate.cpp:665,2011`) |
| presence checks | `src/program/generate.cpp:7644-7645` requires `dense.txt`, `experts.bin`, and `draft_vocab.bin` |

---

## 5. Q2_0 variant, mmproj, mtp-unsloth

### 5.1 `models/Q2_0/` (symlink target)

`/srv/strata-data/models/Q2_0` → `/mnt/agent-storage-b/ai-dev/ai-runtime/home/korphaus/models/qwen-dual/Q2_0`
(both are the same filesystem view; the directory holds the real files).

* `…Q2_0-00001-of-00002.gguf` — 37,623,740,192 B. Header: GGUF v3, 1,223 tensors, **73**
  metadata KVs. Identical qwen4exp/tokenizer values to IQ3_XXS, plus
  `general.basename`, `general.license` (`apache-2.0`), `general.base_model.count=1`,
  `general.base_model.0.*`, `general.description = "GSQ-RCO non-uniform mixed-precision GGUF
  (Deep Algorithms and Systems Lab, ISTA)"`, `general.file_type = 41` (`MOSTLY_Q2_0`).
  Types: Q2_0=202, BF16=483, F32=292, Q3_K=91, IQ4_XS=56, Q4_K=38, Q4_0=16, Q5_K=16, Q5_0=7,
  Q6_K=12, IQ4_NL=7, Q8_0=2, F16=1 (matches the allocation file header exactly).
* `…Q2_0-00002-of-00002.gguf` — a **symlink** to the IQ3_XXS shard 2 (the shared PLE table).

### 5.2 `mmproj-Qwen3.8-Flash-Next-BF16.gguf` — 907,543,008 B

GGUF v3, 334 tensors, 31 KVs, `general.architecture = clip`, `general.type = mmproj`,
`general.file_type = 32` (BF16), 224 F32 + 110 BF16 tensors (nothing quantised).

```
clip.has_vision_encoder=true  clip.vision.projection_dim=2560  image_size=768  patch_size=16
clip.vision.embedding_length=1152  feed_forward_length=4304  block_count=27  attention.head_count=16
clip.projector_type='qwen3vl_merger'  spatial_merge_size=2  use_gelu=true
clip.vision.image_mean=[0.5,0.5,0.5]  image_std=[0.5,0.5,0.5]
clip.vision.layer_norm_epsilon=1e-6  clip.vision.is_deepstack_layers=[false ×27]
```

Tensor patterns: `v.patch_embd.weight/.1`, `v.patch_embd.bias`, `v.position_embd.weight`,
`v.post_ln.{weight,bias}`, `v.blk.0..26.{attn_qkv,attn_out,ffn_up,ffn_down}.{weight,bias}` and
`v.blk.N.{ln1,ln2}.{weight,bias}` (27 each), `mm.0.*` / `mm.1.*` merger `{weight,bias}` (2+2).
Configured as `"mmproj": "/srv/strata-data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf"`.

### 5.3 `mtp-unsloth/` — alternative draft head

`MTP/mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, 4,137,429,120 B + `dl.log` (253 B, HF download output).
GGUF v3, 34 tensors, 51 KVs, `general.architecture = qwen4exp`, `general.name = Ckpt_Q38`,
`general.size_label = 512x1.4B`, `general.file_type = 7` (`MOSTLY_Q8_0`), and notably:

* `qwen4exp.block_count = 49` (48 + the MTP block) and `qwen4exp.nextn_predict_layers = 1`;
* `qwen4exp.attention.compress_ratios` has 49 entries ending `…,4,4`;
* tensors: `token_embd.weight` + `output.weight` (Q8_0), and the whole 49th block
  (`blk.48.*`, all Q8_0 except F32/BF16) including `blk.48.nextn.{eh_proj,enorm,hnorm,
  hc_head_down,hc_head_up,hc_head_norm}.weight`;
* the same tokenizer KV set (248,320 tokens).

It is a self-contained "embedding + output + MTP layer" GGUF — a different packaging from
`mtp/rt/`, and not wired into `strata-iq3_xxs.json`.

---

## 6. Source references (where the format is defined)

**GGUF reader / types**

* `tools/gguf_reader.py:1-25` — standalone GGUF v3 header reader written because gguf-py has no
  type 42; contains the `GGML_TYPES` table and `BLOCK_GEOMETRY`.
* `include/strata/artifact/gguf_reader.hpp` — header-only reader used by the engine:
  `ggml_type_name()` (:53), `block_geometry()` (:122, incl. `Q2_0 = 64/18`), `MetaType` (:265),
  `check_architecture()` (:588), `tensor_data() = base + data_start + offset` (:373).
* `src/artifact/gguf_reader.cpp` — `gguf_reader <file> [--check]` CLI: prints version/tensors/
  metadata/`data_start`, counts types, brackets every tensor's byte size against its offset gap
  (<32 B padding), runs the architecture guard.
* `tools/quantscope.py:113-124` — `GGML_TYPES` with `42: "Q2_0"  # ISTA-DASLab GSQ-RCO 2-bit
  (Strata, llama.cpp fork type)`.
* llama.cpp fork: `ggml/include/ggml.h:390-433` (type ids), `include/llama.h:119-160`
  (`llama_ftype`, incl. `MOSTLY_Q2_0 = 41`), `include/llama.h:43` (`GGSQ` state magic — unrelated
  to the model format).

**GSQ-RCO / RCO / allocation-file references** (grep results, all read-only):

| File | What it says |
|---|---|
| `strata/docs/DETAILS.md:367-374` | GSQ-RCO Coder description; the release's `rco-allocation.txt` maps the shipped expert ranking onto kept experts |
| `strata/bench/results/2026-09-28-coder/README.md:26-30` | "section 2 of the release's `tensor-allocation/*.rco-allocation.txt`" used to build `data/expert-profile-coder.bin` |
| `strata/bench/results/2026-10-06-llamacpp-0.6.0/prompt-32k.txt:318` | same prose in a captured prompt |
| `Breccia/Reference Projects/Strata/tools/nvfp4_verify_gguf.py:142` | only code-level consumer: `--reference` accepts an `rco-allocation.txt` for tensor *names* |
| `strata/tools/gguf_reader.py:19`, `strata/tools/quantscope.py:123`, `strata/tools/iq_pack.py:64`, `strata/tools/make_profile.py:22`, `strata/tools/strata_inspect.py:35-42` | GSQ-RCO/Q2_0 as a *variant* marker, no allocation parsing |
| `llama.cpp-qwen4exp-mmap` | **no** RCO/allocation references (only `LLAMA_FILE_MAGIC_GGSQ` in `include/llama.h:43`) |

**Pack / MTP consumption**

* `tools/iq_pack.py` — pack writer and format docstring (`index.txt`, `dense.bin`,
  `native_experts.txt`, `experts.bin`, `conversions.json`, `NOT_IN_PACK = per_layer_token_embd`).
* `tools/pack_index.py:1-60` — why the index is text, the 18/19 column list, `kind` semantics,
  the fp16-scale-plane trap.
* `src/core/weights.cpp:34-44` (`IndexRow`), `:91-121` (`pool_bytes`), `:123-145`
  (`index_code_bits`), `:146-215` (`load`, strict 19-field parse, kind 4/5 remap).
* `src/core/expert_source.cpp:1066-1110` (native GGUF expert mapping), `:4032-4055`
  (fingerprint over `manifest.json`/`index.txt`/`native_experts.txt` + sampled experts),
  `:4061-4068` (`expert_gguf_file`, per-layer shard column).
* `tools/mtp_pack.py`, `tools/mtp_rt.py`, `tools/draft_vocab.py`, `src/core/mtp.cpp:270-331,
  637-660`, `include/strata/core/mtp.hpp:66,207,253`, `src/program/generate.cpp:665,809,2011,
  7644-7645`.
* PLE: `src/kernels/ngram.cpp:190-200,242-267` (`per_layer_token_embd.weight`, accepted formats
  IQ4_NL / Q5_0 / I8-FP8 / Q5_1 / Q8_0 / Q4_0 / BF16), `src/program/generate.cpp:2421-2454`
  (`--ple-gguf` defaults to the shard that contains `per_layer_token_embd.weight`).
* Architecture keys: `llama.cpp conversion/qwen4exp.py:70-150` (writer side),
  `llama.cpp src/models/qwen4exp.cpp:33-83,148,223-260` (reader side, incl. QSA compress-ratio
  rules and tensor creation).

---

## 7. What a new engine must parse — checklist

### A. GGUF container (both shards of the model, plus mmproj / any variant)

1. Magic `GGUF`, **version 3**, `u64 tensor_count`, `u64 metadata_kv_count`.
2. All 13 value types incl. nested arrays (array = `u32 elem_type + u64 count + values`), with
   `bool` (1 B) and UTF-8 strings (`u64 len + bytes`).
3. `general.architecture == "qwen4exp"` (or `clip` for mmproj) — reject anything else.
4. Tensor info list: name, `u32 n_dims`, `u64 dims[]` (dim 0 contiguous), `u32 type`, `u64 offset`.
5. `data_start = align(offset_after_tensor_info, 32)`; tensor bytes = `base + data_start + offset`;
   bracket each tensor's size against the next offset (padding < 32 B) as a sanity check.
6. Split rules: `split.no/count/tensors.count`, each shard's own header, total = 1224.
7. ggml type ids incl. **42 = Q2_0 (64 elems / 18 B)** and the block geometry table of §1.6 —
   14 types for IQ3_XXS, 18 for Q2_0. Read the type **per tensor**; never infer from the name.

### B. `qwen4exp` semantics (from metadata, not from tensor names)

8. Geometry: `block_count=48`, `context_length=262144`, `embedding_length=2560`,
   `embedding_length_per_layer_input=160`, `head_count=24`, `head_count_kv=2`,
   `key_length=value_length=256`, `layer_norm_rms_epsilon`.
9. MoE: `expert_count=512`, `expert_used_count=10`,
   `expert_feed_forward_length=640`, `expert_shared_feed_forward_length=640`.
10. Layer roles: `full_attention_interval=4` **and** `attention.compress_ratios[48]`
    (ratio 4 on layers ≡3 mod 4 → 12 QSA layers with `attn_q/k/v` + `indexer`; 36 SSM layers with
    `attn_qkv` + `ssm_*`). Validate `top_k % ratio == 0` and one shared ratio.
11. SSM: `conv_kernel=4`, `state_size=128`, `group_count=16`, `time_step_rank=48`,
    `inner_size=6144`.
12. Hyper connections: `count=4`, `low_rank=320` → the `hc_*` tensor shapes
    (`[10240,320]`, `[320,10240]`, `[10240,4]`, `[10240]`; 10240 = 4×2560).
13. Indexer/QSA: `head_count=4`, `key_length=128`, `top_k=2048`.
14. Rope: `freq_base=1e7`, `dimension_count=64`, `dimension_sections=[11,11,10,0]`.
15. PLE: `ple.layers=[1]`, `ngram_size=3`, `heads_per_ngram=8`, `conv_kernel=4`,
    `eos_token_id`, `image_token_id`, `layer_multipliers/head_offsets/head_vocab_sizes` as opaque
    int64 hash constants, `embedding_length_per_layer_input=160`.
16. Tokenizer: `gpt2` + `qwen35` pre-type, 248,320 tokens, 247,587 merges, token types,
    bos/eos/pad = 248044/248046/248044, `add_bos_token=false`, the Qwen3 pre-tokenizer regex,
    and the Jinja chat template.
17. `general.file_type` is an `llama_ftype`, useful only for labelling
    (23 = IQ3_XXS, 41 = Q2_0, 32 = BF16, 7 = Q8_0).

### C. Model-specific tensor wiring

18. Embedding head: `token_embd.weight` + `output.weight` (both `[2560,248320]`, different quants)
    + the `output_hc_*` hyper-connection on the final layer.
19. Block tensor sets: 25 tensors (SSM), 26 (QSA), 31 (block 1 with PLE), 6 global + PLE table.
20. `per_layer_token_embd.weight` lives in shard 2, `[160, 320001536]`, IQ4_NL
    (16 × 20,000,096 rows of 160 dims) — needs its own dequant path; not in the pack.
21. Routed experts are `[2560,640,512]` / `[640,2560,512]` with the expert axis **last**; the
    router `ffn_gate_inp` is `[2560,512]` BF16, shared-expert router is `[2560]`.

### D. Optional: the strata pack (only if not serving experts/PLE from the GGUF)

22. `index.txt`: two `#` headers (`align`, `pool`, `tensors`) then 19-field rows; refuse any row
    that does not parse; remap `kind 4→1`, `5→3`; `code_bits != 0` ⇒ served from the GGUF.
23. `dense.bin`: headerless, rows at `dst_off`, 64-byte alignment.
24. `native_experts.txt` v3 (optional `[shard]`, v4 adds per-role shard): `layer gu_type d_type
    offset blob_bytes gate_off up_off down_off`; verify `Σ 512×blob_bytes == experts.bin size`.
25. `experts.bin` + `experts.bin.src.json` (schema/shard sizes/`native_experts_sha256`/bytes).
26. `conversions.json` (schema 1) — must be checked if any conversion was recorded.
27. `tokenizer/` (`tokenizer.json` manifest → vocab/merges/token_type/template).

### E. Optional: MTP draft head

28. `rt/dense.txt` (`name kind rows cols offset bytes`) → `rt/dense.bin` (kinds q8_0/bf16/f32).
29. `rt/experts.bin` = `n_expert × 1,382,400` B fixed blobs (Q2_0 codes + fp16 scales,
    gate/up interleaved then down).
30. `rt/draft_vocab.bin` = raw int32 token ids, `size/4` rows (106,299).
31. Provenance files (`mtp-manifest.json` sha256/ranges) are for rebuilds, not for loading.
32. `mtp-q2_0.gguf` and `mtp-unsloth/MTP/…Q8_0.gguf` are alternative packagings of the same head;
    the unsloth one declares `block_count=49` + `nextn_predict_layers=1` and carries its own
    `token_embd`/`output`.
