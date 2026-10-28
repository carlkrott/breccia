# qwen4exp (Qwen3.8-Flash-Next GSQ-RCO, "512x56B") - implementation-ready architecture spec

Target: a new inference engine (Breccia) implementing this model from scratch.

Everything below is derived from code and artifacts on this machine, cited as `path:line`.
Two roots are used throughout:

* **LC** = `<LLAMA_ROOT>` (llama.cpp fork, HIP/Vulkan builds at
  `build-hip-q5kv`, `build-hip-mmap`)
* **ST** = `<STRATA_ROOT>` (C++/HIP engine that runs this model in production)

and the artifact:

* **GGUF** = `/srv/strata-data/models/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000{1,2}-of-00002.gguf`
* **MTP** = `/srv/strata-data/mtp/` (`mtp-q2_0.gguf`, `mtp-inventory.{md,json}`, `rt/`, `tensors/`)

Every shape below is written in **ggml `ne` order**, i.e. `ne[0]` is the fastest-varying axis
(that is also how `gguf-py` reports `ReaderTensor.shape`). A linear weight stored as HF
`[out, in]` appears as `[in, out]`.

---

## 1. Verified artifact metadata

Dumped directly from shard 1 (67 KV entries) and cross-checked against both loaders.

| GGUF key (arch prefix `qwen4exp.`) | value | consumed at |
|---|---|---|
| `block_count` | 48 | `LC/src/llama-model.cpp` |
| `context_length` | 262144 | |
| `embedding_length` | 2560 | |
| `attention.head_count` | 24 | |
| `attention.head_count_kv` | 2 | |
| `attention.key_length` / `value_length` | 256 / 256 (head dim) | |
| `attention.layer_norm_rms_epsilon` | 1e-6 | `LC/src/models/qwen4exp.cpp:29` |
| `rope.dimension_count` | 64 (rotated dims of the 256) | |
| `rope.dimension_sections` | `[11, 11, 10, 0]` | `qwen4exp.cpp:31` |
| `rope.freq_base` | 10000000.0 | |
| `expert_count` / `expert_used_count` | 512 / 10 | |
| `expert_feed_forward_length` | 640 | `qwen4exp.cpp:27` |
| `expert_shared_feed_forward_length` | 640 | `qwen4exp.cpp:28` |
| `ssm.conv_kernel` | 4 | `qwen4exp.cpp:33` |
| `ssm.state_size` | 128 | `qwen4exp.cpp:35` |
| `ssm.group_count` | 16 | `qwen4exp.cpp:37` |
| `ssm.time_step_rank` | 48 | `qwen4exp.cpp:36` |
| `ssm.inner_size` | 6144 | `qwen4exp.cpp:34` |
| `full_attention_interval` | 4 | `qwen4exp.cpp:151` |
| `hyper_connection.count` | 4 | `qwen4exp.cpp:45` |
| `hyper_connection.low_rank` | 320 | `qwen4exp.cpp:46` |
| `attention.indexer.head_count` | 4 | `qwen4exp.cpp:56` |
| `attention.indexer.key_length` | 128 | `qwen4exp.cpp:57` |
| `attention.indexer.top_k` | 2048 | `qwen4exp.cpp:58` |
| `attention.compress_ratios` | `[0,0,0,4,0,0,0,4,...]`, 48 entries, `4` at layers 3,7,...,47 | `qwen4exp.cpp:62` |
| `ple.layers` | `[1]` (0-based) | `qwen4exp.cpp:90` |
| `ple.ngram_size` / `ple.heads_per_ngram` | 3 / 8 | `qwen4exp.cpp:106-107` |
| `ple.conv_kernel` | 4 | `qwen4exp.cpp:108` |
| `ple.eos_token_id` | 248044 | `qwen4exp.cpp:109` |
| `embedding_length_per_layer_input` | 160 | `qwen4exp.cpp:112` |
| `ple.layer_multipliers` | `[23703573157769, 20109073645365, 8052911324071]` (int64) | `qwen4exp.cpp:129` |
| `ple.head_offsets` | 16 int64, `0, 20000003, 40000026, ...` | `qwen4exp.cpp:134` |
| `ple.head_vocab_sizes` | 16 int64, `20000003, 20000023, 20000033, ...` | `qwen4exp.cpp:135` |
| `tokenizer.ggml.model` / `.pre` | `gpt2` / `qwen35`, vocab 248320 | |

Notes:

* There is **no** `nextn_predict_layers` key and **no `blk.48.*` tensor** in either shard, so the
  main GGUF carries **no MTP block** (`LC/src/llama-model.cpp:1380`, `qwen4exp.cpp:290-301`).
  The MTP head ships separately (section 11).
* Shard 2 contains exactly one tensor: `per_layer_token_embd.weight`, ne `[160, 320001536]`,
  IQ4_NL (the PLE table, ~28.8 GB).
* Model file name says IQ3_XXS; quantisation is **mixed per tensor and per layer** (section 6.4).

---

## 2. Global geometry

`ST/include/strata/core/layout.hpp:27-51` is the canonical, load-time-checked statement:

```
n_embd            = 2560        hc (hyper-connection streams) = 4
n_layers          = 48          hc_dim = hc * n_embd          = 10240
qsa_interval      = 4           hc_lr  (HC bottleneck rank)   = 320

GDN layers (36):                 Full-attention / QSA layers (12):
  ssm_state_size  = 128            n_head     = 24
  ssm_k_heads     = 16             n_head_kv  = 2
  ssm_v_heads     = 48             head_dim   = 256   (= key/value_length)
  ssm_d_conv      = 4              n_rot      = 64    (= rope.dimension_count)
  ssm_conv_channels = 10240        idx_n_head = 4     idx_key_dim = 128
  ssm_value_dim   = 6144           idx_block  = 4 (compress ratio)  idx_top_k = 2048
  conv_dim = 2*(128*16) + 128*48 = 10240

MoE (all 48 layers): n_expert = 512, k = 10, n_ff = 640 (routed and shared)
PLE: layer 1 only, ngram 3, heads_per_ngram 8 -> 16 heads, head_dim 160
```

Derived identities the new engine should assert at load time:

* `conv_dim = 2*ssm_k_heads*state + ssm_v_heads*state = 10240` (`ST/src/core/layer.cpp:225`,
  `qwen4exp.cpp:223-229`)
* `ssm_v_heads * state = 6144 = ssm.inner_size` (asserted `qwen4exp.cpp:1032`,
  `ST/src/core/layer.cpp`)
* `ple_n_heads = (ngram_size-1) * heads_per_ngram = 16`; `16 * 160 = 2560`
* `hc_lr = 320 = 10240 / 32 = hc_dim / 32`

---

## 3. Layer taxonomy

Layer index is 0-based. The rule is derived, not a list:

```
is_gdn(i)   <=>  (i + 1) % 4 != 0        // LC/src/models/qwen4exp.cpp:149-156
is_qsa(i)   <=>  i % 4 == 3              // ST/include/strata/core/layout.hpp:56-65
```

* **36 GDN layers**: 0,1,2, 4,5,6, 8,9,10, ... (gated-delta-net linear attention)
* **12 full-attention layers**: 3, 7, 11, ..., **47** (GQA attention + attention indexer/QSA)
  Both facts are confirmed by the tensor inventory: `attn_qkv` appears 36 times,
  `attn_q`/`indexer.q_proj` 12 times.
* **PLE** lives on layer 1 only, and layer 1 is a GDN layer (enforced `qwen4exp.cpp:159-163`).
* "full attention" means **"not a linear-attention layer"** - it does *not* mean dense attention
  over the whole context: above 2051 cached cells the layer is sparse (section 10.4).

---

## 4. One transformer block - dataflow

### 4.0 Trunk skeleton (identical for both layer types)

Source of truth: `LC/src/models/qwen4exp.cpp:427-521`, mirrored by
`ST/src/core/layer.cpp:1331-1504` (`block_layer_pre` / `block_layer_post`).

```
 inpL          = embedding(tok_embd)                        [2560]            (427)
 res_hc        = repeat(inpL, hc)          -> [n_embd, hc, T] = [2560,4,T]   (441-447)
                                                             == 4 copies of the embedding

 for i in 0..47:
   (A) if PLE(i):  res_hc = PLE(res_hc, ple_emb)            // layer 1 only   (452-454)

   (B) [mixed, inject] = HC_READ(res_hc, hc_attn_{norm,down,up,inject})       (457-462)

   (C) if GDN(i): block_out = GDN(mixed)     else  block_out = QSA(mixed)    (466-470)

   (D) res_hc = HC_WRITE(res_hc, block_out, inject)          // per-stream     (482)

   (E) [mixed, inject] = HC_READ(res_hc, hc_ffn_{norm,down,up,inject})        (484-489)

   (F) block_out = MoE_ffn(mixed)                            // routed + shared (491)

   (G) res_hc = HC_WRITE(res_hc, block_out, inject)                            (494)

   (H) optional: crop rows with inp_out_ids (only around i = 47)              (472-480)

 h_nextn       = flatten(res_hc)   -> [hc*n_embd, T] = [10240, T]  (MTP input)(501)
 cur           = HC_READ_FINAL(res_hc, output_hc_*)   -> [2560, T] (512-517)
 logits        = output @ cur                          -> [248320, T]       (519-521)
```

There is **no `output_norm`**: the final hyper-connection mixer *is* the output norm
(`qwen4exp.cpp:184`, `511-517`; `ST/src/core/layer.cpp:1249-1267`).

### 4.1 HC_READ / HC_WRITE (hyper-connections)

`LC/src/models/qwen4exp.cpp:313-397`; transcribed verbatim by
`ST/include/strata/kernels/gr.hpp:77-124`.

```
HC_READ (build_hc_mix)                                  input R [hc, n_embd]
  xn  = rmsnorm(R, eps) * w_norm                         RMS over ONE stream (2560 dims), then
                                                         gamma [n_embd, hc] applied elementwise
  lo  = silu( (xn @ w_down.T) / hc )                    w_down: [hc_dim, hc_lr] = [10240,320]
  gate= lo @ w_up.T                                    w_up:   [hc_lr, hc_dim] = [320,10240]
  gated = xn * sigmoid(gate)                            elementwise, per stream
  mixed = mean_over_streams(gated)                      -> [n_embd]   (the mixer's input)
  inject = xn @ w_inject.T                             w_inject: [hc_dim, hc] = [10240,4] -> [hc]
                                                        (inject = null for the FINAL mixer)

HC_WRITE (build_hc_combine)                             R_out [hc, n_embd]
  w[c]   = 2 * sigmoid(inject[c] / hc)                  centres the gate on 1
  R_out[c] = R[c] + block_out * w[c]                    the SAME block_out added to every stream
```

`gr.hpp:83` states the HC gammas arrive pre-folded as `(1 + w)` from the converter, and
`qwen4exp.cpp:326` says the same. See open question Q1.

### 4.2 GDN (linear-attention) layer

Spec order: `ST/include/strata/core/layer.hpp:8-32` (transcription of
`ref/gdn.py::gdn_decode`); implementation `ST/src/core/layer.cpp:225-335` and
`LC/src/models/qwen4exp.cpp:1015-1140`.

```
 x (2560)  --GEMV-->  qkv = attn_qkv @ x                  [10240]      layer.cpp:243
                      attn_gate @ x -> z                  [6144]       layer.cpp:321
                      ssm_beta  @ x -> beta  -> sigmoid   [48]
                      ssm_alpha @ x -> alpha              [48]

 qkv = [ q(2048) | k(2048) | v(6144) ]   128-wide heads:
       q: 16 heads x 128 | k: 16 heads x 128 | v: 48 heads x 128
 conv_input = concat(conv_state[3, 10240], qkv)           [4, 10240]
 h          = silu( conv1d(conv_input) )                  SiLU on the WHOLE 10240 first
              (qwen4exp.cpp:1074-1078; layer.hpp:11,21)
 q,k = l2_normalize(h_q), l2_normalize(h_k)   // v is NOT normalised
              (models.h:14-18: rms_norm(eps/n) * 1/sqrt(n) == x / sqrt(sum x^2 + eps))
 gate = softplus(alpha + ssm_dt) * ssm_a      // ssm_a is NEGATIVE (=-exp(A_log))
              (qwen4exp.cpp:1049-1056)       // verified: all 48 ssm_a values < 0
 y    = delta_rule(q,k,v,gate,beta,state)     // section 5.2
 out  = rmsnorm(y, ssm_norm) * sigmoid(z)     // SIGMOID here, qwen3.5 used SiLU
              (qwen4exp.cpp:634-644, layer.hpp:16)
 o    = ssm_out @ out                         [6144 -> 2560]
```

Head pairing: v-head `h` uses key/query head `h % 16` (**modulo**, not interleaved) -
`ST/include/strata/kernels/gdn.hpp:54-57`, and `qwen4exp.cpp:1114-1118` tiles the 16 q/k
heads up to 48 to match on the unfused path.

### 4.3 Full-attention (QSA) layer

`LC/src/models/qwen4exp.cpp:928-1013` (+ `738-867` selection, `870-926` masked attention);
`ST/src/core/layer.cpp:1026-1180`.

```
 (0) sel = INDEXER_SELECT(x, ...)          // only when compress_ratio > 0   (942)
 (1) qkv_full = attn_q @ x                 [12288] = 24 heads * 2 * 256     (945)
     Q = first 256 of each head's 512 block                                (948-951)
     gate = second 256 of each head's block                                (966-970)
 (2) Q = rmsnorm(Q, attn_q_norm)           [256] per head                   (953)
 (3) K = attn_k @ x  -> rmsnorm(K, attn_k_norm)   [512] = 2 heads * 256     (956-964)
     V = attn_v @ x                        [512]                           (959)
 (4) Q = IMRoPE(Q, pos);  K = IMRoPE(K, pos)   n_rot 64, base 1e7           (976-986)
 (5) A = masked_attention(Q, K, V, mask = sel (+ causal))   scale 1/sqrt(256)(992-1000)
 (6) A = A * sigmoid(gate)                 <-- gate BEFORE the out-proj     (1003-1007)
 (7) out = attn_output @ A                 [6144 -> 2560]                   (1009)
```

### 4.4 PLE insertion (layer 1 only)

`qwen4exp.cpp:452-454`, `ST/src/core/layer.cpp:1332-1388`. It is a **residual update on the
wide stack that happens before layer 1's own HC_READ**, i.e. "the gathered rows arrive before
layer 2" (`layer.cpp:1336-1338`). Section 9 has the math.

### 4.5 MTP branch

`t_h_nextn` is taken from the trunk residual **before** the final mixer (`qwen4exp.cpp:500-503`),
so the draft head sees the raw `[10240, T]` stack, not the mixed `[2560, T]`.

---

## 5. Op-by-op math

### 5.1 Where the block's two halves sit (ordering is the contract)

`ST/src/core/layer.cpp:1331-1504` splits a block into `pre` and `post` so a CPU expert pool can
overlap; the stage order is fixed:

```
stage 0  HC_READ  (hc_attn_*)        layer.cpp:1406-1418
stage 1  token mixer: qsa_layer | gdn_layer   layer.cpp:1419-1421
stage 2  HC_WRITE (attn half)        layer.cpp:1422
stage 3  HC_READ  (hc_ffn_*)         layer.cpp:1429-1441
stage 4  router (+ optional shared expert) layer.cpp:1442-1446
--- host expert pool runs here ---
stage 5  combine                     layer.cpp:1483-1486
stage 6  HC_WRITE (ffn half)         layer.cpp:1487-1504
```

`bb.mixed` and `bb.inject` written by stage 3 must survive until stage 5 (`layer.cpp:1481-1482`).

### 5.2 GDN step (delta rule)

Two implementations, same math:

* **Decode (T=1)** `LC/src/models/delta-net-base.cpp:286-368` (`build_delta_net_autoregressive`)
* **Prefill (T>1)** `delta-net-base.cpp:16-284` (chunked, chunk size 64, `:61`) or a fused op
  (`:370-420`, `ggml_gated_delta_net`, `LC/ggml/include/ggml.h:2639-2658`)

Per step, per v-head `h` with `k/q` head `idx = h % 16`, state `S[i][j][h]` (128x128):

```
 dec_h    = exp(gate_h)                                gate_h < 0, so dec in (0,1)
 S       *= dec_h                                      decay FIRST
 sk[j]    = sum_i S[i][j][h] * k[idx][i]
 d[j]     = (v[h][j] - sk[j]) * beta_h
 S[i][j][h] += k[idx][i] * d[j]                        rank-1 update
 o[h][j]  = sum_i S[i][j][h] * q[idx][i]
 q is scaled by 1/sqrt(S) = 1/sqrt(128) AFTER the L2 norm
        (ST/include/strata/kernels/gdn.hpp:19-31, LC delta-net-base.cpp:316-318)
```

Layout choice: strata stores `S` as `(S, h_v, S)` for coalescing, llama.cpp as
`(S, S, H_v, n_seqs)` (`gdn.hpp:13-18`; `qwen4exp.cpp:1070-1071`). Both index the same three
numbers - pick one and assert it in tests.

Conv history: `conv_state [d_conv-1, channels] = [3, 10240]`, oldest first, row-fastest
(`gdn.hpp:60-70`; `qwen4exp.cpp:1295-1347` builds `concat(state, x^T)` then writes the new tail
back, per rollback slot).

### 5.3 Attention

Standard GQA softmax attention, `n_head=24`, `n_head_kv=2`, `head_dim=256`,
`scale = 1/sqrt(256)` (`qwen4exp.cpp:992`), causal, plus the indexer selection mask (section
10.4). The per-head **output gate** is `sigmoid(gate)` where `gate` is the second half of the
`attn_q` projection (`qwen4exp.cpp:1003-1007`).

### 5.4 MoE FFN

`qwen4exp.cpp:1142-1190` calls `build_moe_ffn` (`LC/src/llama-graph.cpp:2121-2280`):

```
 logits  = ffn_gate_inp @ x                    [512]   (BF16 activation in strata, see 14)
 probs   = softmax(logits)  over ALL 512               llama-graph.cpp:2171
 ids     = topk(probs, 10)                            llama-graph.cpp:2237
 w       = probs[ids]                                 llama-graph.cpp:2251
 w      /= clamp(sum(w), 6.103515625e-5 = 2^-14, inf) llama-graph.cpp:2265-2273
 out     = sum_{j in ids} w_j * down_j( silu(gate_j @ x) * (up_j @ x) )
 shared  = sigmoid(ffn_gate_inp_shexp @ x) * down_s( silu(gate_s @ x) * (up_s @ x) )
 out    += shared                                     qwen4exp.cpp:1163-1184
```

`expert_weights_scale` is absent from the file and defaults to `0.0`
(`LC/src/llama-hparams.h:134`), so the `w_scale != 0` branch is skipped
(`llama-graph.cpp:2274`). No expert groups (`n_expert_groups == 1`).
strata's independent kernel states the same contract: softmax over all experts, then
`max(sum, 2^-14)` renormalisation (`ST/src/kernels/cuda/router_top10.cu:3-17,108`).

---

## 6. Tensor inventory

GGUF names are `blk.<i>.<suffix>`. Types are `GGML_TYPE_*` ids read from the file
(`LC/ggml/include/ggml.h` enum).

### 6.1 Shared by every layer (48 each)

| suffix | ne | role | type (blk.0) |
|---|---|---|---|
| `hc_attn_norm.weight` | `[10240]` (semantically `[2560,4]`) | HC read gammas, per stream | F32 |
| `hc_attn_down.weight` | `[10240, 320]` | HC bottleneck down | BF16 |
| `hc_attn_up.weight` | `[320, 10240]` | HC bottleneck up | BF16 |
| `hc_attn_inject.weight` | `[10240, 4]` | per-stream scatter weights | BF16 |
| `hc_ffn_{norm,down,up,inject}` | same four | same, FFN half | F32/BF16 |
| `ffn_gate_inp.weight` | `[2560, 512]` | MoE router | BF16 |
| `ffn_gate_exps.weight` | `[2560, 640, 512]` | expert gate | IQ4_NL / Q2_0 ... |
| `ffn_up_exps.weight` | `[2560, 640, 512]` | expert up | IQ2_XS |
| `ffn_down_exps.weight` | `[640, 2560, 512]` | expert down | IQ4_NL / Q2_0 ... |
| `ffn_gate_inp_shexp.weight` | `[2560]` | shared-expert scalar gate | BF16 |
| `ffn_gate_shexp.weight` | `[2560, 640]` | shared expert | IQ4_XS |
| `ffn_up_shexp.weight` | `[2560, 640]` | shared expert | IQ4_XS |
| `ffn_down_shexp.weight` | `[640, 2560]` | shared expert | IQ4_XS |

Loaded at `qwen4exp.cpp:232-282`.

### 6.2 GDN layers only (36 each: 0,1,2,4,5,...,46)

| suffix | ne | role |
|---|---|---|
| `attn_qkv.weight` | `[2560, 10240]` | q(2048) \| k(2048) \| v(6144) |
| `attn_gate.weight` | `[2560, 6144]` | output gate `z` |
| `ssm_conv1d.weight` | `[4, 10240]` | depthwise causal conv |
| `ssm_dt.bias` | `[48]` | time-step bias, F32 |
| `ssm_a` | `[48]` | `-exp(A_log)`, F32, **all negative** (verified) |
| `ssm_alpha.weight` | `[2560, 48]` | alpha projection, BF16 |
| `ssm_beta.weight` | `[2560, 48]` | beta projection, BF16 |
| `ssm_norm.weight` | `[128]` | post-recurrence RMS gamma, F32 |
| `ssm_out.weight` | `[6144, 2560]` | output projection |

`qwen4exp.cpp:254-264`.

### 6.3 Full-attention layers only (12 each: 3,7,...,47)

| suffix | ne | role |
|---|---|---|
| `attn_q.weight` | `[2560, 12288]` | q\|gate interleaved per head (2 x 256 x 24) |
| `attn_k.weight` | `[2560, 512]` | 2 kv heads x 256 |
| `attn_v.weight` | `[2560, 512]` | |
| `attn_q_norm.weight` | `[256]` | per-head q RMS gamma |
| `attn_k_norm.weight` | `[256]` | per-head k RMS gamma |
| `attn_output.weight` | `[6144, 2560]` | 24 x 256 -> 2560 |
| `indexer.q_proj.weight` | `[2560, 512]` | 4 heads x 128 (BF16) |
| `indexer.k_proj.weight` | `[2560, 128]` | **ONE** shared key head (BF16) |
| `indexer.q_norm.weight` / `indexer.k_norm.weight` | `[128]` | RMS gammas (converter adds +1) |

`qwen4exp.cpp:241-253`. `indexer.head_count = 4` counts **query** heads only; the key is a
single 128-wide head (`ST/include/strata/core/layout.hpp:6-10`).

### 6.4 Layer 1 extra (PLE)

| suffix | ne |
|---|---|
| `ple_key.weight` | `[2560, 10240]` (in = 16 heads x 160) |
| `ple_value.weight` | `[2560, 2560]` |
| `ple_norm_key.weight` / `ple_norm_query.weight` / `ple_norm_conv.weight` | `[10240]` F32 |
| `ple_conv1d.weight` | `[4, 10240]` F16, flat index `k + 4*c` |

`qwen4exp.cpp:266-273`.

### 6.5 Model-level

| name | ne | type | notes |
|---|---|---|---|
| `token_embd.weight` | `[2560, 248320]` | IQ3_S (3.438 bpw) | |
| `output.weight` | `[2560, 248320]` | Q5_K (5.5 bpw) | **separate from the embedding - do not tie** |
| `output_hc_norm.weight` | `[10240]` | F32 | final mixer gamma (= output norm) |
| `output_hc_down.weight` | `[10240, 320]` | BF16 | |
| `output_hc_up.weight` | `[320, 10240]` | BF16 | |
| `per_layer_token_embd.weight` | `[160, 320001536]` | IQ4_NL | PLE table, shard 2 |

Quantisation is **heterogeneous**: e.g. `blk.1.ffn_down_exps` is Q2_0 (2.25 bpw) while
`blk.0`/`blk.47` are IQ4_NL (4.5 bpw); `blk.0.attn_qkv` is IQ4_XS (4.25 bpw); small
projections (`indexer.*`, `hc_*`, `ple_key`, `ffn_gate_inp`) are BF16; norms/biases are F32.
Dispatch must be per tensor, never per layer ("the pack decides per tensor",
`ST/src/core/layer.cpp:1071-1073`, `ST/include/strata/core/layer.hpp:64-65`).

---

## 7. Tensor shapes table

Convention: activation shapes exclude the batch axis; `T` = tokens processed in this call.

### 7.1 Decode, batch = 1, T = 1 (one token)

| point | GDN layer | QSA layer |
|---|---|---|
| embedding / `mixed` (HC_READ out) | `[2560]` | `[2560]` |
| residual stack `R` | `[4, 2560]` = 10240 | `[4, 2560]` |
| mixer projection out | `attn_qkv` -> `[10240]` | `attn_q` -> `[12288]` |
| q / k / v inside mixer | q `[128,16]`, k `[128,16]`, v `[128,48]` | Q `[256,24]`, K `[256,2]`, V `[256,2]` |
| gate | `attn_gate` -> `[6144]` | 2nd half of `attn_q` -> `[24,256]` |
| conv window | `[10240, 4]` (3 history + 1 new) | - |
| recurrent state (per seq) | `[128,128,48]` f32 (3 MiB) | - |
| conv history (per seq) | `[3,10240]` f32 (120 KiB) | - |
| KV cells | none | K `[256,2]`, V `[256,2]` per cell |
| indexer | - | q `[128,4]`, raw k `[128]` |
| selection width | - | `min(n_kv, 2051)` cells |
| mixer out / block_out | `[2560]` | `[2560]` |
| router logits / ids / w | `[512]`, `[10]`, `[10]` | same |
| expert intermediate | `[640]` x10 + shared | same |
| head logits | `[248320]` | same |

### 7.2 Prefill example, T = 512

| point | GDN layer | QSA layer |
|---|---|---|
| `mixed` / residual | `[2560, 512]`, `[4, 2560, 512]` | same |
| projections | qkv `[10240, 512]`, z `[6144, 512]` | `attn_q` `[12288, 512]`, `attn_k/v` `[512, 512]` |
| heads | q `[128,16,512]`, k `[128,16,512]`, v `[128,48,512]` | Q `[256,24,512]`, K `[256,2,512]`, V `[256,2,512]` |
| conv window | `[10240, 3+512]` | - |
| GDN kernel | chunked, chunk size 64 (`delta-net-base.cpp:61-64`) or fused | flash/prompt attention over `n_kv` cells with causal+selection mask |
| states | updated once for the whole chunk | K/V/indexer appended for all 512 cells |
| indexer raw k | - | `[128, 512]` |
| router | `[512, 512]` logits, `[10, 512]` ids | same |
| head | `[248320, 512]` | same |

Both engines keep a **per-token** order for GDN prefill: strata's T-token kernels are asserted
bit-equal to the single-token kernel applied per token (`ST/include/strata/core/verify.hpp:3-11`).

---

## 8. MoE routing: math, tensors, and `-ot`

* Router tensor: `blk.<i>.ffn_gate_inp.weight`, ne `[2560, 512]`, BF16.
  Activation is quantised to BF16 **before** this GEMV: strata measured logits 8.100e-3 off
  without it, enough to flip a top-10 pick (`ST/include/strata/core/layer.hpp:134-141`).
* `p = softmax(logits)` over all 512, `ids = top10(p)`, `w = p[ids]`,
  `w = w / max(sum(w), 2^-14)` (`llama-graph.cpp:2171,2237,2251,2269`).
  Computing the softmax over the selected subset instead is *not* equivalent once the clamp is
  applied (`ST/src/kernels/cuda/router_top10.cu:15-17`).
* Expert tensors: `ffn_gate_exps` `[2560,640,512]`, `ffn_up_exps` `[2560,640,512]`,
  `ffn_down_exps` `[640,2560,512]` - `ne[2]` is the expert index, `ne[1]` the output width,
  `ne[0]` the input width (`qwen4exp.cpp:275-277`). A fused `ffn_gate_up_exps`
  `[2560, 1280, 512]` is optional and **absent** from this file (`llama-model.cpp:3818-3821`).
* Shared expert: `ffn_gate_{shexp,up,down}` + scalar gate `ffn_gate_inp_shexp` (`[2560]`, BF16,
  sigmoid, applied after the shared FFN) - `qwen4exp.cpp:1163-1184`, tensor check
  `ST/src/core/layer.cpp:427`.
* **llama.cpp `-ot` mapping.** `-ot` is `<regex>=<buffer type>`, matched with
  `std::regex_search` against the full tensor name
  (`LC/common/arg.cpp:252-283,2757-2761`, `LC/src/llama-model-loader.cpp:1242-1246`).
  Therefore:
  * `blk.N.ffn_gate_exps` -> `blk.N.ffn_gate_exps.weight`
  * `blk.N.ffn_up_exps`   -> `blk.N.ffn_up_exps.weight`
  * `blk.N.ffn_down_exps` -> `blk.N.ffn_down_exps.weight`
  * `-cmoe` / `-ncmoe` use the shared regex `\.ffn_(up|down|gate|gate_up)_(ch|)exps`
    (`LC/common/common.h:1239`); `-otl`-style helpers build `blk\.N` + that regex
    (`common/common.h:1243-1245`).
  These three tensors are by far the largest block - **39.97 GiB = 91.2% of shard 1 and ~57% of
  the 70.6 GB file** (measured by summing `n_bytes` over all 1223 shard-1 tensors) - and they are
  the natural unit of offload: strata keeps them on the CPU/pinned pool
  (`ST/include/strata/core/layer.hpp:126-166`).

---

## 9. PLE / n-gram subsystem

**What it is**: a single-layer "per-layer embedding" module on layer 1 that adds a *hash of the
last 3 token ids* to the residual stack, plus a dilated depthwise conv over its own history.

### 9.1 Host hash (no tensor op)

`LC/src/models/qwen4exp.cpp:1192-1291`; the same rules re-asserted and tested at
`ST/include/strata/kernels/ngram.hpp:9-27`.

For token at position `t`, with `n_gram = 3`, `heads_per_ngram = 8`, `n_heads = 16`:

```
ctx[0] = tok(t)
for s = 1..2:  ctx[s] = tok(t-s), but any EOS/missing predecessor
                cuts the window FORWARD (that slot and every older one = EOS)
for n = 2..3:                                    // n-gram order
    mixed = (ctx[0]*m[0]) XOR (ctx[1]*m[1]) XOR ... (ctx[n-1]*m[n-1])   // uint64
    for g = 0..7:                                // 8 heads for this order
        h = (n-2)*8 + g
        row[h] = mixed % head_vocab_sizes[h] + head_offsets[h]
```

* XOR, not add; `%`, not a power-of-two mask (vocab sizes are `20000003, 20000023, ...`)
* the token's own EOS does not cut its own context
* `tok(-1)` reads as EOS; token id 0 is a real token
* image/embd batches stand in with `ple.image_token_id` (or EOS) - `qwen4exp.cpp:1223-1232`
* predecessors come from the attention KV cells' stored token ids
  (`qwen4exp.cpp:1251`, `mctx->get_prev_tokens`)

Rows are gathered from `per_layer_token_embd` (`[160, 320001536]`, IQ4_NL) and flattened
head-slowest into `[2560, T]` (`qwen4exp.cpp:1361-1364`).

### 9.2 Device block

`LC/src/models/qwen4exp.cpp:1369-1457`; transcribed at `ST/include/strata/kernels/ple.hpp:5-13`:

```
key       = grouped_rmsnorm(ple_key @ emb)         [10240], per stream
query     = grouped_rmsnorm(R)                     [10240], per stream
value     = ple_value @ emb                        [2560]
s[c]      = sum_d key[c,d]*query[c,d] / sqrt(2560)
gate[c]   = sigmoid( sign(s[c]) * sqrt(max(|s[c]|, 1e-6)) )     // signed sqrt
gated     = value (broadcast over hc) * gate                    [4, 2560]
normalized= grouped_rmsnorm(gated)                              <-- THIS is the conv input
conv      = sum_{k=0..3} w[k][c] * padded[hist + t - (3-k)*3][c] // dilation = ngram_size = 3
out       = R + gated + silu(conv)                               (ple.hpp:13)
```

* history length `hist = (4-1)*3 = 9` rows of `[10240]` per sequence
  (`qwen4exp.cpp:1413-1424`, `ngram.hpp:35`)
* conv weight layout is GGML-native `k + 4*c`, i.e. the transpose of a row-major
  `[4, 10240]` (`ple.hpp:27-34`, view at `qwen4exp.cpp:1438-1441`)
* the conv is causal and the history is shifted once per token by the *caller*
  (`ST/src/core/layer.cpp:1375-1383`), so chunked prefill == single-shot prefill
* the hash + gather **must** be re-run per token; if baked into a captured graph they replay the
  capture-time token forever (`ST/src/core/layer.cpp:1340-1348`)

---

## 10. Attention indexer (QSA)

### 10.1 Purpose

Select at most ~2048 of the cached cells for each query, so attention cost is bounded by the
selection width rather than by context length. `indexer_budget = 2048`,
`compress_ratio = 4` (blocks of 4 cells keyed by one pooled key).

### 10.2 Key-side state (per QSA layer, per sequence)

`ST/include/strata/kernels/qsa.hpp:197-237`, `LC/src/models/qwen4exp.cpp:756-792`:

```
raw[j]     = index_k_proj @ x                     [128], NOT normed, NOT rotated
pooled[b]  = rope( rmsnorm( mean(raw[4b..4b+3]) ), pos = 4b )      b = 0..floor(n_kv/4)
spare      = rope( rmsnorm(raw[0]), pos = 0 )      constant, appended at row n_bid
```

* pooling happens when cell `4b+3` arrives (`qsa.cu:169-220`)
* rotation uses the **block's first** cell's position (`qsa.cu:224-233`, `qsa.hpp:226-228`)
* the spare row makes the incomplete tail scoreable (`qsa.hpp:211-214`)

### 10.3 Query-side scoring

```
q     = rmsnorm(indexer.q_proj @ x, indexer.q_norm) then IMRoPE   (qwen4exp.cpp:794-800)
score(b) = sum_{h=0..3} relu( sum_d pooled[b,d] * q[h,d] )         (qsa.cu:276-277)
         + 1e9  if b is the incomplete tail block                  (qsa.cu:286-288)
```

llama.cpp's equivalent uses a per-head weight of `1/sqrt(idx_dim)` and a `-inf`/`0` mask instead
of the `+1e9` bias (`qwen4exp.cpp:802-811`, `ggml_lightning_indexer` in
`LC/ggml/src/ggml.c:6444-6476`). See open question Q4.

### 10.4 Selection and attention

```
width   = min(n_kv, idx_top_k + idx_block - 1) = min(n_kv, 2051)   (qsa.hpp:127-136)
ids     = top_width(scores)   (ascending order, numpy tie rules)   (qsa.cu:293-400)
K,V     = gather(ids)  ->  dense causal attention over `width` cells
```

Below 2051 cells the selection is the identity, i.e. the layer degenerates to plain dense
attention (`qsa.hpp:130-132`) - which is why short-context output can be validated against a
trivial dense implementation. Unselected cells must contribute exactly zero weight: the
reference fills the KQ mask with `-inf` and leaves selected cells at 0
(`qsa.hpp:191-193`, `LC/src/models/qwen4exp.cpp:825-864`).

### 10.5 Caches involved

1. **Attention KV**: `K/V [head_dim, n_head_kv] = [256,2]` per cell, appended after
   norm+RoPE of K (V is never rotated) (`qsa.hpp:176-195`).
2. **Indexer key cache**: raw + pooled keys, one row per cell, tracking the KV cache cell for
   cell (`qwen4exp.cpp:414-418`; `LC/src/llama-memory-hybrid-idx.h:14-16`).

---

## 11. MTP (multi-token prediction) head

### 11.1 Where it lives

The main GGUF has **no** MTP block. The head ships as its own artifact:

* `/srv/strata-data/mtp/mtp-q2_0.gguf`, `general.architecture = qwen4exp-mtp`, 31 tensors
* raw BF16 tensors in `/srv/strata-data/mtp/tensors/*.bin`, runtime pack in
  `/srv/strata-data/mtp/rt/` (`dense.bin`, `experts.bin`, `draft_vocab.bin`)
* inventory: `/srv/strata-data/mtp/mtp-inventory.md` (31 tensors, 5.214 GB BF16 pre-quant)

llama.cpp would read it from `blk.48.*` behind `qwen4exp.nextn_predict_layers`
(`LC/src/llama-arch.cpp:230`, `llama-model.cpp:1380`, `qwen4exp.cpp:290-301`) - that key is
absent, so **this fork cannot run the MTP from this GGUF**; strata loads the separate artifact
(`ST/include/strata/core/mtp.hpp:43-46`).

### 11.2 Structure

`ST/include/strata/core/mtp.hpp:1-19` and `qwen4exp.cpp:524-616`:

```
e = fc_embedding( rmsnorm(embed(tok)) * (1+w) )            [2560 -> 2560]
h = fc_hidden(    rmsnorm_10240(R) * (1+w) )               [10240 -> 2560] per stream
R0 = h + e                                                 -> [4, 2560]
   (llama.cpp writes this as concat(e,h) then eh_proj [5120,2560]; algebraically identical,
    qwen4exp.cpp:581-589)
-> HC_READ(attn) -> QSA/full attention with ITS OWN K/V -> HC_WRITE
-> HC_READ(ffn)  -> MoE (512, top-10, shared expert)      -> HC_WRITE
-> final mixer (hyper_connection_mixer / nextn.hc_head_*)  -> [2560]
-> head (shared `output.weight`) -> logits
```

Tensor shapes (BF16 unless noted):

| tensor | ne | role |
|---|---|---|
| `mtp.pre_fc_norm_embedding.weight` | `[2560]` | enorm gamma (folded `1+w`) |
| `mtp.pre_fc_norm_hidden.weight` | `[10240]` | hnorm gamma (folded `1+w`) |
| `mtp.fc_embedding.weight` | `[2560, 2560]` | |
| `mtp.fc_hidden.weight` | `[2560, 2560]` | |
| `mtp.layers.0.self_attn.q_proj.weight` | `[2560, 12288]` | q\|gate |
| `...k_proj` / `...v_proj` | `[2560, 512]` | 2 x 256 |
| `...o_proj.weight` | `[6144, 2560]` | |
| `...q_norm` / `...k_norm` | `[256]` | |
| `...indexer.index_qk_proj.weight` | `[2560, 640]` | **fused** q(512)+k(128); split at load (`third_party/llama.cpp/conversion/qwen4exp.py:98-107`) |
| `...mlp.experts.gate_up_proj` | `[2560, 1280, 512]` | fused gate\|up, q2_0 |
| `...mlp.experts.down_proj` | `[640, 2560, 512]` | q2_0 |
| `...mlp.gate.weight` | `[2560, 512]` | router |
| `...mlp.shared_expert.*` | `[2560,640]` / `[640,2560]` | |
| `...mlp.shared_expert_gate.weight` | `[2560, 1]` | |
| `mtp.layers.0.{attn,mlp}_hyper_connection.*` | `hc_norm [10240]`, `down [10240,320]`, `up [320,10240]`, `block_inject [10240,4]` | |
| `mtp.hyper_connection_mixer.*` | same three | final mixer for the draft logits |

Note the differences from the trunk: no GDN layers (the MTP block is a single full-attention
block), a **fused** `index_qk_proj`, and a **fused** `gate_up_proj`.

### 11.3 Drafting and verification

Pairing convention (`mtp.hpp:5-7`): cell `i` of the MTP sequence pairs the trunk's final
residual `R_i` with token `t_{i+1}`, is run at rope position `i`, and predicts `t_{i+2}`; its own
residual feeds the next draft step (`qwen4exp.cpp:600-605`).

strata's flags (`ST/src/program/generate.cpp`):

| flag | meaning | line |
|---|---|---|
| `--spec N` | verify window: tokens proposed per check (setup writes 4) | 798-799, 1933 |
| `--spec-min-p P` | confidence needed to extend the window | 800-801, 1955 |
| `--spec-oracle`, `--spec-follow`, `--spec-corrupt N`, `--spec-split` | exactness/A-B instrumentation | 542-546, 1934-1957 |
| `--mtp-max-t M` | cap the MTP's window (0 = `--spec`) | 653, 798 |
| `--suffix-draft N` | prompt-lookup drafts alongside MTP (default 3, 0 = MTP only) | 651-652 |
| `--lookup-chain K`, `--lookup-chain-min M` | chained prompt-lookup drafts | 803-805 |
| `--mtp-hnorm pooled\|stream` | one RMS over all four HC streams vs per-stream | 806, 656-659 |
| `--batch-mtp` | one MTP proposal per batch slot | 780-782 |

Verification (`ST/include/strata/core/verify.hpp:1-18`):

1. `T` tokens (last accepted + `T-1` drafts) run through **all 48 layers in one captured graph**;
   every row's argmax equals plain greedy decode bit for bit, so a draft is accepted exactly when
   greedy decode would have produced it.
2. The window **appends K/V and indexer keys for all T positions but leaves the GDN state
   untouched**; `commit(n_keep)` then (a) replays the accepted prefix through the GDN conv
   history and recurrent state, (b) restores the indexer key tail from a snapshot and re-appends
   accepted keys, (c) sets the PLE history to its snapshot after token `n_keep-1`, and (d) lets
   the rejected K/V cells be overwritten when those positions are processed again.
3. Measured draft acceptance for this head: 0.89 / 0.86 / 0.85 of greedy drafts at steps 1-3
   (`mtp.hpp:4`).

llama.cpp's equivalent machinery for the trunk is the rollback-slot design: `K = n_rs_seq + 1`
GDN state snapshots plus `n_rs_seq + 1` conv-history slots, so a rollback of `s` tokens reads a
history that never saw them (`LC/src/models/delta-net-base.cpp:1323-1344, 494-519`,
`qwen4exp.cpp:1330-1344`).

---

## 12. KV / state layout: how the hybrid model differs from plain KV

Three *different* per-layer states exist. Only 12 of 48 layers have anything resembling a KV cache.

### 12.1 Attention layers (12) - real KV cache

* per cell: `K [2, 256]` + `V [2, 256]`. In fp16 that is `1024 B + 1024 B = 2048 B/cell/layer`.
* strata layout: `k_pool/v_pool [page][kv_head][page_size][head_dim]` fp16 with
  `page_size = 4` cells (`ST/include/strata/kernels/qsa.hpp:176-184, 113-116`), optionally
  quantised (int8 / q4 / "K8V4" hybrid, `ST/src/core/layer.cpp:1081-1101`).
* llama.cpp: `llama_memory_hybrid`'s attn cache, type from `--cache-type-k/v` (default F16).
* cost at full 262144 context: `2048 B * 262144 * 12 layers = 6.0 GiB`.
  This - not the weights - is the dominant cache cost; GQA-2 with head_dim 256 is unusually wide.
* Position bookkeeping: the indexer cache must stay cell-for-cell with the KV cache
  (`qwen4exp.cpp:414-418`).

### 12.2 Attention layers (12) - indexer side-cache

* llama.cpp: one row per cell of `2 * idx_dim = 256` f32 (`raw | pooled`), i.e. 1024 B/cell/layer
  → 3.0 GiB at full ctx (`qwen4exp.cpp:756-792`, `llama-memory-hybrid-idx.h:14-16`).
* strata: `tail [3,128]` f32 + `dead [128]` f32 + `pooled [max_cells/4 + 1, 128]` f32, i.e.
  ~128 B/cell amortised plus a 1.5 KiB tail (`qsa.hpp:199-214`).

### 12.3 GDN layers (36) - **no KV at all**: two small states

| state | shape (per sequence) | bytes (f32) |
|---|---|---|
| recurrent state `S` | `[128, 128, 48]` | 3,145,728 (3 MiB) |
| conv history | `[3, 10240]` | 122,880 (120 KiB) |

`36 * (3 MiB + 120 KiB) = ~112 MiB` per sequence, **independent of context length**. This is
the "GDN state as pseudo-KV": the only thing that must be carried across steps, updated in place
(`LC/src/models/delta-net-base.cpp:552-556`; strata's `gdn_layer`, `ST/src/core/layer.cpp:225-335`,
zeroed at sequence start by `ST/src/core/layer.cpp:220`). strata's test suite names the combined
structure `kv_hybrid_parity` (`ST/src/kernels/kv_hybrid_parity.cpp`).

### 12.4 The one remaining per-sequence state

* PLE conv history: `9 x 10240` f32 = 360 KiB, layer 1 only (section 9.2).
* MTP head: its own K/V for its window (`mtp.hpp:43-46` binds `window = 32768` cells →
  64 MiB at fp16) and it has no GDN layers, so it needs no recurrent state
  (`qwen4exp.cpp:568-569` allocates the recurrent input but the draft memory has no recurrent
  layer).

### 12.5 Rollback / speculative requirements

Anything that speculates needs, per sequence:

1. `K = n_rs_seq + 1` GDN state snapshots and the same number of conv-history slots
2. an indexer key-tail snapshot (3 rows) plus the ability to re-append accepted keys
   (`verify.hpp:15-17`)
3. the PLE history snapshot after token `n_keep-1`
4. K/V cells past the accepted prefix simply overwritten later

---

## 13. Ops inventory - every primitive a new engine must implement

| # | primitive | shapes here | citations |
|---|---|---|---|
| 1 | quantised embedding gather (IQ3_S / S-form row) | `[2560,248320] -> [2560]` | `ST/src/core/layer.cpp:1194-1247` |
| 2 | quantised GEMV / GEMM (dequant weight x (bf16\|f32\|q8) activation) | all projections | `ST/src/core/layer.cpp:96-174` |
| 3 | grouped RMSNorm over `[n_embd, hc]` | gamma `[2560,4]`, input `[2560,4,T]` | `qwen4exp.cpp:327`, `ple.hpp:5-8`, `ST/src/kernels/gr.hpp:77-93` |
| 4 | per-head RMSNorm | `[256]` over `head_dim=256`; `[128]` over `idx_dim` | `qwen4exp.cpp:953,963,796` |
| 5 | plain RMSNorm | `ssm_norm [128]`, PLE norms `[10240]` | `qwen4exp.cpp:1129,1381-1387` |
| 6 | L2 norm (`rms_norm(eps/n) * 1/sqrt(n)`) | q/k of GDN, `[128]` | `LC/src/models/models.h:14-18` |
| 7 | HC low-rank mixer (norm -> down -> silu(/hc) -> up -> sigmoid -> mean + inject) | 10240/320/4 | `qwen4exp.cpp:313-397`, `gr.hpp:77-124` |
| 8 | HC write (2*sigmoid per-stream scale + add) | `[4,2560]` | `qwen4exp.cpp:369-397`, `gr.hpp:121-124` |
| 9 | SiLU, sigmoid, softplus, exp elementwise | several | `qwen4exp.cpp:332,641,1042,1050` |
| 10 | depthwise causal conv1d (GDN) | `[4,10240]` over `[3,10240]` history | `qwen4exp.cpp:1074`, `gdn.hpp:60-70` |
| 11 | **GDN step** (decay, rank-1 delta rule, output) | `S=128, h_k=16, h_v=48` | `delta-net-base.cpp:286-368`, `gdn.hpp:19-57` |
| 12 | **GDN chunked prefill** (decay mask, triangular solve, state carry) | chunk 64 | `delta-net-base.cpp:16-284` |
| 13 | IMRoPE / interleaved M-RoPE (Neox pairs, `n_rot=64`, sections `[11,11,10,0]`) | `[256,24,T]`, `[128,4,T]` | `qwen4exp.cpp:976-986,781-799`, `LC/ggml/src/ggml-cpu/ops.cpp:6219-6276` |
| 14 | GQA attention with additive mask (dense + sparse-selected) | 24 heads / 2 kv, d=256 | `qwen4exp.cpp:870-1013` |
| 15 | top-k selection over `width = min(n_kv,2051)` scores with numpy tie rules | `qsa.cu:293-400` | `ST/src/kernels/cuda/qsa.cu:293-400` |
| 16 | indexer score: per-head ReLU dot, sum, tail bias | 4 heads x 128 | `qsa.cu:247-289`, `qwen4exp.cpp:794-811` |
| 17 | indexer key pool: mean of 4 raw keys -> RMS -> RoPE at block start | `[128]` | `qsa.cu:158-245` |
| 18 | KV append / gather (fp16, or quantised) | `[256,2]` per cell | `qsa.hpp:176-195` |
| 19 | MoE router: BF16 GEMV -> softmax(512) -> top-10 -> clamp-renormalise | `[2560,512]` | `llama-graph.cpp:2171-2273`, `router_top10.cu:69-140` |
| 20 | SwiGLU expert FFN (gate/up/down, batched over 10 experts) | `[*,640]`, `[640,*]` | `qwen4exp.cpp:1142-1190` |
| 21 | shared expert + sigmoid scalar gate | `ffn_gate_inp_shexp [2560]` | `qwen4exp.cpp:1163-1184` |
| 22 | PLE host hash (u64 mul/xor/mod) | 16 rows/token | `qwen4exp.cpp:1245-1279`, `ngram.hpp:9-27` |
| 23 | PLE gather (`[160, 320001536]` row lookup) + head-slowest flatten | `[2560,T]` | `qwen4exp.cpp:1361-1364` |
| 24 | PLE gate (signed-sqrt + sigmoid) and dilated depthwise conv (dilation 3, hist 9) | `[10240]` | `qwen4exp.cpp:1390-1456` |
| 25 | linear head GEMV | `[2560,248320]` | `ST/src/core/layer.cpp:1270-1286` |
| 26 | (speculative) snapshot/rollback of state 12.5 | - | `verify.hpp:13-18`, `delta-net-base.cpp:1323-1344` |

RoPE detail worth pinning early: for text, M-RoPE degenerates to plain RoPE over the first 64
dims of each head - all three section thetas are initialised from the same base and advanced by
the same `theta_scale` each pair (`ops.cpp:6224-6276`), so `theta_h == theta_w == theta_t`.
`theta_scale = freq_base^(-2/n_rot) = 1e7^(-2/64)`. Only dims `0..63` of the 256-dim head are
rotated (`n_offs + n_dims <= ne0`, `ops.cpp:6360`).

---

## 14. Numerical / dtype contract

* RMSNorm epsilon is `1e-6` everywhere (`attention.layer_norm_rms_epsilon`).
* HC, PLE and MTP gammas: treat the **stored** value as the multiplier. The converter folds
  `(1 + w)` for PLE and indexer norms explicitly
  (`ST/third_party/llama.cpp/conversion/qwen4exp.py:119-121`) and both engines assert the HC
  gammas arrive folded (`gr.hpp:83`, `qwen4exp.cpp:326`).
* `ssm_alpha`/`ssm_beta` are BF16; `attn_qkv`, `ssm_out`, `attn_q/k/v`, `attn_output` are
  k/i-quantised and want Q8_K or Q8_0 activations. strata measured 0.66-1.41% error from using
  the wrong activation format (`ST/include/strata/core/layer.hpp:29-32,63-65`).
* The router **must** see a BF16 activation (8.100e-03 logit error otherwise, enough to flip
  top-10) - `layer.hpp:134-141`.
* HC weights are BF16 (288 of shard 1's 483 BF16 tensors are `hc_*`) - `gr.hpp:9-12`.
* `ssm_a` is negative (`-exp(A_log)`), `ssm_dt` in `[-8, +2.53]` - verified from the artifact.
* Attention gate and MoE shared gate are **sigmoid**, not SiLU; the GDN output gate is sigmoid;
  the GDN conv activation is SiLU applied to the whole 10240 before the q/k/v split.

---

## 15. Open questions / ambiguities

* **Q1 - HC gamma folding.** `gr.hpp:83` and `qwen4exp.cpp:326` say the converter stores
  `(1 + w)`, but `conversion/qwen4exp.py` only folds `+1` for PLE/indexer norms
  (`:119-121`) and no `hc_norm` folding rule was found in `conversion/base.py`. Determine where
  the fold happens for `hc_*`/`output_hc_norm` (pack-time or HF-side) before implementing;
  a wrong assumption is a plausible-looking scale error on every layer.
* **Q2 - GDN k-head broadcast.** Upstream marks it unresolved: `ggml.h:2639` `TODO: add
  ggml_gated_delta_net_set_bcast ... tiled vs interleaved`. strata pins **modulo**
  (`gdn.hpp:54-57`, reference property 8/11) and llama.cpp's non-fused path reproduces it by
  tiling 16 heads to 48 (`qwen4exp.cpp:1114-1118`). Use modulo and lock it with a parity test;
  confirm the fused ggml/HIP kernel agrees.
* **Q3 - state layout.** `(S, S, H_v)` (llama.cpp) vs `(S, H_v, S)` (strata) - same numbers,
  different strides (`gdn.hpp:13-18`). Pick one; document it in the kernel contract.
* **Q4 - indexer score scale.** llama.cpp applies `1/sqrt(idx_dim)` per head
  (`qwen4exp.cpp:804`); strata's `qsa_index_kernel` sums raw dots (`qsa.cu:276-277`). Selection
  ranking is scale-invariant (both keep the tail via a large bias/`-inf`), but any use of the
  raw score as a probability would differ. Verify equivalence empirically.
* **Q5 - "full attention" is not dense.** Above 2051 cells these layers are sparse; only the
  layer *type* is "full attention" (`full_attention_interval = 4`). Do not build a dense
  262144-cell attention path and call it correct.
* **Q6 - MTP packaging.** The MTP GGUF is a separate architecture (`qwen4exp-mtp`) with fused
  `index_qk_proj` and fused `gate_up_proj`, and strata's runtime pack (`rt/`) quantises the big
  projections to Q8_0 and the experts to q2_0 (`mtp.hpp:15-19`, `mtp-q2_0.gguf.report.json`
  records 0.42 relative RMS error per expert). Decide whether to ingest `mtp-q2_0.gguf` or the
  BF16 tensors, and whether q2_0 draft quality is acceptable.
* **Q7 - `--mtp-hnorm pooled|stream`.** `pre_fc_norm_hidden [10240]` reads naturally as one
  RMS over all four streams; llama.cpp's graph normalises per stream
  (`qwen4exp.cpp:581`, `mtp.hpp:57-59`); strata defaults to pooled and offers `--mtp-hnorm
  stream`. Draft-only, but pick deliberately and record which was validated (0.89/0.86/0.85).
* **Q8 - KV precision.** fp16 is the baseline; strata's int8/q4/K8V4 caches are engine
  features, not model requirements. Any of them changes output; measure before adopting
  (`ST/src/core/layer.cpp:1081-1101`).
* **Q9 - tie between `output.weight` and `token_embd.weight`.** They are separate tensors with
  different quants (Q5_K vs IQ3_S); tying them would be wrong.
* **Q10 - batch > 1.** Both references were read at batch 1 (decode) and one-sequence prefill;
  `qsa.hpp`'s kernels are documented as "fixed single-sequence geometry" for the indexer
  (`native_qsa_indexer.hpp:10-14`). Multi-sequence batching needs its own design pass.
* **Q11 - RoPE section semantics under YaRN.** The artifact ships no `rope.scaling` keys
  (`qsa.hpp:121-125`); if a future build adds scaling, the `theta_e` branch and the pooled-key
  rotation at `pos 0` (`qsa.cu:200-215`) must be re-checked.
* **Q12 - the "56B" in `general.size_label = 512x56B` is not the parameter count.** Counting from
  the file's own shapes: routed experts alone are `48 * 512 * 3 * 2560 * 640 = 120.8B` params
  (confirmed by their 39.97 GiB of storage at a measured ~2.84 bpw average), the PLE table is
  `160 * 320001536 = 51.2B` rows (`ST/include/strata/kernels/ngram.hpp:9,46`), and the dense
  parts are ~4.5B - roughly 176B stored rows in total. The per-token compute path touches ~6B of
  them (2.36B routed experts + 0.24B shared + ~3.3B dense), plus 2.54B embedding/lm_head rows and
  16x160 PLE rows. Size memory from these numbers, not from the label; clarify with the artifact's
  provenance what `56B` denotes before quoting it anywhere.
