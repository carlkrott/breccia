# Inkling-2-Mini in Breccia: graph notes, coupling map, kernel gaps, abstraction plan

## Status (2026-11-23, supersedes the sections below where they differ)

- **S0 done.** Golden: llama.cpp CPU-only build (`build-cpu`, `-fa off --temp 0 -t 16`) of `~/llama.cpp-inkling` commit e3ada8dbc (golden-dump hook in common/debug.cpp). Prompt "The capital of France is Paris, and the capital of Germany is" (13 ids, no BOS), top-1 154328 " Berlin", identical on two runs. Dumps and README in `~/projects/inkling-dual/golden/` (outside the repo, 81 MB).
- **Requant drops S2 and S3.** `llama-quantize --allow-requantize --pure --tensor-type-file inkling-tt.txt` turns the 128 dense Q8_0 tensors into Q6_K and the 24 ffn_down_exps (Q8_0/Q5_0) into IQ4_NL; everything else is a byte copy. Every tensor is now F32, Q4_K, Q6_K or IQ4_NL, all of which have Breccia kernels. Perplexity (llama.cpp docs, c512, 16 chunks, CPU): 3.2602 -> 3.2720 (+0.36%). File: `/srv/strata-data/models/inkling2-unbound-v2-breccia/Inkling-2-Mini-Unbound-v2-breccia-Q4_K_M.gguf` (14373 MiB). The binder rejects the stock file with a requant hint. GEMV conformance at every Inkling cell passed on both cards (2026-11-24, `BRECCIA_SHAPES` in `tests/conformance.rs`; worst max rel 1.7e-4 on the 201024x2304 Q6_K head, gate 1e-3): Q6_K 3072/1024/32x2304, 2304x3072; Q4_K 384/24/2304x2304; IQ4_NL 2304x1152; F32 128x2304, 64x4608, 2304x64, 4x2304; batched-expert and split-K checks for each type. F32 has no multi-user GEMM shader, so prefill for the F32 weights (router, loop control, mix queries) uses batched GEMV. Multi-user GEMM at the real shapes is checked against the golden at the prefill stage (`gemm_many.rs` needs the Qwen layout).
- **Correction:** the model input has an RMS norm `token_embd_norm.weight` (F32, 2304) after the embedding lookup (inkling2.cpp:96, :268). Sections 1.2 and 1.10 miss it; S11 must apply it.
- `output.weight` and `token_embd.weight` have 201024 rows; logits are sliced to `vocab_size_unpadded` = 200058.
- **S1 done** (`crates/breccia-engine/src/inkling.rs`): `InklingConfig::from_metadata` (per-layer head arrays must be uniform), `LoopPlan` (48 slots; execution order prelude 0-5, pass 0 slots 6-17, pass 1 slots 24-35, pass 2 slots 36-47, post 18-23; full-attention slots 5/11/17/23/29/35/41/47), and `bind` (exact shape and type of all 704 tensors). `Model::load` binds an inkling2_loop file and then refuses it until the graph exists. Real-file test: `BRECCIA_INKLING_GGUF=<path> cargo test -p breccia-engine --lib inkling`.
- **S4 done** (3a731e5): `ActMode::TanhCap` (`s*tanh(x/s)`: loop-control tanh at s=1, softcap at s=30) and `RowClip` (`ops_row_clip.comp`: channel scale, L2 clip, tau). On both cards, max rel vs the f64 CPU reference is <= 3e-7 and vs the golden Qcur/Kcur/Rcur for all 48 slots is 3.16e-7 (inputs from the golden's pre-clip nodes, `dump-s4/`). Tau is exactly 1 below pos 8192, so only the CPU reference covers it (positions to 1M). A >8192-token golden must also check tau end to end, and a >1024-token prompt must check SWA. Known limit, to fix by S11: `RmsNorm` dispatches `(rows,1,1)` with no grid_for wrap, so it fails above 65535 rows (24 heads x 4096-token chunk).
- Next: S5 onward. S2/S3 became the conformance runs above (done). The golden now names the norm/conv/AttnRes/loop nodes (llama.cpp b8f1aefa0, see golden/README.md).

Date: 2026-11-23. Read-only analysis. No edits, builds, GPU runs or commits in the Breccia repo. The GGUF was only read.

Sources:
- llama.cpp reference: /home/korphaus/llama.cpp-inkling at commit dbcb52e8e (clean tree when checked). Short names: `inkling2.cpp` = src/models/inkling2.cpp, `loop.cpp` = src/models/inkling2-loop.cpp, `llama-graph.cpp` = src/llama-graph.cpp.
- GGUF: /srv/strata-data/models/inkling2-unbound-v2/Inkling-2-Mini-Unbound-v2-Q4_K_M.gguf. Still being written. Header parsed with a local script (hdr.py, dump hdr.txt in the scratchpad ggufdump folder). Tensor data offsets and checksums NOT validated.
- Breccia: /var/tmp/breccia-coord/wt, branch claude/tools.

Marks:
- UNCONFIRMED: not checked against code or file in this pass.
- (earlier read): cite came from an earlier pass and was not re-read in the 2026-11-23 verification pass. This covers llama-graph.cpp cites outside 2286-2490, 2531-2545 and 2602-2616, llama-kv-cache.cpp, llama-hparams.h, and all Breccia code cites. Re-read at implementation time.
- Verified 2026-11-23: inkling2.cpp and inkling2-loop.cpp read in full; hdr.txt metadata lines and Breccia AGENTS.md rules grepped and matched; llama-graph.cpp MoE lines grepped; llama-graph.cpp:2525-2620 (fused split, SITU) read in full.

---

## 1. Inkling graph (llama.cpp reference)

### 1.1 Model facts (GGUF header, hdr.txt)

| Item | Value | Source |
|---|---|---|
| Architecture | inkling2_loop, 24 physical layers | hdr.txt:2, :11 |
| n_embd / expert ff | 2304 / 1152 | :13, :14, :24 |
| Heads | 24 q, 8 kv, head_dim 128 (k and v, SWA and full) | :15-16, :20-21, :25-26 |
| Experts | 128, top-8, expert_weights_scale 8.0 | :18-19, :31 |
| Vocab | 201024 padded, 200058 real | :22-23 |
| Context metadata | 1048576 | :12 |
| SWA | window 1024. Full-attention layers 5, 11, 17, 23 | :28-29 |
| AttnRes block size | 6 | :34 |
| Shortconv | l_cache 4 | :30 |
| Relative bias | 16 per head, extent 1024 | :36-37 |
| Clips | qk 22.627, relative 8.0 | :38-39 |
| Log scale (tau) | floor 8192, alpha 0.1 | :40-41 |
| SITU | beta 4.0, linear beta 25.0 | :32-33 |
| Loop | start 6, end 18, count 3, gate rank 64, varying 0 | :42-46 |
| Output softcap | 30.0 | :35 |
| RoPE | dimension_count 0 (no RoPE) | :27 |
| Tokenizer | model gpt2 (hdr.txt:47), pre = inkling2 (hdr.txt:48) | verified |

### 1.2 Layer order, passes, slots, blocks

- Prelude: physical layers 0-5, slot = layer index (loop.cpp:83-85).
- Core: physical layers [6,18) run 3 passes (loop.cpp:90-109).
  - Pass 0: slot = ip (6..17).
  - Pass p >= 1: slot = 24 + (p-1)*12 + (ip-6) (loop.cpp:101).
  - Weights are shared: layer_phys(il) maps slot to physical layer (loop.cpp:3-5, :65-70). Each slot keeps its own KV and conv state.
- Post: physical layers 18-23, slots 18-23 (loop.cpp:111-113).
- Total slots: 24 + 2*12 = 48 (loop.cpp:24).
- Full-attention slots (derived from is_swa copy at loop.cpp:38): 5, 11, 17, 23, 29, 35, 41, 47 (8 slots). SWA slots: the other 40.
- Block starts: ip % 6 == 0, i.e. physical layers 0, 6, 12, 18 (inkling2.cpp:381).
- Blocks list (loop.cpp:81-115):
  - Prelude pushes b0.
  - Pass 0 core pushes b6, b12.
  - Pass p >= 1: bank = [b12 of the previous pass] (loop.cpp:95, blocks after prelude size + 1). bank_out = previous pass output (loop.cpp:96). Then cur and blocks reset to prelude output and prelude blocks (loop.cpp:97-98). The core pushes new b6, b12.
  - Post pushes b18 at layer 18. Output mixes [b0, b6, b12, b18, cur] (5 states max).
  - Max AttnRes states in any layer: 4 blocks + cur.
- The message mix is built per core layer (loop.cpp:106-107) from the same bank. Each core layer has its own loop control weights (loop.cpp:106).

### 1.3 Attention in one layer (inkling2.cpp:274-385)

Let `in` be the layer input (cur), `blocks` the AttnRes stack.

1. AttnRes input (SWA and full). If `attn_res_query` exists (layer 0 has none, so inputs = cur), mix(blocks, cur, attn_res_query) gives 3 views for full layers (Q, K, V) and 1 view for SWA (inkling2.cpp:298-304). Full layers use column j of the 2304x3 query. SWA uses one 2304 query.
2. Loop control (pass >= 1 only). inputs[j] = control(inputs[j], msg[j]) for j = Q, K, V (inkling2.cpp:306-311).
3. Norms (inkling2.cpp:313-324):
   - SWA with messages: attn_norm on each of the 3 inputs.
   - SWA without messages: attn_norm on inputs[0], then alias to K and V.
   - Full: attn_q_inp_norm, attn_k_inp_norm, attn_v_inp_norm, one per input.
4. Projections (inkling2.cpp:328-333). Routing: Q, R, sinks and gate read in0; K reads in1; V reads in2. SWA without messages: in1 = in2 = in0 (normed once).
   - Q = wq * in0 (Q8_0, 2304 -> 3072).
   - K = convolve(wk * in1, attn_k_conv, dynamic = attn_k_conv_proj * in1 on full layers only, state offset 0).
   - V = convolve(wv * in2, attn_v_conv, dynamic = attn_v_conv_proj * in2 on full only, state offset 1024).
   - R = attn_rel * in0 (Q4_K, 2304 -> 384 = 24 heads x 16).
5. Normalisation of q, k, v, r (inkling2.cpp:335-338):
   - Q = clip(reshape 128x24, scale attn_q_clip (128x24), thr 22.627).
   - K = clip(128x8, no scale, thr 22.627).
   - V = weightless RMS, eps 1e-6 (after the V convolve).
   - R = clip(16x24, scale attn_rel_clip (16x24), thr 8.0).
6. Tau, full layers only (inkling2.cpp:259-262, :340-343). tau = 1 + 0.1 * log(max(1, (pos+1)/8192)). Q *= tau and R *= tau.
7. Sinks: sinks = attn_sink_proj * in0 (Q4_K, 2304 -> 24) (inkling2.cpp:355).
8. Bias table and sink fold (inkling2.cpp:358-367):
   - table = pad(attn_rel_bias, 16x1024 -> 16x1025 with a zero row) x R. Shape 1025 x 24 x T (F32 weights).
   - table -= sinks, broadcast over all 1025 rows (including the zero pad row).
   - kq_b = get_rows(table, rel_pos). rel_pos is self_rel_pos (full) or self_rel_pos_swa (SWA).
9. Attention (inkling2.cpp:369-373). sinks = 0 is passed to build_attn, so the softmax has an implicit zero logit. Scale 1/sqrt(128). Causal mask. SWA layers also apply the SWA mask. GQA 24 q heads over 8 kv heads.
10. Gate and output (inkling2.cpp:375-376):
    - attn = attn * sigmoid(wqkv_gate * in0). wqkv_gate is the attn_gate tensor (Q8_0, 2304 -> 3072; inkling2.cpp:118).
    - out = convolve(wo * attn, attn_post_conv (F32 4x2304), static only, state offset 2048). wo is attn_output (Q8_0, 3072 -> 2304; inkling2.cpp:117).
11. Block rule (inkling2.cpp:381-385):
    - If ip % 6 == 0: blocks.push(inpSA). No residual add. The attention output is the AttnRes "cur".
    - Else: cur = inpSA + attn_out.
12. FFN (inkling2.cpp:387-415):
    - ffn_inp = cur.
    - x = mix(blocks, cur, ffn_res_query) (includes the block just pushed, if any). Loop control FFN if pass >= 1 (:392-394).
    - x = RMS(x) with ffn_norm (:395).
    - MoE (:397-410, see 1.7).
    - out = convolve(moe_out, ffn_post_conv (F32 4x2304), static, state offset 4352) (:412). The residual is inside this scaled sum.
    - l_out = ffn_inp + out (:415).

The residual inside `convolve` is scaled. Its total is (x + conv)/sqrt(5) for static and (x + conv)/sqrt(9) for dynamic (inkling2.cpp:237).

#### Exact bias and sink semantics

- Stored bias per query i, head h, distance d: raw_ih[d] = sum_k attn_rel_bias[k][d] * R_ih[k] for d < 1024, and raw_ih[1024] = 0 (pad row).
- Graph bias after the fold: b_ih[d] = raw_ih[d] - s_ih for all d including 1024.
- Keys j: softmax input a_ij = (q_i . k_j) / sqrt(128) + b_ih[d_ij]. Inkling2 calls build_attn with an llm_graph_input_mem_hybrid_iswa_rel (inkling2.cpp:252, :371), which selects the rel overload at llama-graph.cpp:4388. That overload adds kq_b into the mask (:4415), and soft_max_ext scales only the q.k term, so the bias is not scaled.
- Equivalent form, non-flash path (matches the comment at inkling2.cpp:360): keys in the overflow bucket have raw bias 0 (pad row), and the sink is a zero-value key with logit s_ih. With a_ij = q_i.k_j/sqrt(128) + raw_ih[d_ij], weight_ij = exp(a_ij) / (exp(s_ih) + sum_j exp(a_ij)).
- Flash path: the kq_b null check (llama-graph.cpp:3134) sees kq_b = nullptr on the rel path, so cparams.flash_attn alone decides flash. The rel overload then casts the combined mask to F16 (llama-graph.cpp:4416-4417), so the bias is rounded to F16. The default flash_attn setting is not checked; the S0 golden pins it off.
- Mask shape: the table is per head (inkling2.cpp:358-367), so the combined rel mask is per head, not shared across heads. Breccia's kernel needs a per-head additive mask (exact ne order not checked).
- Implementation contract for Breccia: the bias is not multiplied by kq_scale; it is added after the 1/sqrt(128) scale. Tau is separate: it still scales R on full layers (item 6), so do not drop tau when building the bias. Do not follow the direct kq_b path (build_attn_mha adds kq_b before soft_max_ext, llama-graph.cpp:3212-3217), which would multiply the bias by 1/sqrt(128). Inkling does not use that path.
- Attention softcap: not used. Inkling2 reads only the final-logit softcap (inkling2.cpp:28-29, :435-436), so the attention softcap branch in build_attn_mha (:3203-3210) is not set for this arch (default value not checked).
- Overflow bucket (d >= 1024) exists only on full layers. SWA masks d >= 1024 (see 1.9), so SWA never reads row 1024.

### 1.4 Shortconv (convolve, inkling2.cpp:196-238)

- Inputs: history state of n_conv-1 = 3 past inputs per channel, stored per slot at a channel offset (inkling2.cpp:209-214).
- Output = causal conv over [past | current]. The static weight is 4 taps per channel (F32 4x C). Static tap index k: c[0] multiplies the oldest sample, c[3] the current token (ggml ssm_conv, ggml-cpu/ops.cpp:10078-10121, read this pass).
- Dynamic taps on full layers (inkling2.cpp:219-234). The dynamic tensor is 2304x32, reshaped to 8 heads x 4 lags. Lag 0 is the current token and lag 3 is the oldest. This is the reverse of static tap order. Verified: window offset (n_conv-1-lag) at inkling2.cpp:226.
- Result: scale((cur + out), 1/sqrt((dynamic ? 2 : 1)*4 + 1)) (inkling2.cpp:237). No activation.
- Channel layout of the state (verified): K 0-1023, V 1024-2047, attn_post 2048-4351, ffn_post 4352-6655. Total 6656 channels (n_sconv_channels at inkling2.cpp:72 and :79; offsets at :329, :332, :376, :412).
- The static weight layout (ne0 = 4 taps fastest) matches Breccia gdn_conv's w[c*4+k] (earlier read). Breccia gdn_conv applies SiLU and is static only. Inkling uses neither.

### 1.5 AttnRes mix (inkling2.cpp:179-192)

- states = [b_newest, ..., b_oldest, cur]. Each block is prepended (concat dim 1), so cur is last (inkling2.cpp:180-183).
- states = weightless RMS(states, eps 1e-6) (:184).
- scores = states^T x queries (queries = query repeated per token). Query columns: attn_res_query 3 (full) or 1 (SWA) (inkling2.cpp:147); msg_query 4 (loop.cpp:56); ffn_res_query and output_res_query 1 (inkling2.cpp:102, :149).
- softmax over states with scale 1/sqrt(n_embd) (2304), no mask, no sink (:189).
- Output = sum over states of softmax weight times the RMS-normalised states (inkling2.cpp:191). Block pushes and residual adds (:381-385, :415) use the raw inpSA and ffn_inp, not the normalised states.
- out = states x weights (:191).

### 1.6 Loop control (inkling2.cpp:241-249, verified 2026-11-23)

- gate = ctrl_down (4608 -> 64) * concat(RMS(cur), RMS(msg)) (eps 1e-6). 4608 = 2 x 2304.
- gain = tanh(ctrl_gain * gate). forget = tanh(ctrl_forget * gate). Both 64 -> 2304.
- out = cur - forget * cur + gain * msg.
- msg = mix(bank, bank_out, msg_query) column ctrl (loop.cpp:106-107), ctrl in {Q, K, V, FFN}. msg_query is 2304x4.

### 1.7 MoE (inkling2.cpp:397-410; llama-graph.cpp 2531-2545 and 2602-2616 read 2026-11-23; 2286-2490 grepped only)

- Router logits = ffn_gate_inp (F32, 2304 -> 128) * x.
- probs = sigmoid(logits). Selection score = probs + exp_probs_b (F32 128). The bias affects selection only (llama-graph.cpp:2396-2399; sigmoid branch :2409).
- Top-8 by selection score (llama-graph.cpp:2441).
- Weights gathered from the unbiased probs (gather line not re-checked). Normalised as sum + 1e-20, from expert_weights_norm_eps (inkling2.cpp:57; applied at llama-graph.cpp:2475-2476). Scaled by 8.0 (llama-graph.cpp:2488-2489).
- Experts: gate_up fused, Q4_K, shape 2304 x 2304 x 128 (inkling2.cpp:155; passed to build_moe_ffn at :410; fused path llama-graph.cpp:2531-2533). Gate = rows [0,1152), up = rows [1152,2304) (read 2026-11-23, llama-graph.cpp:2541-2544).
- down: 1152 -> 2304 per expert. Quant varies per block (see 1.10).
- Activation SITU (llama-graph.cpp:2602-2616, read this pass):
  - act = beta * tanh(gate/beta) * sigmoid(gate), beta = 4.0.
  - up' = lb * tanh(up/lb), lb = 25.0.
  - out = act * up'. Node name ffn_moe_situ.
- No shared expert in the graph.

### 1.8 Output head (inkling2.cpp:421-446)

- mix(blocks, cur, output_res_query) (:422). Blocks = [b0, b6, b12, b18] + cur.
- RMS with output_norm (:423).
- get_rows by inp_out_ids (:425-427).
- lm_head = output.weight (Q6_K, 2304 x 201024). Softcap 30 * tanh(x/30) and slice to 200058 (softcap :435-439, slice :442; verified).

### 1.9 Masks and relative index fill

- Causal skip when p0 > p1. SWA mask when p1 - p0 >= n_swa (llama-hparams.h:563-, UNCONFIRMED for this path). llama-graph.cpp:636-648 is llm_graph_input_attn_no_cache::set_input (:608), not the KV-cache path Inkling uses. The KV-cache fill (llama-kv-cache.cpp set_input_attn_mask) was not read. SWA distances 0..1023 is derived from the rule above and is UNCONFIRMED until that fill is read.
- Relative index (llama-kv-cache.cpp:2010-2022, earlier read): distance = pos_i - pos_j for non-empty cells. Index = distance if 0 <= distance < extent, else extent (1024). extent = n_swa (1024) for SWA and n_relative_extent (1024) for full (inkling2.cpp:111).

### 1.10 Tensor inventory (hdr.txt, 704 tensors)

- Quant mix: F32 478, Q8_0 140, Q4_K 73, Q5_0 12, Q6_K 1. Filename says Q4_K_M but the content is mixed. file_type 15 is the nominal Q4_K_M label.
- Q8_0 140 = 24 layers x 5 dense (attn_q, attn_k, attn_v, attn_output, attn_gate) + 4 full layers x 2 (k/v_conv_proj) + 12 ffn_down blocks. The dense Q8_0 projections are in every layer, not only the MoE.
- ffn_down_exps per block (1152 x 2304 x 128):
  - Q8_0: blocks 0-5, 8, 11, 14, 17, 20, 23 (12 blocks).
  - Q5_0: blocks 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22 (12 blocks).
- ffn_gate_up_exps: Q4_K on all 24 blocks.
- Q4_K 73 = 72 (attn_rel, attn_sink_proj, gate_up for 24 blocks each) + token_embd.weight (hdr.txt:70). The TYPE COUNTS line (hdr.txt:66) says 73, which matches the tensor lines once that line itself is excluded from the grep.
- Loop tensors: 156 = 13 per core block x 12 physical core blocks (msg_query 2304x4; ctrl_down 4608x64, ctrl_gain 64x2304, ctrl_forget 64x2304 for each of Q, K, V, FFN). All F32.
- F32: norms, attn_rel_bias (16x1024), attn_rel_clip (16x24), attn_q_clip (128x24), conv weights (4x1024, 4x2304), exp_probs_b, ffn_gate_inp (2304x128), attn_res_query, ffn_res_query, output_res_query, output_norm.
- output.weight Q6_K.
- Layer 0 has no attn_res_query, no attn_q/k/v_inp_norm. It has attn_norm and ffn_res_query (inkling2.cpp:134-149).
- Tensor name to struct field mapping verified: attn_gate -> wqkv_gate, attn_output -> wo (inkling2.cpp:117-118).

---

## 2. Breccia coupling map

Classes: REPLACE = must change for Inkling. EXTEND = add a path or variant, qwen path unchanged. OPTIONAL = guard or feature-gate. UNTOUCHED = reuse as is.

| # | Location (breccia-coord/wt) | What it assumes | Inkling needs | Class |
|---|---|---|---|---|
| 1 | crates/breccia-engine/src/config.rs:176-248 from_metadata | qwen4exp keys: indexer, PLE, hyper_connection, full_attention_interval, ssm, rope.freq_base | inkling2_loop.* keys: relative, log_scale, SITU, loop, attn_res, softcap | REPLACE (arch-keyed reader) |
| 2 | config.rs:252-261 validate | bails unless arch == qwen4exp | allow inkling2_loop with the checks from inkling2.cpp:38-40 and loop.cpp:19-20 | REPLACE |
| 3 | config.rs:308-312 layer_kind | (i+1) % qsa_interval -> Qsa, else Gdn | per-layer Rel (SWA or full) plus loop schedule | REPLACE |
| 4 | config.rs:240-245 BRECCIA_EXPERTS_USED clamp | top-k <= 16 | 8 is fine | UNTOUCHED |
| 5 | model.rs:368-375 load | ShardSet mmap | same | UNTOUCHED |
| 6 | model.rs:582-600 block loop, MoeW | MoE requires shexp (:589-600) | no shared expert: make shexp optional | REPLACE (MoeW) |
| 7 | model.rs:601-611 gate_up concat | separate gate and up tensors | fused ffn_gate_up_exps, split at row 1152 | REPLACE (or view) |
| 8 | model.rs:706-709 expert keys | ffn_gate_exps, ffn_up_exps, ffn_down_exps | ffn_gate_up_exps + ffn_down_exps (per-block type) | REPLACE |
| 9 | model.rs:587-588, :1931 hc_weights | hyper-connection weights | none; AttnRes weights instead | REPLACE (Inkling path) |
| 10 | model.rs:612-648 GdnW, QsaW | qwen tensors | Rel attention tensors, attn_res_query, ffn_res_query, sink, gate, post_conv, loop ctrl | REPLACE (new structs) |
| 11 | model.rs:649-652 "no routed-expert tensors" error | name check | fine after 6-8 | OPTIONAL |
| 12 | graph.rs:7732-7802 run_gdn, mixer_chunked; :7808 record_hc_chunk | GDN plus HC | not used by Inkling | UNTOUCHED (qwen); new Inkling record path |
| 13 | graph.rs:8158-8226 run_qsa | QSA: n_blocks, max_ids 512 | Inkling rel attention | REPLACE (new path) |
| 14 | graph.rs:3519-3520, :7790-7794, :7921, :8076-8077 LayerKind dispatch | 2 kinds | add Rel and loop schedule | EXTEND |
| 15 | graph.rs:7671 run_placed_moe | MoE path (shexp handling UNCONFIRMED) | sigmoid router, selection bias, SITU, fused gate_up | EXTEND |
| 16 | graph.rs:9957 "has no kernel" | unsupported quant | Q8_0 / Q5_0 | EXTEND |
| 17 | crates/breccia-engine/src/head.rs:235-244 handoff_bytes(hc_dim) | one [hc, n_embd] f32 stack, 40 KiB | AttnRes stack: up to 4 blocks + cur | REPLACE |
| 18 | placement.rs:339-350 per-layer bytes | GDN constants 128*48*128*4 + 3*10240*4 | Inkling per-layer bytes: conv state, KV per slot | REPLACE (derive from arch) |
| 19 | placement.rs placement rule | measured capabilities, no host list | same | UNTOUCHED (BRIEF.md:44) |
| 20 | crates/breccia-format/src/layout.rs:96-98 ExpertRole | gate, up, down suffixes | ffn_gate_up_exps not recognised (:135-141 returns None) | EXTEND |
| 21 | breccia-format/src/pack.rs:147-157 | one gate_up_type, one down_type, offsets | per-block down type (Q8_0 or Q5_0), fused gate_up Q4_K | REPLACE |
| 22 | breccia-format/src/quant.rs:148-150 geometry | Q5_0 (32,22), Q8_0 (32,34) | reuse | UNTOUCHED |
| 23 | breccia-format/src/tokenizer.rs:101-102 gpt2 only; :309-311 qwen35 pre-tokenizer | no inkling2 pre type | GPT-4o regex, no BOS, ignore_merges = true, clean_spaces = false (llama-vocab.cpp:2671-2676) | EXTEND |
| 24 | breccia-kernels/src/quant.rs:213,215 NOT_IMPLEMENTED | Q8_0, Q5_0 absent | Q8_0 dense + MoE, Q5_0 MoE | EXTEND (hard blocker) |
| 25 | breccia-kernels/src/gemm.rs:676-683 | errors outside multi-user GEMM set | add Q8_0 | EXTEND |
| 26 | breccia-kernels/src/real.rs:367, :566 | "has no kernel yet" | Q8_0 | EXTEND |
| 27 | breccia-kernels/src/ops/mod.rs:334 RmsNorm | wgroups 0 = weightless | AttnRes, V norm, control | UNTOUCHED (reuse) |
| 28 | ops/mod.rs:467 RopeTable, :537 Rope | RoPE | Inkling has no RoPE | UNTOUCHED (unused) |
| 29 | ops/mod.rs:718 SwiGlu | silu gate | not used (SITU) | UNTOUCHED |
| 30 | ops/mod.rs:817 Softmax | no mask, no sink | AttnRes softmax with scale 1/sqrt(2304); attention with bias | EXTEND (scale arg or new op) |
| 31 | ops/mod.rs:903 Router | softmax over all, then top-k (n_expert <= 512, k <= 16) | sigmoid, selection bias, unbiased weights, renorm eps 1e-20, scale 8 | EXTEND |
| 32 | ops/mod.rs:1218-1228 ActMode | Silu, SiluDiv, Sigmoid, Scale, SiluScale | SITU; needs tanh (no tanh op) | EXTEND |
| 33 | ops/mod.rs:2387 Add | residual add | residual and control | UNTOUCHED |
| 34 | crates/breccia-kernels/src/moe.rs:133 | qwen shared-expert sigmoid | none for Inkling | OPTIONAL (guard) |
| 35 | crates/breccia-kernels/src/qsa/mod.rs:1-30, :74 MODEL_HEAD_DIM 256 | 256 head, 2 KV heads, indexer, top-512 blocks | 128 head, 8 KV heads, 1024 window, bias | REPLACE for Inkling (do not reuse qsa) |
| 36 | crates/breccia-engine/src/mtp_draft.rs:77-78 | Q8_0 widen fallback | Inkling has no MTP (loop.cpp:19-20 requires n_layer_nextn = 0) | OPTIONAL (disable MTP for Inkling) |
| 37 | crates/breccia-chat/src/template.rs:74, :80 | think tags <think>, <seed:think>, <\|START_THINKING\|> | <\|content_thinking\|> (earlier read) | EXTEND |
| 38 | docs/BRIEF.md:44 agnosticism invariant | shapes from GGUF | Inkling fits: 128 experts (<= 512), top-8 (<= 16), n_embd 2304 | UNTOUCHED (keep the rule) |

Scale of the change: items 1-3, 6-10, 13-14, 17-18, 21 touch the engine core. Items 24-32 are kernels. Items 22, 27-29, 33 are reused.

---

## 3. Kernel coverage and gaps

| Op | Inkling source | Breccia today | Gap | Size (est.) |
|---|---|---|---|---|
| Dense Q8_0 GEMV/GEMM: attn_q/k/v/output/gate (24 layers), k/v_conv_proj (4 full layers) | inkling2.cpp:328-333, :375-376 | none (quant.rs:213) | new dequant + GEMV + GEMM. Hard blocker | large |
| MoE down Q8_0 (12 blocks) | llama-graph.cpp build_moe | none | MoE GEMV Q8_0. Hard blocker | medium |
| MoE down Q5_0 (12 blocks) | same | none (Breccia has Q5_K, a different block format) | MoE GEMV Q5_0. Hard blocker | medium |
| Q4_K GEMV/GEMM: attn_rel, attn_sink_proj, gate_up_exps | inkling2.cpp:333, :355; llama-graph.cpp:2531 | gemv_q4_k, gemm_q4_k, gemv_q4_k_moe | reuse | none |
| Q6_K head | inkling2.cpp:421-446 | gemv_q6_k, gemm_q6_k | reuse | none |
| Fused gate_up split (rows 0-1151 gate, 1152-2303 up) | llama-graph.cpp:2541-2544 | separate gate and up GEMV | row offset into fused tensor | small |
| Router: sigmoid, selection bias, unbiased weights, renorm 1e-20, scale 8 | llama-graph.cpp:2286-2490 | ops/mod.rs:903 softmax | new mode and bias input | small-medium |
| SITU: beta tanh(g/beta) sigmoid(g) times lb tanh(u/lb) | llama-graph.cpp:2602-2616 | ActMode has no tanh | tanh op + SITU act (shader) | small |
| Softcap 30 tanh(x/30) | inkling2.cpp:435-439 | none | tanh elementwise (shared with SITU) | trivial |
| Row L2 clip with optional per-channel scale | inkling2.cpp:165-176 | none | new elementwise + row reduce | small |
| tau = 1 + 0.1 log(max(1,(pos+1)/8192)) multiply on Q and R, full layers | inkling2.cpp:259-262, :340-343 | none | per-token scalar multiply | trivial |
| Weightless RMS (eps 1e-6): V, AttnRes states, control inputs | inkling2.cpp:184, :337, :395 | RmsNorm wgroups 0 | reuse | none |
| AttnRes mix: RMS, query GEMM, softmax (scale 1/sqrt(2304), no mask), weighted sum | inkling2.cpp:179-192 | RmsNorm, GEMM, Softmax (scale arg UNCONFIRMED), no weighted-sum op | compose + maybe 1 small shader | small |
| Loop control: 4608->64 gate, two 64->2304 GEMMs, tanh, combine | inkling2.cpp:241-249 | GEMM only | needs tanh (shared) + composition | small |
| Shortconv: static 4 taps + dynamic per-head taps, state store, residual in scaled sum, no activation | inkling2.cpp:196-238 | gdn_conv (static, SiLU, oldest-first state) | new variant: no SiLU, dynamic taps, scale | medium |
| Relative table: F32 16 -> 1025 x 24 per token, sink subtract, gather by distance | inkling2.cpp:355-367 | none | small GEMM + subtract + gather | small-medium |
| Attention with additive bias (added after the 1/sqrt(128) scale, see 1.3), zero sink, causal + SWA 1024 mask, 24 q / 8 kv heads, head 128, scale 1/sqrt(128) | inkling2.cpp:369-373; dense base mask llama-graph.cpp:4359 (build_attn_inp_kq_mask), fill in llama-kv-cache.cpp set_input_attn_mask (not read, UNCONFIRMED), rel bias per head (see 1.3) | qsa (256 head, 2 kv, indexer, top-512 blocks) | new kernel. Largest single item | large |
| Sigmoid gate multiply on attention output | inkling2.cpp:375 | Act Sigmoid | reuse + elementwise mul (Mul op UNCONFIRMED) | trivial |
| Output: mix, RMS, get_rows, lm_head, trim vocab | inkling2.cpp:421-446 | Q6_K head | composition | small |
| Static shortconv on attn_out and ffn_out | inkling2.cpp:376, :412 | see shortconv row | same variant | (in shortconv) |
| No shared expert | inkling2.cpp (graph has no shexp) | MoeW requires shexp | make optional | small |

F32 handling: many Inkling tensors are F32 (norms, router, clips, conv weights, rel bias). Whether Breccia's upload_tensor (model.rs:1836) and the kernels take F32 weights directly is UNCONFIRMED.

### Blockers in order

1. Q8_0 dense GEMV/GEMM (every layer). Without it no layer runs. This is larger than the earlier summary implied, which named only the MoE Q8_0.
2. Q8_0 and Q5_0 MoE down.
3. Relative attention kernel with bias, sink and SWA mask (every layer).
4. Shortconv variant (every layer).

The "widen Q8_0 to fp16" fallback (mtp_draft.rs:77-78) saves nothing for these: about 8.1 GB at fp16 for the Q8_0 weights, so it is not a plan.

---

## 4. Minimal Arch abstraction and ordered plan

### 4.1 Abstraction (sketch, not applied)

Keep the qwen4exp path byte-for-byte. Do not add a trait over architectures: the two graphs differ in residual type (HC vs AttnRes), attention (GDN/QSA vs rel), and loop structure. A trait would make shared code branch on arch anyway.

```rust
// sketch only
pub enum Arch { Qwen4Exp, Inkling2Loop }   // from general.architecture

pub enum Attn { Gdn, Qsa, Rel { window: Option<u32> } }   // window = Some(1024) for SWA
pub enum Residual { Hc, AttnRes }
pub enum FfnAct { Silu, Situ { beta: f32, lin_beta: f32 } }

pub struct LoopPlan {
    pub n_phys: usize,              // 24
    pub prelude: Range<usize>,      // 0..6
    pub core: Range<usize>,         // 6..18
    pub passes: usize,              // 3
    pub post: Range<usize>,         // 18..24
}
impl LoopPlan {
    // pass 0 uses the physical index; later passes map into cache slots n_phys..
    pub fn slot(&self, pass: usize, ip: usize) -> usize {
        if pass == 0 { ip } else { self.n_phys + (pass - 1) * self.core.len() + (ip - self.core.start) }
    }
}
```

- `Arch` is read once in config.rs. Inkling keys use the prefix `inkling2_loop.*`.
- `LoopPlan` is built only for Inkling. Its schedule is structured (prelude, core x passes, post), not a flat list. The bank for pass p >= 1 is [b12 of pass p-1] plus pass p-1 output (loop.cpp:95-96).
- Handoff: `Stack { blocks: Vec<Buf> (max 4), cur: Buf }` replaces the HC stack for Inkling (head.rs:235-244).
- Per-layer state and KV sized from `LoopPlan` slots (48 slots, 8 full, 40 SWA).
- Kernels are shared. Graph code for Inkling lives in its own module and calls them.

### 4.2 Arithmetic (not measured)

- KV per token per full slot at f16: 8 heads x 128 x 2 (K and V) x 2 bytes = 4 KiB. Eight full slots: 32 KiB per token. About 3.1 GiB at 100k tokens, about 30.5 GiB at 1M, before overhead.
- SWA slots bounded to the 1024 window: 40 x 1024 x 4 KiB = 160 MiB.
- Conv state per slot: 3 x 6656 x 4 bytes = 78 KiB. About 3.7 MiB for 48 slots.
- Quantised KV (q6_0 K+V, as in Breccia's default) would roughly halve the full-slot cost. Not measured.

### 4.3 Ordered plan, with gates and size estimates

Estimates are rough line counts (new code plus tests), about +/-50%, not measured. Shader counts are new shaders. AGENTS.md:107-108 requires one new shader per process and at most 4 fresh dispatched shapes per process, so each new shader needs its own conformance run.

| Stage | Work | Gate | Size (est.) | Shaders |
|---|---|---|---|---|
| S0 | Finish GGUF: validate sizes, offsets, checksum (read-only). Freeze llama.cpp-inkling commit. Build a llama.cpp CPU golden: 16-token prompt, eval callback for l_out per layer, kq_b, ffn_moe_situ, router ids. Flash attention pinned off (flag name UNCONFIRMED for this fork); optional second golden with it on, to size the F16 bias rounding | GGUF complete and stable. Golden reproducible on two runs | 150 (Python, outside the repo) | 0 |
| S1 | Arch enum, Inkling config keys, tensor binding (fused gate_up, per-block down type, optional shexp, rel/sink/gate/post_conv, res queries, loop tensors). Inkling refuses to run until its ops exist | Header-only bind of the real file: all 704 tensors bound with asserted shapes. qwen4exp unit tests unchanged | 500 | 0 |
| S2 | Q8_0 dequant + GEMV + GEMM, default-off flag | Per-op parity on attn_q output vs golden, one quant type per run. Tolerance: UNCONFIRMED, set from measured noise | 500 | 2 |
| S3 | MoE GEMV for Q8_0 and Q5_0 (down) | Expert output parity on layer 0 and layer 6 | 400 | 2 |
| S4 | tanh, row clip (with channel scale), tau multiply, softcap | Per-op parity vs golden | 300 | 2 |
| S5 | Router: sigmoid, selection bias, unbiased weights, renorm 1e-20, scale 8 | Top-8 expert ids match golden on 16 tokens | 250 | 1 |
| S6 | SITU act and fused gate/up split (row offsets) | ffn_moe_situ parity on layer 0 | 150 | 1 |
| S7 | AttnRes mix: RMS, query GEMM, softmax with scale, weighted sum | Mix output parity on blocks 0-6 | 250 | 1 |
| S8 | Shortconv variant: static and dynamic taps, state store, residual in scale, no activation | Layer 0 (SWA) and layer 5 (full) conv output parity | 350 | 1 |
| S9 | Rel attention: rel table (F32 GEMM, subtract sink, gather), bias attention with zero sink, causal + SWA mask, GQA 3:1, head 128. Decode first, then chunked prefill | Layer 0 and layer 5 attention output parity at context 16. Then context 1100 so the overflow bucket on full layers is exercised | 1000 | 2-3 |
| S10 | Loop control and message bank (composition of S4, S7 ops) | Pass 1 and pass 2 layer 6 output parity | 200 | 0 |
| S11 | Engine graph: prelude, core x3 with slots, post, output. Handoff stack. Per-slot KV and conv state. Placement bytes from arch | Full-stack hidden-state parity after prelude, after pass 0, after each loop pass | 1200 | 0 |
| S12 | Tokenizer: inkling2 pre type (GPT-4o regex, no BOS, ignore_merges, clean_spaces off). Chat template kwargs and think tags | Token id parity on 50 prompts vs llama.cpp | 200 | 0 |
| S13 | Logit parity: 32-token greedy at 16-64 tokens, then ~2k, then >1024 | 32/32 greedy tokens match golden, top-1 logit within tolerance (UNCONFIRMED) | tests 500 | 0 |
| S14 | Dual-GPU placement and perf (not a parity gate). Only after S13 | Bench per Breccia rules (AGENTS.md:111: parity first, benchmarks second) | 300 | 0 |

Total new code: about 6,000-6,500 lines (estimate). About 12 new shaders.

Per-process rule (AGENTS.md:36, :46-47, :107-109): every new shader gets its own conformance invocation, `timeout 90`, spirv-val in build.rs (AGENTS.md:103), VK_LAYER_KHRONOS_validation on first runs, `BRECCIA_KERNEL_DEVICE` gate, default-off flag. A GPU reset is a stop-the-line event (AGENTS.md:115).

### 4.4 Verification summary

- V0 (no GPU): GGUF completeness and binding (S0, S1). Config and loader unit tests.
- V1 (per op, conformance): parity against the llama.cpp golden tensors (S2-S9). One device per process, both cards back to back (AGENTS.md:48). One new shader per process, timeout 90, VK validation (AGENTS.md:107-109). Single-device runs need a named reason (AGENTS.md:68).
- V2 (structure): hidden-state parity after prelude, after pass 0, after each loop pass (S10, S11). The loop slot mapping and bank are where a bug would show first.
- V3 (end to end): greedy parity 32/32 at short context, then at 2k, then past 1024 so the overflow bucket and tau are exercised (S13). Run in the default dual run shape, both devices by PCI id with dpm high (AGENTS.md:62). Breccia's qwen4exp parity set (32/32 tokens, top-1 13.170561; AGENTS.md:112) stays as the regression guard for every stage that touches shared code.
- V4 (perf): only after V3 passes (S14).

---

## 5. Open items and risks

UNCONFIRMED:
1. GGUF is still being written. Data offsets and checksums not validated. S0 gates everything else.
2. Gate and up row order in the source checkpoint. Both converter branches (conversion/inkling2.py: packed-experts :140-176, yield :251-254; w13 de-interleave reshape :255-258; read 2026-11-23) assume interleaved source rows and output contiguous [gate; up] halves, which is the runtime contract (llama-graph.cpp:2541-2544). UNCONFIRMED: that the source checkpoint is interleaved as assumed. Check one expert row against llama.cpp.
3. Static shortconv tap order: ggml ssm_conv checked this pass. Breccia gdn_conv tap order matched in an earlier pass (ops.cpp:10120-10121, gdn_conv.comp). Re-check in S8.
4. Breccia Softmax scale argument (ops/mod.rs:817): UNCONFIRMED.
5. Breccia Mul op for the sigmoid gate: UNCONFIRMED.
6. Breccia recurrent state dtype: UNCONFIRMED.
7. Breccia tokenizer: whether qwen35 pre-tokenizer matches the GPT-4o regex: UNCONFIRMED. ignore_merges: UNCONFIRMED.
8. Resolved 2026-11-23: attn_gate -> wqkv_gate, attn_output -> wo (inkling2.cpp:117-118).
9. Q4_K tensor count: resolved. 73 = 72 per-layer tensors + token_embd.weight (hdr.txt:70; TYPE COUNTS hdr.txt:66).
10. SWA KV storage in llama.cpp (bounded cells or full context): UNCONFIRMED. Breccia's behaviour: not checked.
11. Citations marked "(earlier read)" (kv-cache fill, hparams mask) were not re-read. MoE router lines (llama-graph.cpp 2396-2488) were grepped on 2026-11-23, not read in full. The SITU and fused split were read (2525-2620), and the output softcap was read (inkling2.cpp:435-442).

Risks:
- Dense Q8_0 is in every layer. Schedule for S2 comes first.
- Loop bank: pass >= 1 uses only [b12 of the previous pass] and the previous pass output. Easy to get wrong.
- Overflow bucket: row 1024 is the zero pad, then minus the sink. Dropping the sink subtract on the pad row changes overflow-bucket logits. Only full layers reach it.
- Per-block down quant: pack.rs assumes one down_type (pack.rs:149). Mixed Q8_0 and Q5_0 needs a per-block type.
- No shexp: MoeW requires it (model.rs:589-600).
- Fused gate_up: layout.rs:135-141 does not recognise it.
- The GGUF name says Q4_K_M but contains Q8_0, Q5_0 and Q6_K. Plans must follow the tensors, not the name.
- Context metadata is 1M. The KV arithmetic above assumes f16. Breccia's placement must account for it.
- Tokenizer and chat template mismatch shows up as wrong outputs, not errors. Gate it with token id parity (S12).

Not done: no GPU run, no build, no edits in the Breccia repo, no data offset or checksum check, no llama.cpp run, no Breccia code read beyond the cites above. Verification pass 2026-11-23 re-read inkling2.cpp and inkling2-loop.cpp, and grepped hdr.txt, llama-graph.cpp and AGENTS.md. Not re-read: llama-kv-cache.cpp, llama-hparams.h, Breccia source.
