#!/bin/bash
# Stage 0 reference: llama.cpp ROCm (qwen-dual flags, f16 KV, 8k ctx), mmproj on CPU, raw prompt + top-20 probs.
set -f
cd /var/tmp/breccia-coord/vision
export LLAMA_MEDIA_MARKER="<__media__>" GGML_VK_VISIBLE_DEVICES=0,1 LLAMA_MMAP_FADVISE=random LLAMA_MMAP_CPU_MADVISE=random
D=/home/korphaus/llama.cpp-qwen4exp-mmap/build-vulkan/bin
M=/srv/strata-data/models
$D/llama-server -m $M/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --device Vulkan0,Vulkan1 -ngl 99 -ts 9,15 \
  -ot 'blk\.(0|1|2|4|8|18|20|22|23)\.ffn_(up|down|gate)_exps\.=Vulkan0,blk\.(24|35)\.ffn_(up|down|gate)_exps\.=Vulkan1' \
  -cmoe --moe-cache-mib 0 -lzm on -lm none --no-host -c 8192 -np 1 -fa on -cram 0 \
  --mmproj $M/mmproj-Qwen3.8-Flash-Next-BF16.gguf --no-mmproj-offload --image-min-tokens 1024 --image-max-tokens 2048 \
  -t 16 -tb 16 --host 127.0.0.1 --port 8199 --fit off --no-op-offload -ub 256 > ref/server.log 2>&1 &
S=$!
for i in $(seq 1 120); do curl -sf -m 2 127.0.0.1:8199/health >/dev/null && break; kill -0 $S 2>/dev/null || { echo "server died"; exit 1; }; sleep 5; done
for sz in wide; do
  python3 - "$sz" <<'PY'
import base64,json,sys
sz=sys.argv[1]
b=base64.b64encode(open('/home/korphaus/Pictures/variety-copied-wallpaper-8d16c88555ef4d72ad93d322239cbe4a.jpg','rb').read()).decode()
p="<|im_start|>user\n<__media__>Describe this image in one sentence.<|im_end|>\n<|im_start|>assistant\n"
json.dump({"prompt":{"prompt_string":p,"multimodal_data":[b]},"n_predict":48,"temperature":0,"top_k":1,
 "n_probs":20,"post_sampling_probs":False,"return_tokens":True,"cache_prompt":False},open(f'ref/req{sz}.json','w'))
PY
  curl -s -m 900 127.0.0.1:8199/completion -H 'content-type: application/json' -d @ref/req$sz.json > ref/resp$sz.json
  echo "== $sz"; python3 -c "import json;r=json.load(open('ref/resp$sz.json'));print(repr(r.get('content')));print(r.get('timings'));print(r.get('tokens_evaluated'))"
done
kill $S; wait $S 2>/dev/null; echo REF DONE
