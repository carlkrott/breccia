#!/usr/bin/env bash
# A/B: --moe-cache-mib {0,512,1024} on the production qwen4exp HIP build.
# Baseline (docs/phase-0/01): 20.79 tok/s @ moe-cache 0. Everything else identical
# to qwen-dual-serve.sh. Ends each run cleanly; leaves no services started.
set -uo pipefail
BIN=/home/korphaus/llama.cpp-qwen4exp-mmap/build-hip-q5kv/bin/llama-server
M=/srv/strata-data/models
PORT=8164
OUT=${1:-/home/korphaus/projects/Breccia/logs/moe-cache-ab-$(date +%Y%m%d-%H%M).log}
SLOT=/mnt/agent-storage-a/kv-slots/breccia-moeab/
mkdir -p "$SLOT"

echo "# moe-cache A/B $(date -u +%FT%TZ)" | tee "$OUT"
for CACHE in 0 512 1024; do
  echo "## cache=$CACHE MiB" | tee -a "$OUT"
  sudo -n env GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LLAMA_MMAP_FADVISE=random LLAMA_MMAP_CPU_MADVISE=random \
    "$BIN" -m "$M/IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf" \
    --device ROCm0,ROCm1 -ngl 99 -ts 9,15 \
    -ot 'blk\.(0|1|2|4|8|18|20|22|23)\.ffn_(up|down|gate)_exps\.=ROCm0,blk\.(24|35)\.ffn_(up|down|gate)_exps\.=ROCm1' \
    -cmoe --moe-cache-mib "$CACHE" -lzm on -lm none --no-host \
    -c 200000 -kvu -np 1 --kv-unified-per-slot 200000 -ctk q4_0 -ctv q4_0 -fa on \
    -cram 0 -ctxcp 1 --slot-save-path "$SLOT" \
    -lv 3 --reasoning off \
    -t 16 -tb 16 --host 127.0.0.1 --port "$PORT" --jinja --metrics --fit off --no-op-offload -ub 256 \
    > "/tmp/moeab-$CACHE.log" 2>&1 &
  SRV=$!
  ok=0
  for i in $(seq 1 60); do sleep 5; curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ok=1; break; }; kill -0 $SRV 2>/dev/null || break; done
  if [ $ok -ne 1 ]; then echo "FAILED to become healthy (cache=$CACHE); see /tmp/moeab-$CACHE.log" | tee -a "$OUT"; tail -5 "/tmp/moeab-$CACHE.log" | tee -a "$OUT"; kill $SRV 2>/dev/null; sleep 3; continue; fi
  timeout 300 python3 /home/korphaus/projects/Breccia/harnesses/bench_decode.py \
    --url "http://127.0.0.1:$PORT" --out "/tmp/moeab-$CACHE.json" --runs 3 --gen-tokens 256 \
    --tag "moe-cache-$CACHE" 2>&1 | grep -vi deprecat | tee -a "$OUT"
  echo "server timings:" | tee -a "$OUT"
  grep -E 'eval time|tg = |prompt eval time' "/tmp/moeab-$CACHE.log" | grep -v '0.00' | tail -6 | tee -a "$OUT"
  kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; sleep 5
  echo | tee -a "$OUT"
done
rm -rf "$SLOT"
echo "# done $(date -u +%FT%TZ)" | tee -a "$OUT"
