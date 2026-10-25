#!/usr/bin/env bash
# bench_attrib.sh — llama-bench runs with per-GPU busy%% attribution.
#
# llama-bench in the qwen4exp fork labels the backend "ROCm" even when HIP init
# failed and it silently fell back to CPU (see docs/phase-0/03-*.md), so we never
# trust the label alone: we sample /sys busy%% for every card while it runs.
#
# NOTE: HIP needs /dev/kfd (root:gpu-compute); korphaus is not in gpu-compute, so
# HIP runs go through passwordless sudo (service units get SupplementaryGroups=gpu-compute
# instead). Vulkan/RADR only needs the world-readable render nodes.
#
# Usage: bench_attrib.sh <hip|vulkan> <device-index> [extra llama-bench args...]
set -u
BACKEND="$1"; DEV="$2"; shift 2
EXTRA=("$@")

case "$BACKEND" in
  hip)    BIN=/home/korphaus/llama.cpp-qwen4exp-mmap/build-hip-q5kv/bin/llama-bench
          # llama-bench -d does NOT isolate a device (layer-split across all visible
          # devices stays on) — pin via visibility env instead. HIP index: 0=RX 6950 XT,
          # 1=Radeon VII/MI50, 2=iGPU (per ggml banner). Set ONLY CUDA_VISIBLE_DEVICES:
          # setting CUDA+HIP+ROCR together double-masks and HIP init fails outright
          # ("no ROCm-capable device is detected"). Needs /dev/kfd -> sudo.
          WRAP=(sudo -n env -u CUDA_VISIBLE_DEVICES -u HIP_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES
                CUDA_VISIBLE_DEVICES="$DEV") ;;
  vulkan) BIN=/home/korphaus/llama.cpp-qwen4exp-mmap/build-vulkan/bin/llama-bench
          # MESA_VK_DEVICE_SELECT=1002:164e! is inherited in this session and hides the
          # discrete GPUs from RADV (docs/phase-0/03-*.md). GGML_VK_VISIBLE_DEVICES picks
          # from ggml's own enumeration: 0=iGPU, 1=RX 6950 XT, 2=Radeon VII.
          WRAP=(env -u MESA_VK_DEVICE_SELECT GGML_VK_VISIBLE_DEVICES="$DEV") ;;
  *) echo "backend must be hip|vulkan" >&2; exit 2 ;;
esac
MODEL=${MODEL:-/home/korphaus/.cache/eg2-b/embeddinggemma-2-Q8_0.gguf}

# card -> pci id mapping for attribution (only real cards, not connector dirs)
declare -A CARD_PCI
for c in /sys/class/drm/card[0-9]*; do
  n=$(basename "$c")
  [[ "$n" =~ ^card[0-9]+$ ]] || continue          # skip card0-DP-4 etc
  [ -e "$c/device/device" ] || continue
  CARD_PCI[$n]="$(cat "$c/device/device" 2>/dev/null)"
done
CARDS=($(ls -d /sys/class/drm/card[0-9]* | while read -r c; do n=$(basename "$c"); [[ "$n" =~ ^card[0-9]+$ ]] && echo "$n"; done | sort -V))
declare -A CARD_IDX; i=0; for n in "${CARDS[@]}"; do CARD_IDX[$n]=$i; i=$((i+1)); done

BUSYLOG=$(mktemp)
sample() { # sampler runs for at most 400s
  local end=$((SECONDS+400))
  while [ $SECONDS -lt $end ]; do
    local line=""
    for n in "${CARDS[@]}"; do
      line+=" $(cat /sys/class/drm/$n/device/gpu_busy_percent 2>/dev/null || echo -)"
    done
    echo "$line" >> "$BUSYLOG"
    sleep 0.4
  done
}

echo "== $BACKEND pin $DEV  (cards: $(for n in "${CARDS[@]}"; do echo -n "$n=0x${CARD_PCI[$n]} "; done)) =="
sample & SAMP=$!
"${WRAP[@]}" "$BIN" -m "$MODEL" -p 512 -n 128 -t 16 -r 3 "${EXTRA[@]}" 2>&1 | tail -5
kill "$SAMP" 2>/dev/null; wait "$SAMP" 2>/dev/null

echo "-- busy%% peaks per card --"
LABELS=""; for n in "${CARDS[@]}"; do LABELS+="$n(0x${CARD_PCI[$n]}) "; done
awk -v names="$LABELS" 'BEGIN{split(names,n," ")} {for(i=1;i<=NF;i++) if($i+0>m[i]+0) m[i]=$i+0} END{for(i=1;i in n;i++) printf "%s_peak=%s%% ", n[i], m[i]; print ""}' "$BUSYLOG"
rm -f "$BUSYLOG"
