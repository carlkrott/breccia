#!/bin/bash
# 2026-10-29 (Carl OK): host-import A/B on x1r (memlock raised), then scx_lavd on/off/on at x1r, x3r9, x4b (mixed).
D=/var/tmp/breccia-coord; BIN=$D/breccia-mx
M=0000:0e:00.0; R=0000:03:00.0
C=(--prefill-chunk 64 --ctx 320 --gen 160 --runs 3 --threads 30 --prompt-ids-file $D/prompt-64.ids)
CC=--setenv=BRECCIA_MULTI_DECODE_CONCURRENT=1; SH=--setenv=BRECCIA_SHARED_DEVICE=1; MX=--setenv=BRECCIA_BENCH_MIXED_PROMPTS=1
LS=--setenv=BRECCIA_LANE_STATS=1
A=0,1,2,4,8,18,20,22,23; B=$A,3,5,6,7
trap 'sudo -n systemctl restart scx_loader; sleep 3; echo "restored: $(scxctl get)"' EXIT
$D/run.sh hi0-x1r $BIN $LS -- --streams 1 --devices $R "${C[@]}"
$D/run.sh hi1-x1r $BIN $LS -- --streams 1 --devices $R --host-expert-import "${C[@]}"
arms() {
  echo "== $1 sched: $(scxctl get) state=$(cat /sys/kernel/sched_ext/state)"
  $D/run.sh $1-x1r $BIN -- --streams 1 --devices $R "${C[@]}"
  MEMMAX=44G $D/run.sh $1-x3r9 $BIN $CC $SH $MX -- --streams 3 --devices $R,$M --gpu-experts $A "${C[@]}"
  MEMMAX=44G $D/run.sh $1-x4b $BIN $CC $SH $MX -- --streams 4 --devices $R,$M,$R,$R --gpu-experts "$A;$B;$A" "${C[@]}"
}
arms lv1
sudo -n scxctl stop; sleep 3
arms eev
sudo -n systemctl restart scx_loader; sleep 5
arms lv2
echo "matrix-hiscx done"
