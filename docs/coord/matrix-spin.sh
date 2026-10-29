#!/bin/bash
# 2026-10-29 Phase 5: Phase 1 spin patch (expert.rs.p1) re-tested at x4b, where the pool is contended.
D=/var/tmp/breccia-coord; BIN=$D/breccia-p5spin
M=0000:0e:00.0; R=0000:03:00.0
C=(--prefill-chunk 64 --ctx 320 --gen 160 --runs 3 --threads 30 --prompt-ids-file $D/prompt-64.ids)
CC=--setenv=BRECCIA_MULTI_DECODE_CONCURRENT=1; SH=--setenv=BRECCIA_SHARED_DEVICE=1; MX=--setenv=BRECCIA_BENCH_MIXED_PROMPTS=1; LS=--setenv=BRECCIA_LANE_STATS=1
A=0,1,2,4,8,18,20,22,23; B=$A,3,5,6,7
export MEMMAX=44G
for n in 0 50 200; do
  $D/run.sh sp$n-x4b $BIN $CC $SH $MX $LS --setenv=BRECCIA_LANE_SPIN_US=$n -- --streams 4 --devices $R,$M,$R,$R --gpu-experts "$A;$B;$A" "${C[@]}"
done
echo "matrix-spin done"
