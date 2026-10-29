#!/bin/bash
# 2026-10-29 Phase 5: combining pool (BRECCIA_POOL_COMBINE=1) gates vs mx-x4b / mx-x3r9 / g1off-x1r.
D=/var/tmp/breccia-coord; BIN=$D/breccia-cmb
M=0000:0e:00.0; R=0000:03:00.0
C=(--prefill-chunk 64 --ctx 320 --gen 160 --runs 3 --threads 30 --prompt-ids-file $D/prompt-64.ids)
CC=--setenv=BRECCIA_MULTI_DECODE_CONCURRENT=1; SH=--setenv=BRECCIA_SHARED_DEVICE=1; MX=--setenv=BRECCIA_BENCH_MIXED_PROMPTS=1; LS=--setenv=BRECCIA_LANE_STATS=1
CB=--setenv=BRECCIA_POOL_COMBINE=1
A=0,1,2,4,8,18,20,22,23; B=$A,3,5,6,7
export MEMMAX=44G
$D/run.sh cmb-x4b $BIN $CC $SH $MX $CB -- --streams 4 --devices $R,$M,$R,$R --gpu-experts "$A;$B;$A" "${C[@]}"
$D/run.sh cmbls-x4b $BIN $CC $SH $MX $CB $LS -- --streams 4 --devices $R,$M,$R,$R --gpu-experts "$A;$B;$A" "${C[@]}"
$D/run.sh cmboff-x4b $BIN $CC $SH $MX -- --streams 4 --devices $R,$M,$R,$R --gpu-experts "$A;$B;$A" "${C[@]}"
$D/run.sh cmb-x3r9 $BIN $CC $SH $MX $CB -- --streams 3 --devices $R,$M --gpu-experts $A "${C[@]}"
$D/run.sh cmb-x1r $BIN $CB -- --streams 1 --devices $R "${C[@]}"
echo "matrix-cmb done"
