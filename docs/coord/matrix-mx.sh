#!/bin/bash
# 2026-10-29 mixed prompts (BRECCIA_BENCH_MIXED_PROMPTS=1) + single-stream route traces per prompt (2b premise).
D=/var/tmp/breccia-coord; BIN=$D/breccia-mx
M=0000:0e:00.0; R=0000:03:00.0
C=(--prefill-chunk 64 --ctx 320 --gen 160 --runs 3 --threads 30 --prompt-ids-file $D/prompt-64.ids)
CC=--setenv=BRECCIA_MULTI_DECODE_CONCURRENT=1; SH=--setenv=BRECCIA_SHARED_DEVICE=1; MX=--setenv=BRECCIA_BENCH_MIXED_PROMPTS=1
A=0,1,2,4,8,18,20,22,23; B=$A,3,5,6,7
export MEMMAX=44G
$D/run.sh mx-x4b $BIN $CC $SH $MX -- --streams 4 --devices $R,$M,$R,$R --gpu-experts "$A;$B;$A" "${C[@]}"
$D/run.sh mx-x3r9 $BIN $CC $SH $MX -- --streams 3 --devices $R,$M --gpu-experts $A "${C[@]}"
for s in 0 1 2 3; do
  f=$D/prompt-64.ids; [ $s -gt 0 ] && f=$D/prompt-64-s$s.ids
  rm -f $D/out/route-s$s.txt
  $D/run.sh rt-s$s $BIN --setenv=BRECCIA_TRACE_ROUTE_FILE=$D/out/route-s$s.txt -- --streams 1 --devices $R --prefill-chunk 64 --ctx 320 --gen 160 --runs 1 --threads 30 --prompt-ids-file $f
done
echo "matrix-mx done"
