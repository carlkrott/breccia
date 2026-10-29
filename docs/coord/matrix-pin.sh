#!/bin/bash
# 2026-10-29 Phase 5 premise: live x4b mixed prompts with BRECCIA_CPU_PIN (vs mx-x4b 47.70 unpinned).
D=/var/tmp/breccia-coord; BIN=$D/breccia-mx
M=0000:0e:00.0; R=0000:03:00.0
C=(--prefill-chunk 64 --ctx 320 --gen 160 --runs 3 --prompt-ids-file $D/prompt-64.ids)
CC=--setenv=BRECCIA_MULTI_DECODE_CONCURRENT=1; SH=--setenv=BRECCIA_SHARED_DEVICE=1; MX=--setenv=BRECCIA_BENCH_MIXED_PROMPTS=1
A=0,1,2,4,8,18,20,22,23; B=$A,3,5,6,7
export MEMMAX=44G
X=(--streams 4 --devices $R,$M,$R,$R --gpu-experts "$A;$B;$A")
$D/run.sh pn30-x4b $BIN $CC $SH $MX --setenv=BRECCIA_CPU_PIN=0-29/30,31 -- "${X[@]}" --threads 30 "${C[@]}"
$D/run.sh pn28-x4b $BIN $CC $SH $MX --setenv=BRECCIA_CPU_PIN=1-15,17-29/0,16,30,31 -- "${X[@]}" --threads 28 "${C[@]}"
$D/run.sh un28-x4b $BIN $CC $SH $MX -- "${X[@]}" --threads 28 "${C[@]}"
echo "matrix-pin done"
