#!/usr/bin/env bash
# head-race-capture.sh — run the stability gate and, if a process diverges, capture that
# process **with every instrument on** so a recurrence is data instead of another hunt.
#
# The step-0 wrong-sequence family is parked (docs/phase-4/01-forward-pass.md §8bc/§8bd):
# 96 generations over 48 exclusive processes clean, 95 % upper bound 6.1 %/process, and
# every state/ordering/coverage instrument clean. The one thing that would settle it is a
# *failing process with its instruments already switched on*, and that is what this does:
#
#   1. run `scripts/head-handoff-race.sh` (same env knobs, same quiet-window guard);
#   2. if it reports RED, re-run the failing process indices — and the first clean one as
#      the control — with
#        BRECCIA_DUMP_LAYOUT=1  BRECCIA_VERIFY_WEIGHTS=all
#        BRECCIA_FAULT_POISON_LOGITS=1  --dump-logits
#      into $SCRATCH/head-capture/<UTC timestamp>/
#      (failing/ and control/ side by side, plus a README with the commands);
#   3. exit with the gate's status.
#
# Exercised with `BRECCIA_FAULT_DIVERGE_STEP=4` (a deliberate divergence at forward 4), so
# the capture path is known to work before it is needed:
#
#   BRECCIA_FAULT_DIVERGE_STEP=4 RACE_PROCS=2 RACE_RUNS=1 RACE_GEN=8 \
#     scripts/head-race-capture.sh
#
# Environment: everything `head-handoff-race.sh` accepts (RACE_*), plus CAPTURE_ROOT.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

CAPTURE_ROOT=${CAPTURE_ROOT:-${SCRATCH:?SCRATCH must point at the scratch dir}/head-capture}
BIN=${RACE_BIN:-./target/release/breccia}
DEVICES=${RACE_DEVICES:-0000:03:00.0,0000:0e:00.0}
SPLIT=${RACE_SPLIT:-28,20}
GEN=${RACE_GEN:-8}
RUNS=${RACE_RUNS:-4}
WARM=${RACE_WARM:-0}
PROCS=${RACE_PROCS:-8}
ARENA=${RACE_ARENA:-none}
PROMPT_MODE=${RACE_PROMPT:-short}
INBATCH=${RACE_INBATCH:-true}

case "$PROMPT_MODE" in
  parity) PROMPT=$(cat "${SCRATCH:?}/p4-token/parity-server.json.ids") ;;
  *)      PROMPT="760,13017,1433" ;;
esac
arena_args=()
[ "$ARENA" != "none" ] && arena_args=(--gpu-experts "$ARENA")

stage=$(mktemp -d "${TMPDIR:-/tmp}/racecap.XXXXXX")
trap 'rm -rf "$stage"' EXIT
echo "== gate"
RACE_OUT="$stage" scripts/head-handoff-race.sh 2>&1 | tee "$stage/gate.log"
gate=${PIPESTATUS[0]}
echo "== gate exit $gate"

if [ "$gate" = 0 ]; then
  echo "capture: nothing to capture (gate green); artifacts would go to $CAPTURE_ROOT/<timestamp>/"
  exit 0
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
out="$CAPTURE_ROOT/$stamp"
mkdir -p "$out/failing" "$out/control"
echo "capture: writing to $out"

# Which processes diverged? Two signals: a process that disagrees with its neighbours
# (`UNSTABLE proc N`, which names it) or a window that disagrees with the *reference*
# (`PARITY:`, which names none — every process in it is then a capture candidate).
mapfile -t bad < <(grep -oE 'UNSTABLE proc [0-9]+' "$stage/gate.log" | awk '{print $3}' | sort -un)
if [ "${#bad[@]}" -eq 0 ] && grep -q '^PARITY:' "$stage/gate.log"; then
  mapfile -t bad < <(ls -1 "$stage"/ship-*.json 2>/dev/null | sed -E 's/.*ship-([0-9]+)\.json/\1/' | sort -un | head -4)
fi
echo "capture: diverged process(es): ${bad[*]:-<none derived>}"

capture_one() { # $1 = dir, $2 = proc index, $3 = extra env
  local dir=$1 p=$2 extra=$3
  env $extra BRECCIA_REQUIRE_QUIET=1 "$BIN" bench token \
    --devices "$DEVICES" --split-blocks "$SPLIT" "${arena_args[@]}" \
    --inbatch-read "$INBATCH" --prompt-ids "$PROMPT" --gen "$GEN" --runs "$RUNS" \
    --threads 16 --warm "$WARM" \
    --dump-logits "$dir/logits-proc$p.txt" --top-logits 32 \
    --json "$dir/proc$p.json" >"$dir/proc$p.log" 2>&1
  echo "  proc $p -> $dir/proc$p.log (exit $?)"
}

instruments="BRECCIA_DUMP_LAYOUT=1 BRECCIA_VERIFY_WEIGHTS=all BRECCIA_FAULT_POISON_LOGITS=1"
for p in "${bad[@]}"; do
  [ -z "$p" ] && continue
  capture_one "$out/failing" "$p" "$instruments"
done
capture_one "$out/control" 1 "$instruments"

cat >"$out/README" <<EOF
capture $stamp — a diverged process, re-run with every instrument on.

gate: RACE_PROCS=$PROCS RACE_RUNS=$RUNS RACE_GEN=$GEN RACE_WARM=$WARM RACE_PROMPT=$PROMPT_MODE
      RACE_ARENA=$ARENA RACE_INBATCH=$INBATCH devices=$DEVICES split=$SPLIT
diverged process(es): ${bad[*]:-<none derived>}

failing/procN.{json,log,logits-procN.txt}  the process that generated a different sequence
control/proc1.*                             the first clean process, same window

What to look at, in order (docs/phase-4/01-forward-pass.md §8bd):
 1. logits-proc*.txt  first 32 entries: is the row tilted or is one token moved?
 2. log               "layout (BRECCIA_DUMP_LAYOUT)" block: diff failing vs control sizes/handles
 3. log               "weights ... read back and compared": 0 divergences expected
 4. log               "head coverage OK" / "lane scratch coverage OK": 0 unwritten expected
 5. log               "readback N in-batch readback(s) of which 0 ..."
EOF
echo "capture: wrote $out"
ls -1 "$out/failing" "$out/control"
exit "$gate"
