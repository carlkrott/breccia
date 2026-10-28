#!/usr/bin/env bash
# session-reset-test.sh — the deterministic acceptance test for per-generation state.
#
# The field symptom was "sometimes a wrong first token", and the one correlate that
# survived every audit was the *warm* pass: a session that generates twice in one
# process. So the state that is not reset between generations is tested directly here,
# with no intermittency and no GPU-load luck:
#
#   A. the same prompt twice in one process  (`--runs 2`): the two runs must agree;
#   B. a **second prompt in the same process** (`--then-prompt-ids`) must generate the
#      same tokens as a *fresh process* given that prompt — with the second prompt
#      shorter, longer, and equal in length to the first.
#
# B is the case a missing reset breaks: stale paged-KV cells, stale pooled indexer
# keys and the stale indexer tail of the previous request all sit *past* the new
# request's length, which is exactly what a reused session leaves behind.
#
#   scripts/session-reset-test.sh                 # no arena, ~4 GiB load, ~2 min
#   RACE_ARENA=auto scripts/session-reset-test.sh # the shipped configuration
#
# Exit 0 only if every comparison matches. Prints the first divergence (step + ids).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

BIN=${RACE_BIN:-./target/release/breccia}
DEVICES=${RACE_DEVICES:-0000:03:00.0,0000:0e:00.0}
SPLIT=${RACE_SPLIT:-28,20}
GEN=${RESET_GEN:-12}
ARENA=${RACE_ARENA:-none}
OUT=${RESET_OUT:-${SCRATCH:?SCRATCH must point at the scratch dir}/session-reset}
mkdir -p "$OUT"

# Three prompt pairs: long→short, short→long, equal. The "long" prompt is the 27-id
# parity prompt; the short one is the 3-id prefix of it (a strict prefix, so a correct
# session must reproduce the fresh answer exactly).
LONG="760,13017,1433,364,279,5447,220,19,2463,11,12076,383,279,3603,6206,11,2640,279,17504,9039"
SHORT="760,13017,1433"
MID="760,13017,1433,364,279,5447"

arena_args=()
[ "$ARENA" != "none" ] && arena_args=(--gpu-experts "$ARENA")

if [ "$(pgrep -c -x breccia)" != "0" ]; then
  echo "RESET: REFUSING — another breccia process is running (a co-tenant invalidates a run)"
  exit 3
fi

fresh() { # $1 = tag, $2 = prompt -> writes $OUT/$1.json
  timeout 600 "$BIN" bench token --devices "$DEVICES" --split-blocks "$SPLIT" \
    "${arena_args[@]}" --prompt-ids "$2" --gen "$GEN" --runs 1 --threads 16 \
    --json "$OUT/$1.json" >"$OUT/$1.log" 2>&1 || { echo "RESET: $1 failed to run"; exit 4; }
}

reuse() { # $1 = tag, $2 = first prompt, $3 = second prompt
  timeout 600 "$BIN" bench token --devices "$DEVICES" --split-blocks "$SPLIT" \
    "${arena_args[@]}" --prompt-ids "$2" --then-prompt-ids "$3" --gen "$GEN" --runs 1 \
    --threads 16 --json "$OUT/$1.json" >"$OUT/$1.log" 2>&1 \
    || { echo "RESET: $1 failed to run"; exit 4; }
}

echo "== A: the same prompt twice in one process (short)"
timeout 600 "$BIN" bench token --devices "$DEVICES" --split-blocks "$SPLIT" \
  "${arena_args[@]}" --prompt-ids "$SHORT" --gen "$GEN" --runs 2 --threads 16 \
  --json "$OUT/twice.json" >"$OUT/twice.log" 2>&1 || { echo "RESET: twice failed"; exit 4; }

echo "== B1: long -> short"
fresh fresh-short "$SHORT"; reuse long-short "$LONG" "$SHORT"
echo "== B2: short -> long"
fresh fresh-long "$LONG"; reuse short-long "$SHORT" "$LONG"
echo "== B3: equal length (mid -> mid)"
fresh fresh-mid "$MID"; reuse mid-mid "$MID" "$MID"

# C: the *reason* the missing resets were harmless. Poison every mutable QSA buffer
# (K/V pools, indexer tail, pooled keys) with NaN after construction: if any kernel
# read a cell past the current position, the logits would be NaN. Identical output is
# the proof that the masking is what protects the sequence, not luck.
echo "== C: poisoned QSA state must not change a token"
BRECCIA_FAULT_POISON_QSA=1 timeout 600 "$BIN" bench token --devices "$DEVICES" \
  --split-blocks "$SPLIT" "${arena_args[@]}" --prompt-ids "$LONG" --gen "$GEN" \
  --runs 1 --threads 16 --json "$OUT/poison.json" >"$OUT/poison.log" 2>&1 \
  || { echo "RESET: poison failed"; exit 4; }

python3 - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
def gen(tag, key="generated"):
    d = json.load(open(os.path.join(out, tag + ".json")))
    if key == "generated":
        return d["runs"][0]["generated"]
    return d.get("then_generated", [])

bad = 0
def cmp(what, a, b):
    global bad
    if not a or not b:
        bad += 1
        print(f"{what}: EMPTY ({len(a)} vs {len(b)})")
        return
    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i] != b[i]), None)
    if first is None and len(a) == len(b):
        print(f"{what}: identical ({n} tokens)")
        return
    bad += 1
    i = first if first is not None else n
    print(f"{what}: DIVERGES at step {i}  reused={a[i] if i < len(a) else '-'} "
          f"fresh={b[i] if i < len(b) else '-'}")

d = json.load(open(os.path.join(out, "twice.json")))
runs = [r["generated"] for r in d["runs"]]
cmp("A  run0 vs run1 (same process)", runs[1], runs[0])
# The reused session's *second* generation must match a fresh process's first.
d2 = json.load(open(os.path.join(out, "long-short.json")))
cmp("B1 then vs fresh short", d2["then_generated"], gen("fresh-short"))
d2 = json.load(open(os.path.join(out, "short-long.json")))
cmp("B2 then vs fresh long", d2["then_generated"], gen("fresh-long"))
d2 = json.load(open(os.path.join(out, "mid-mid.json")))
cmp("B3 then vs fresh mid", d2["then_generated"], gen("fresh-mid"))
cmp("C  poisoned QSA vs clean long", gen("poison"), gen("fresh-long"))
print(f"RESET: {'RED' if bad else 'GREEN'} ({bad} mismatch(es))")
sys.exit(1 if bad else 0)
PY
