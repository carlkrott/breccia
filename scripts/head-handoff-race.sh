#!/usr/bin/env bash
# head-handoff-race.sh — the acceptance gate for the head *placement*, and the
# regression test that keeps the device-crossing hand-off out of the tree.
#
# History: with the head on the other card (a residual hand-off every token), 2 of 16
# exclusive processes produced a wrong greedy sequence, and the perturbation was never
# localised even though the residual copy itself was proven byte-exact
# (`docs/phase-4/01-forward-pass.md` §8aw). The crossing path was therefore *removed*
# and `--head-device` may only name the device that owns the last block.
#
# So this script now asserts two things:
#
#   1. **The head's placement is where block 47 runs, and the tokens are stable.**
#      `--procs` processes x `--runs` generations must all generate the *same* sequence,
#      and (with `RACE_PROMPT=parity`) that sequence must be the reference's for the
#      first `--gen` tokens. This is the arm that used to be the control, and it is the
#      arm the engine still ships.
#   2. **A crossing request is refused, with a reason.** One process with the head
#      pinned to the other card must exit non-zero and say so — if it ever succeeds,
#      the removal regressed and this gate fails.
#
#   scripts/head-handoff-race.sh                        # 8 x 4, ~2 min, no arena
#   RACE_ARENA=auto RACE_PROMPT=parity RACE_PROCS=8 RACE_RUNS=1 RACE_GEN=32 \
#     RACE_WARM=1 scripts/head-handoff-race.sh          # the shipped configuration
#
# Environment: RACE_PROCS (8), RACE_RUNS (4), RACE_GEN (8), RACE_WARM (0),
# RACE_PROMPT (short|parity), RACE_ARENA (none|auto|<block list>),
# RACE_DEVICES, RACE_SPLIT, RACE_BIN, RACE_OUT.
#
# A co-tenant is not cosmetic: it shrinks the measured VRAM (so `auto` picks a
# different arena) and slows every stage ~3-15x. Every measured process must start and
# finish exclusive; a process a co-tenant appeared during is discarded, not counted.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

BIN=${RACE_BIN:-./target/release/breccia}
DEVICES=${RACE_DEVICES:-0000:03:00.0,0000:0e:00.0}
SPLIT=${RACE_SPLIT:-28,20}
PROCS=${RACE_PROCS:-8}
RUNS=${RACE_RUNS:-4}
GEN=${RACE_GEN:-8}
ARENA=${RACE_ARENA:-none}
WARM=${RACE_WARM:-0}
OUT=${RACE_OUT:-${SCRATCH:?SCRATCH must point at the scratch dir}/head-race}
PROMPT_MODE=${RACE_PROMPT:-short}
mkdir -p "$OUT"
rm -f "$OUT"/*.json "$OUT"/*.invalid.json

case "$PROMPT_MODE" in
  parity) PROMPT=$(cat "${SCRATCH:?}/p4-token/parity-server.json.ids") ;;
  *)      PROMPT="760,13017,1433" ;;
esac

# Which device owns the last block? A `--split-blocks` run is contiguous, so the last
# block's device is the one whose run ends at block 47.
LAST=0; acc=0
IFS=',' read -ra COUNTS <<<"$SPLIT"
for i in "${!COUNTS[@]}"; do acc=$((acc + COUNTS[i])); [ "$acc" -ge 48 ] && LAST=$i && break; done
IFS=',' read -ra DEVS <<<"$DEVICES"
CTRL=${DEVS[$LAST]}
CROSS=""
for i in "${!DEVS[@]}"; do [ "$i" != "$LAST" ] && CROSS=${DEVS[$i]} && break; done

vram_used() { cat "/sys/bus/pci/devices/$1/mem_info_vram_used" 2>/dev/null || echo 0; }
co_tenant() {
  [ "$(pgrep -c -x breccia)" != "0" ] && return 0
  for d in "${DEVS[@]}"; do [ "$(vram_used "$d")" -gt $((1536 << 20)) ] && return 0; done
  return 1
}
wait_quiet() { local n=$(( $1 / 2 )); for _ in $(seq 1 "$n"); do co_tenant || return 0; sleep 2; done; return 1; }

if ! wait_quiet 600; then
  echo "RACE: REFUSING — a co-tenant holds the cards (it mimics every symptom we chase); re-run when quiet"
  exit 3
fi
for f in /sys/bus/pci/devices/*/power_dpm_force_performance_level; do
  printf 'clocks %s = %s\n' "$(basename "$(dirname "$f")")" "$(cat "$f")"
done

arena_args=()
[ "$ARENA" != "none" ] && arena_args=(--gpu-experts "$ARENA")

echo "== arm 1: head on $CTRL (block $((acc - 1))'s device — the only placement allowed)"
for p in $(seq 1 "$PROCS"); do
  attempt=0 ok=0
  while [ "$attempt" -lt 4 ] && [ "$ok" = 0 ]; do
    attempt=$((attempt + 1))
    wait_quiet 300 || { echo "RACE: co-tenant will not clear; aborting"; exit 3; }
    timeout 900 "$BIN" bench token --devices "$DEVICES" --split-blocks "$SPLIT" \
      --head-device "$CTRL" "${arena_args[@]}" \
      --prompt-ids "$PROMPT" --gen "$GEN" --runs "$RUNS" --threads 16 --warm "$WARM" \
      --inbatch-read "${RACE_INBATCH:-true}" \
      --json "$OUT/ship-$p.json" >"$OUT/ship-$p.log" 2>&1
    rc=$?
    [ "$rc" -ne 0 ] && { echo "RACE: proc $p FAILED TO RUN (rc=$rc, see $OUT/ship-$p.log)"; exit 4; }
    if co_tenant; then
      echo "RACE: proc $p DISCARDED — a co-tenant appeared during it"
      mv "$OUT/ship-$p.json" "$OUT/ship-$p.invalid.json"
    else
      ok=1
    fi
  done
  [ "$ok" = 0 ] && { echo "RACE: proc $p could not be measured exclusively"; exit 4; }
done

echo "== arm 2: a crossing request must be REFUSED"
timeout 300 "$BIN" bench token --devices "$DEVICES" --split-blocks "$SPLIT" \
  --head-device "$CROSS" "${arena_args[@]}" --inbatch-read "${RACE_INBATCH:-true}" \
  --prompt-ids "$PROMPT" --gen 2 --runs 1 --threads 16 --warm 0 \
  >"$OUT/cross.log" 2>&1
cross_rc=$?

python3 - "$OUT" "$PROCS" "$RUNS" "$GEN" "$PROMPT_MODE" "$cross_rc" "$CROSS" <<'PY'
import json, os, sys
out, procs, runs, gen, mode, cross_rc, cross_pci = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5],
    int(sys.argv[6]), sys.argv[7])
bad = 0
rows = []
for p in range(1, procs + 1):
    with open(os.path.join(out, f"ship-{p}.json")) as f:
        rep = json.load(f)
    for r in rep["runs"]:
        rows.append((p, r.get("run", 0), list(r["generated"])))
base = rows[0][2]
for p, r, t in rows[1:]:
    if t != base:
        bad += 1
        n = min(len(t), len(base))
        first = next((i for i in range(n) if t[i] != base[i]), n)
        print(f"UNSTABLE proc {p} run {r}: step {first} ours {t[first] if first < len(t) else '-'} "
              f"expected {base[first] if first < len(base) else '-'}")
if mode == "parity":
    ref = json.load(open(os.environ["SCRATCH"] + "/p4-token/parity-server.json"))["generated"]
    n = min(gen, len(ref), len(base))
    bad_ref = [i for i in range(n) if base[i] != ref[i]]
    if bad_ref:
        bad += 1
        print(f"PARITY: step {bad_ref[0]} ours {base[bad_ref[0]]} ref {ref[bad_ref[0]]} "
              f"({n - len(bad_ref)}/{n})")
# arm 2: refused, and with the reason in the message.
log = open(os.path.join(out, "cross.log")).read()
refused = cross_rc != 0 and "device-crossing hand-off is removed" in log
if not refused:
    bad += 1
    print(f"CROSSING-NOT-REFUSED: rc={cross_rc} expected a non-zero exit and the removal "
          f"message for --head-device {cross_pci}")
print(f"RACE: {len(rows)} generations over {procs} exclusive processes, "
      f"crossing {'refused OK' if refused else 'NOT refused'} -> {'RED' if bad else 'GREEN'}")
sys.exit(1 if bad else 0)
PY
