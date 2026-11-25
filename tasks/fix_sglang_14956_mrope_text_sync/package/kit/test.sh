#!/bin/bash
# test.sh <agent_tree | patch.diff> [name] : score an agent's engine edits on this machine (Harbor-style verifier,
# host mode). Writes {"reward", "speedup", "verdict", "checks"} to ${REWARD_FILE:-/logs/verifier/reward.json}.
#   1. refuses unless harness/gate.json has a gpu_name equal to this GPU's (kit/set_ref.py writes it);
#   2. takes $KIT_LOCK (if set; a machine-wide GPU lock file) for everything below;
#   3. checks the parent engine against the setup's hash record, the package files against MANIFEST.sha256 (gate.json
#      against set_ref.py's record), and the trusted trees (parent venv, interpreter, model, CUDA) against the kit seal
#      (made here on first use, see lib/integrity.sh);
#   4. kit/make_candidate.sh <name> <agent tree>;
#   5. harness/run.sh <fresh label> score <name>, with lib/monitor.py watching the trusted trees, the finished runs'
#      results and leftover candidate processes between runs;
#   6. re-checks the parent engine, the seal and the package files.
# Reward = harness/check.py's reward if every check passes, else 0 (a failed gate, a refusal or any error also give 0).
# Exit 0 when the job was scored (whatever the reward), 3 on a refusal, 1 on an error.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/lib/common.sh"; . "$KIT/lib/integrity.sh"
SRC=${1:-}; NAME=${2:-c$(date -u +%m%d%H%M%S)}
RF=${REWARD_FILE:-/logs/verifier/reward.json}
CHK=$(mktemp "${TMPDIR:-/tmp}/kit_checks.XXXXXX"); trap 'rm -f "$CHK"' EXIT
LABEL=""; SPEED=null; VERDICT=error
chk() { printf '%s\t%s\t%s\n' "$1" "$2" "${3:-}" >> "$CHK"; kit_log "check $1: $2 ${3:-}"; }
finish() {  # <exit code>  writes the reward file from $CHK and results/<label>/verdict.json
  mkdir -p "$(dirname "$RF")" 2>/dev/null
  "$SYSPY" -I "$KIT/lib/finish.py" "$CHK" "$RF" "$ISO_ROOT" "$LABEL" "$NAME" "$VERDICT" || {
    echo '{"reward": 0.0, "speedup": null, "verdict": "error:finish", "checks": {}}' > "$RF"; }
  cat "$RF"; exit "$1"
}
[ -n "$SRC" ] || { echo "usage: test.sh <agent engine tree | venv | patch.diff> [name]"; exit 1; }
# 1. GPU match
want=$("$SYSPY" -I -c "import json,sys; print(json.load(open(sys.argv[1])).get('gpu_name') or '')" "$ISO_ROOT/harness/gate.json" 2>/dev/null)
have=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU" 2>/dev/null | head -1)
if [ -z "$want" ]; then
  VERDICT=refused:uncalibrated; chk gpu_match fail "harness/gate.json has no gpu_name: REF was not set on this GPU type. Calibrate here and run kit/set_ref.py first"
  echo "REFUSED: harness/gate.json has no gpu_name, so its REF was not measured on this GPU type ($have). Run the calibration and kit/set_ref.py first." >&2; finish 3
elif [ "$want" != "$have" ]; then
  VERDICT=refused:gpu_mismatch; chk gpu_match fail "gate.json is for '$want', this GPU ($GPU) is '$have'"
  echo "REFUSED: harness/gate.json was calibrated on '$want' but GPU $GPU is '$have'. Speedups do not transfer between GPU types; calibrate on this GPU and run kit/set_ref.py." >&2; finish 3
fi
chk gpu_match pass "$have"
# 2. quick pre-checks (refuse before waiting for the GPU)
pre() {  # <name> <function>
  local r; if r=$($2); then chk "$1" pass "$r"; else chk "$1" fail "$r"; VERDICT=refused:tampered; echo "REFUSED: $r" >&2; finish 3; fi
}
pre parent_engine_pre verify_parent_engine
pre package_files_pre verify_package_files
# 3. machine lock, then the full checks (nothing else times on the GPU while they hash ~15 GB)
if [ -n "${KIT_LOCK:-}" ]; then exec 8>"$KIT_LOCK"; kit_log "waiting for $KIT_LOCK"; flock 8; kit_log "holding $KIT_LOCK"; fi
pre parent_engine_locked verify_parent_engine
if [ ! -f "$KSTATE/seal.sha256" ]; then r=$(seal_parent) || { chk seal_create fail "$r"; VERDICT=refused:tampered; finish 3; }; chk seal_create pass "$r"; fi
pre seal_pre verify_seal
# 4. candidate
if r=$(bash "$KIT/make_candidate.sh" "$NAME" "$SRC" 2>&1); then chk candidate_built pass "$(echo "$r" | tail -1)"
else chk candidate_built fail "$(echo "$r" | grep 'KIT ERROR' | tail -1)"; VERDICT=error:candidate; finish 1; fi
# 5. scoring job with the monitor
LABEL=kit_${NAME}_$(date -u +%Y%m%d%H%M%S); R=$ISO_ROOT/results/$LABEL
[ -e "$R" ] && { chk label fail "results/$LABEL exists"; finish 1; }
mkdir -p "$KSTATE/mon"; MON=$KSTATE/mon/$LABEL; rm -f "$MON".*
AUX=$("$SYSPY" -I -c "
import sys
def ex(s):
    o=set()
    for p in s.split(','):
        a,_,b=p.partition('-'); o.update(range(int(a),int(b or a)+1))
    return o
n=int(sys.argv[2]); a=sorted(set(range(n))-ex(sys.argv[1])) or list(range(n))
print(','.join(map(str,a)))" "$CPUS" "$(nproc --all)")
taskset -c "$AUX" nice -n 10 "$SYSPY" -I "$KIT/lib/monitor.py" "$R" "$STAGE/venvs/${PR}_$NAME" "$STAGE/tmp" "$MON.json" "$MON.ready" "$MON.stop" \
  $(trusted_trees) > "$MON.log" 2>&1 & MPID=$!
for i in $(seq 1 600); do [ -f "$MON.ready" ] && break; kill -0 $MPID 2>/dev/null || break; sleep 0.5; done
[ -f "$MON.ready" ] || { chk monitor fail "monitor did not start ($(tail -1 "$MON.log"))"; finish 1; }
kit_log "scoring: harness/run.sh $LABEL score $NAME (results/$LABEL)"
bash "$ISO_ROOT/harness/run.sh" "$LABEL" score "$NAME" 8>&-; rc=$?
touch "$MON.stop"; wait $MPID
[ $rc = 0 ] && chk harness_run pass "rc 0" || chk harness_run fail "harness/run.sh rc $rc (see results/$LABEL/run.log)"
[ -s "$MON.json" ] || chk monitor fail "monitor wrote no report ($(tail -2 "$MON.log" | tr '\n' ' '))"
[ -s "$MON.json" ] && "$SYSPY" -I - "$MON.json" "$CHK" <<'PY'
import json, sys
m = json.load(open(sys.argv[1])); f = open(sys.argv[2], "a")
v = m["violations"]
def line(k, ok, d): f.write(f"{k}\t{'pass' if ok else 'fail'}\t{d}\n")
tv = [x for x in v if x["check"] == "trusted_trees"]
line("trusted_trees_between_runs", not tv, f"{m['baseline_files']} files watched after each of {len(m['events'])} runs" + (f"; {tv[0]}" if tv else ""))
rv = [x for x in v if x["check"] == "results_frozen"]
line("results_frozen", not rv, f"{len(m['frozen'])} finished run dirs unchanged until the end" + (f"; {rv[0]}" if rv else ""))
lv = [x for x in v if x["check"] == "leftover_process"]
line("no_leftover_process", not lv, "no candidate process outlived its run" if not lv else str(lv[0])[:300])
ev = [x for x in v if x["check"] == "monitor_error"]
line("monitor", not ev, "ran to the end" if not ev else str(ev[0])[:300])
PY
# 6. post-checks
post() { local r; if r=$($2); then chk "$1" pass "$r"; else chk "$1" fail "$r"; fi; }
post parent_engine_post verify_parent_engine
post seal_post verify_seal
post package_files_post verify_package_files
finish 0
