#!/bin/bash
# dev_run_impl.sh <taskdir> [--ws DIR --out DIR --id ID] [--baseline] [--correct] [VAR=VALUE ...] : the agent's only GPU entry point.
# --ws/--out/--id: used by remote_dev_entry.sh (trial mode with the solver off the pod): workspace to snapshot, agent-visible
# output dir, dev id. Defaults: $TD/workspace, $TD/workspace/dev_runs/<id>, devNNN-<timestamp>.
# 1 snapshot + gate (kernel_gate + locked files); 2 candidate run on the dev workload (official scoring load unless the agent
# overrides allowed workload vars); optional --baseline (parent run right after) and --correct (greedy-output check vs the
# parent reference). Agent-visible results: workspace/dev_runs/<id>/; harness log: $TD/logs/dev_runs.jsonl.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
# Run in a clean environment: nothing from the caller's shell (EXTRA_ENV, P2_PYTHONPATH, GPU, JOB_TIMEOUT, VLLM_*, ...) may
# reach the gate, the runs or the scorer. The GPU comes from task.env, or from $TD/trial_gpu (operator-owned, outside the
# workspace) when present.
if [ "${P2_DEV_CLEAN:-}" != 1 ]; then
  exec env -i P2_DEV_CLEAN=1 HOME="${HOME:-/tmp}" USER=coder LANG=C.UTF-8 ISO_ROOT="$ISO_ROOT" P2T="$P2T" HOST_PY_BIN="$HOST_PY_BIN" P2_CPUS_OVERRIDE="$P2_CPUS_OVERRIDE" CPUS="$CPUS" PORT="$PORT" UV="$UV" \
    PATH=$HOST_PY_BIN:/usr/local/bin:/usr/bin:/bin bash "$0" "$@"
fi
TD=$(readlink -f "$1"); shift; H=$ISO_ROOT/harness_iso13365
source "$TD/task.env"
[ -f "$TD/trial_gpu" ] && GPU=$(tr -dc 0-9 < "$TD/trial_gpu")
BASE=0; CORR=0; WL=(); WS=$TD/workspace; V=""; ID=""
while [ $# -gt 0 ]; do x=$1; shift
  case "$x" in
    --ws) WS=$(readlink -f "$1"); shift ;; --out) V=$1; shift ;;
    --id) ID=$1; shift; [[ "$ID" =~ ^[A-Za-z0-9._-]{1,64}$ ]] || { echo "bad dev id"; exit 2; } ;;
    --baseline) BASE=1 ;; --correct) CORR=1 ;;
    *=*) k=${x%%=*}; v=${x#*=}
         if [[ " $DEV_ALLOWED_VARS " == *" $k "* ]] && [[ "$v" =~ ^[0-9.]+$|^inf$ ]]; then WL+=("$k=$v"); else echo "not allowed: $x (allowed: $DEV_ALLOWED_VARS, numeric values)"; exit 2; fi ;;
    *) echo "unknown arg $x"; exit 2 ;;
  esac
done
mkdir -p "$TD/logs" "$TD/staging"
N=$(( $(ls "$TD/staging" | wc -l) + 1 )); [ -n "$ID" ] || ID=$(printf 'dev%03d-%s' $N "$(date +%Y%m%d-%H%M%S)")
S=$TD/staging/$ID; [ -e "$S" ] && { echo "dev id $ID already used"; exit 2; }
[ -n "$V" ] || { mkdir -p "$WS/dev_runs"; V=$WS/dev_runs/$ID; }; mkdir -p "$V"
echo "queued" > "$V/status.txt"
bash $H/snapshot_gate.sh "$TD" "$WS" "$S"; G=$?
# agent-visible output is built only by agent_view.py (structured fields, fixed-format tails; leakage review 2026-11-23)
if [ $G -ne 0 ]; then
  printf '{"id":"%s","gate":"REJECT"}\n' "$ID" > "$S/summary_full.json"; python3 -I $H/agent_view.py "$S/summary_full.json" "$S" "$V"
  echo "gate_reject" > "$V/status.txt"; cat "$V/gate.txt" | tail -30
  printf '{"id":"%s","ts":"%s","gate":"REJECT","workload":"%s"}\n' "$ID" "$(date -Is)" "${WL[*]:-}" >> "$TD/logs/dev_runs.jsonl"; exit 1
fi
echo "running" > "$V/status.txt"
run() {  # role out mode
  ( flock -w 3600 9 || exit 3; bash $H/run_variant.sh "$TD" "$1" "$S/stage" "p2d_${TASK}_${ID}_$1_$3" "$2" "$3" ${TASK_DEV_ENV:-} "${WL[@]}" ) 9>"$P2T/locks/iso13365_gpu$GPU.lock"
}
mkdir -p $ISO_ROOT/locks
run cand "$S/cand" timed > "$S/cand.log" 2>&1; RC=$?
[ $BASE = 1 ] && { run parent "$S/parent" timed > "$S/parent.log" 2>&1; }
# --correct: the candidate's greedy outputs are compared with a parent run made right AFTER every candidate process has
# ended, not with calib/correct_ref.json: the reference files are sealed away during trials (vault.sh) and a candidate
# process never runs while a reference it could replay exists in its reach (parent outputs are bit-identical, see calib)
[ $CORR = 1 ] && { run cand "$S/correct" correct > "$S/correct.log" 2>&1; run parent "$S/parent_correct" correct > "$S/parent_correct.log" 2>&1; }
: "${CORRECT_MIN_EXACT:?}" "${CORRECT_TIE_NATS:?}"
export CORRECT_MIN_EXACT CORRECT_TIE_NATS GPU
python3 -I - "$TD" "$S" "$V" "$BASE" "$CORR" "$RC" "${WL[*]:-}" <<'PY'
import importlib.util, json, os, subprocess, sys
td, s, v, base, corr, rc, wl = sys.argv[1:8]
spec = importlib.util.spec_from_file_location("m", os.path.join(td, "metric.py")); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
def meta(d):
    try: return json.load(open(os.path.join(d, "job_meta.json")))
    except Exception: return {}
out = {"id": os.path.basename(s), "gpu": os.environ.get("GPU"), "workload_overrides": wl, "rc": int(rc), "candidate": M.metric(os.path.join(s, "cand")),
       "candidate_peak_mem_mib": meta(os.path.join(s, "cand")).get("peak_mem_mib"), "candidate_wall_s": meta(os.path.join(s, "cand")).get("wall_s")}
if base == "1":
    out["baseline"] = M.metric(os.path.join(s, "parent"))
    try: out["speedup_vs_baseline"] = out["baseline"]["primary"] / out["candidate"]["primary"]
    except Exception: pass
if corr == "1":
    env = dict(os.environ)
    ref = os.path.join(s, "parent_correct", "correct.json"); cj = os.path.join(s, "correct", "correct.json")
    if not os.path.exists(ref):
        out["correctness"] = {"pass": None, "error": "parent_reference_run_failed"}
    elif os.path.exists(cj):
        r = subprocess.run([sys.executable, "-I", os.path.join(td, "compare.py"), ref, cj, os.environ["CORRECT_MIN_EXACT"], os.environ["CORRECT_TIE_NATS"]], capture_output=True, text=True)
        out["correctness"] = json.loads(r.stdout or "{}")
    else:
        out["correctness"] = {"pass": False, "error": "no_output"}
json.dump(out, open(os.path.join(s, "summary_full.json"), "w"), indent=1)
log = dict(out, ts=__import__("datetime").datetime.now().isoformat(), gate="PASS")
open(os.path.join(td, "logs", "dev_runs.jsonl"), "a").write(json.dumps(log) + "\n")
PY
python3 -I $H/agent_view.py "$S/summary_full.json" "$S" "$V" && cat "$V/summary.json"
echo "done" > "$V/status.txt"
