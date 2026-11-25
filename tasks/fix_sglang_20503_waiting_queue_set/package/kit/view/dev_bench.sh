#!/bin/bash
# dev_bench.sh [--load cal|cor|smoke] [--reps N] [--venv DIR] [--out DIR] : time the scored load on your tree, in the
# same job environment the scoring harness uses (clean environment, CPU pinning, offline model, same compile caches
# layout, same engine ports), on GPU $GPU / CPUs $CPUS from site.env. Prints the scored metric of this one run.
#   --load cal    the scored load (default); cor = the correctness pass's prompts; smoke = the scored load, 1 rep
#   --reps N      override the number of reps (the scored load uses the value in loads/cal.env)
#   --venv DIR    time another venv (default: ./venv, the one you edit). Tip: `cp -a venv venv_base` before editing
#                 and time venv_base for your own before/after comparison.
#   --out DIR     where the run's files go (default: runs/<load>_<time>)
# The GPU must be idle (the run refuses to start otherwise). DEV_LOCK=<file>: hold that lock file during the run.
set -uo pipefail
V=$(cd "$(dirname "$(readlink -f "$0")")" && pwd -P)
LOAD=cal; REPS=""; VENV=$V/venv; OUT=""
while [ $# -gt 0 ]; do case $1 in
  --load) LOAD=$2; shift 2;; --reps) REPS=$2; shift 2;; --venv) VENV=$(readlink -f "$2"); shift 2;;
  --out) OUT=$2; shift 2;; *) sed -n '2,12p' "$0"; exit 1;; esac; done
[ -f "$V/loads/$LOAD.env" ] || { echo "no load $LOAD (have: $(ls "$V/loads" | sed 's/\.env//' | tr '\n' ' '))"; exit 1; }
[ -x "$VENV/bin/python3" ] || { echo "not a venv: $VENV"; exit 1; }
S=$(mktemp -d /tmp/dvb.XXXXXX) || exit 1; trap 'rm -rf "$S"' EXIT
ln -s "$V/st" "$S/s"                       # short path: engine IPC socket paths are limited to 107 characters
unset C5X_ENV_DONE C5Y_ENV_DONE
export ISO_ROOT=$V STAGE=$S/s
. "$V/env.sh"
role=agent; LABEL=dev; PR=task; tag=$(date +%H%M%S); extra=""
. "$V/harness/dev_env.sh"                  # C (compile caches), EXTRA_ENV, DEV_COLD, DEV_MKDIR, DEV_TIMEOUT
OUT=${OUT:-$V/runs/${LOAD}_$(date +%Y%m%d_%H%M%S)}; mkdir -p "$(dirname "$OUT")"; OUT=$(readlink -f "$(dirname "$OUT")")/$(basename "$OUT")
EF=$V/loads/$LOAD.env
if [ -n "$REPS" ]; then printf '. "%s"\nexport REPS=%d\n' "$EF" "$REPS" > "$S/load.env"; EF=$S/load.env; fi
[ "$DEV_COLD" = 1 ] && rm -rf "$C"; eval "$DEV_MKDIR"
if [ -n "${DEV_LOCK:-}" ]; then exec 7>"$DEV_LOCK"; flock 7; fi
echo "dev_bench: load $LOAD${REPS:+ (reps $REPS)}, venv $VENV, GPU $GPU, CPUs $CPUS -> $OUT"
EXTRA_ENV="$EXTRA_ENV" JOB_TIMEOUT=$DEV_TIMEOUT WATCH_FRAC=97 \
  bash "$V/harness/job.sh" dev_$tag "$VENV" "$OUT" -- bash "$V/bench/run_bench.sh" "$EF" "$OUT" > "$OUT.joblog" 2>&1 7>&-
rc=$?
[ "$DEV_COLD" = 1 ] && rm -rf "$C"
"${SYSPY:-python3}" -I - "$OUT" "$rc" <<'PY'
import json, os, sys
o, rc = sys.argv[1], sys.argv[2]
m = {}
try: m = json.load(open(os.path.join(o, "job_meta.json")))
except Exception: pass
print(f"rc {rc}  wall {m.get('wall_s')} s  peak VRAM {m.get('peak_mem_mib')} MiB  aborted '{m.get('aborted', '')}'")
try:
    L = json.load(open(os.path.join(o, "latency.json")))
    print(f"scored metric (median rep, s): {L['metric_s']:.4f}   reps: {[round(x, 4) for x in L['per_rep_s']]}")
    keys = [k for k in ("startup_s", "out_tokens_per_rep", "out_tokens_turn1_per_rep") if L.get(k) is not None]
    for k in keys: print(f"  {k}: {L[k]}")
except Exception:
    fs = sorted(os.listdir(o)) if os.path.isdir(o) else []
    print(f"no latency.json (files: {fs}); see {o}/job_stdout.log")
PY
exit $rc
