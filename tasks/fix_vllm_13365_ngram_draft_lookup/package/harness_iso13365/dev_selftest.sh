#!/bin/bash
# dev_selftest.sh <taskdir> : exercise the agent-facing dev_run.sh exactly as an agent would (from inside the workspace),
# then move everything it produced (workspace/dev_runs, staging, logs/dev_runs.jsonl) to $TD/selftest/<ts>/ so the
# workspace is pristine again. Checks: a disallowed workload var is refused (rc 2); an injected env var does not reach
# the run; --baseline --correct on the pristine workspace gives speedup ~1 and passes correctness.
set -uo pipefail
TD=$(readlink -f "$1"); WS=$TD/workspace; OUT=$TD/selftest/$(date +%Y%m%d-%H%M%S); mkdir -p "$OUT"
for _w in $(seq 1 120); do [ -f "$TD/calib/correct_ref.json" ] && break; sleep 60; done   # wait for a (re)calibration
cd "$WS" || exit 1
bash dev_run.sh NOT_A_VAR=1 > "$OUT/disallowed.txt" 2>&1; echo "disallowed rc=$? (expect 2)" | tee -a "$OUT/result.txt"
EXTRA_ENV="P2_SELFTEST_INJECTED=1" P2_PYTHONPATH=/nonexistent JOB_TIMEOUT=1 bash dev_run.sh --baseline --correct > "$OUT/devrun.txt" 2>&1
echo "devrun rc=$?" | tee -a "$OUT/result.txt"
id=$(ls -t "$WS/dev_runs" | head -1)
python3 -I - "$WS/dev_runs/$id/summary.json" "$TD/staging/$id" <<'PY' | tee -a "$OUT/result.txt"
import json, sys, glob
s = json.load(open(sys.argv[1])); st = sys.argv[2]
inj = [f for f in glob.glob(st + "/*/job_meta.json") if "P2_SELFTEST_INJECTED" in open(f).read() or "nonexistent" in open(f).read()]
print("gpu", s.get("gpu"), "rc", s["rc"], "speedup_vs_baseline", round(s.get("speedup_vs_baseline", 0), 4),
      "correct", s.get("correctness", {}).get("pass"), "wall", s.get("candidate_wall_s"), "env_leak", bool(inj))
PY
mv "$WS/dev_runs" "$OUT/ws_dev_runs"; mv "$TD/staging" "$OUT/staging"; mv "$TD/logs/dev_runs.jsonl" "$OUT/" 2>/dev/null
(cd "$WS" && git status --porcelain | head -5) > "$OUT/git_status.txt"; echo "workspace git status lines: $(wc -l < "$OUT/git_status.txt") (expect 0)" | tee -a "$OUT/result.txt"
echo "DEV_SELFTEST DONE $OUT"
