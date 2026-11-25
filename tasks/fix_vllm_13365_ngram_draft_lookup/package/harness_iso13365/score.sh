#!/bin/bash
# score.sh <taskdir> <label> [workspace dir (default $TD/workspace)] [--transfer] : final scoring (operator only).
# 0 integrity: scorer / reference / master-venv manifests must match (start and end; manifest hashes go into the summary
#   so the workstation can compare them with its off-pod record)
# 1 snapshot + gate (kernel gate, locked files, P2 diff rules); 2 correctness vs the parent reference;
# 3 untimed warm-ups (parent, cand; other prompts) then SCORE_PAIRS ABBA pairs cand vs parent on the official load;
# 4 (--transfer) one AB pair per hidden transfer workload; 5 summary with REF from $TD/calib/calib.json.
# Verdict, most severe first: gate_reject > integrity_fail > audit_flag > timing_witness_fail > settings_mismatch >
#   output_mismatch > correctness_fail > above_ref / ties_ref / above_baseline_below_ref / no_gain. The speed tiers use a
#   fixed band of +-VERDICT_BAND (default 0.5 %, coordinator 2026-11-23: >= 3x the A/A deviations, covers the REF pairs'
#   spread); reward is continuous and does not depend on it.
# Output: $TD/results/<label>/{gate.txt, changes.patch, correct/, runs/, summary.json, summary.md}
# Two phases (2026-11-23, leakage review): this script runs the gate and every candidate / parent run and reads no
# reference file; harness/score_phase2.sh does the analysis. Normally phase 2 follows at once. While the task is sealed
# ($TD/SEALED, harness/vault.sh) the reference files are not on the pod: phase 1 stops with results/<label>/phase1.json
# and remote_score_finish.sh restores the references (key from the workstation), runs phase 2 and seals them again.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
TD=$(readlink -f "$1"); L=$2; WS=${3:-$TD/workspace}; TR=${4:-}; H=$ISO_ROOT/harness_iso13365
[ "$WS" = "--transfer" ] && { WS=$TD/workspace; TR=--transfer; }
source "$TD/task.env"
export OUT_MIN_EXACT OUT_TIE_NATS SETTINGS_IGNORE P2_AUDIT_NEED_END P2_AUDIT_IGNORE 2>/dev/null
: "${CORRECT_MIN_EXACT:?task.env must set CORRECT_MIN_EXACT}" "${CORRECT_TIE_NATS:?task.env must set CORRECT_TIE_NATS}"
# compile caches (staged tasks): a fresh root for this job; warm-ups fill it, every run reads its own copy; deleted at exit
if [ -n "${P2_CACHE_ROOT:-}" ]; then
  export P2_CACHE_MODE=job P2_CACHE_ROOT=$P2T/cache/$TASK/job-$L-$(date +%s)
  trap 'rm -rf "$P2_CACHE_ROOT"' EXIT
fi
R=$TD/results/$L; mkdir -p "$R"; LOG=$R/score.log
echo "$(date '+%F %T') SCORE START $L ws=$WS host=$(hostname) gpu=$GPU cpus=${P2_CPUS_OVERRIDE:-table}" >> $LOG
bash $H/integrity.sh verify "$TD" "score_${L}_pre" > "$R/integrity_pre.json" 2>>$LOG; IPRE=$?
bash $H/snapshot_gate.sh "$TD" "$WS" "$R/snap"; G=$?
cp "$R/snap/gate.txt" "$R/gate.txt" 2>/dev/null; cp "$R/snap/changes.patch" "$R/changes.patch" 2>/dev/null
if [ $G -ne 0 ]; then echo '{"verdict":"gate_reject"}' > "$R/summary.json"; printf '# score %s\n- verdict: gate_reject\n' "$R" > "$R/summary.md"; echo "gate reject"; exit 1; fi
if [ $IPRE -ne 0 ]; then echo '{"verdict":"integrity_fail","phase":"pre"}' > "$R/summary.json"; printf '# score %s\n- verdict: integrity_fail (pre)\n' "$R" > "$R/summary.md"; echo "integrity fail (pre)"; exit 1; fi
mkdir -p $P2T/locks
exec 9>"$P2T/locks/iso13365_gpu$GPU.lock"   # iso13365 copy: host-local lock (main and n1 share the NFS tree)
flock -w 7200 9 || { echo "lock timeout"; exit 3; }   # package (acceptance fix): this flock was commented out by accident
bash $H/run_variant.sh "$TD" cand "$R/snap/stage" "p2s_${TASK}_${L}_correct" "$R/correct" correct >> $LOG 2>&1
python3 -I $H/pairs.py "$TD" "$R/runs" sc --plan "ab:cand:parent:$SCORE_PAIRS" --warmup parent,cand --warmup-workload "${WARMUP_WORKLOAD:-}" --stage "$R/snap/stage" --workload "${SCORE_WORKLOAD:-}" >> $LOG 2>&1
if [ "$TR" = --transfer ]; then
  for tw in $TRANSFER_WORKLOADS; do
    n=${tw%%:*}; w=${tw#*:}; w=${w//,/ }
    python3 -I $H/pairs.py "$TD" "$R/runs" tr$n --plan "ab:cand:parent:1" --stage "$R/snap/stage" --workload "$w" >> $LOG 2>&1
  done
fi
bash $H/integrity.sh verify "$TD" "score_${L}_post" > "$R/integrity_post.json" 2>>$LOG; IPOST=$?
echo "{\"label\":\"$L\",\"ws\":\"$WS\",\"transfer\":\"$TR\",\"ipost\":$IPOST,\"time\":\"$(date -Is)\"}" > "$R/phase1.json"
echo "$(date '+%F %T') PHASE1 DONE $L" >> $LOG
if [ -e "$TD/SEALED" ]; then
  echo '{"verdict":"awaiting_refs"}' > "$R/summary.json"; printf '# score %s\n- run phase done; analysis waits for the sealed references (remote_score_finish.sh)\n' "$R" > "$R/summary.md"
  echo "PHASE1 DONE $L (sealed: run remote_score_finish.sh $TASK $L with the vault key on stdin)"; exit 0
fi
exec bash $H/score_phase2.sh "$TD" "$L"
