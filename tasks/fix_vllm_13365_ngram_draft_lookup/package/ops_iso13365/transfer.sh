#!/bin/bash
# transfer.sh <label> [gpu=1] [n_pairs=2] : human fix vs parent on each hidden transfer load (task.env TRANSFER_WORKLOADS),
# job-mode caches (fresh root, filled by one untimed warm-up pair on other prompts), ABBA pairs, then summarize.py.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
source $ISO_ROOT/ops_iso13365/common.sh
L=$1; G=${2:-1}; NP=${3:-2}; stop_check "transfer $L"; gpu_free $G || exit 3
source $TD/task.env; export GPU=$G SETTINGS_IGNORE P2_AUDIT_NEED_END P2_AUDIT_IGNORE   # summarize.py reads these
export P2_CACHE_MODE=job P2_CACHE_ROOT=$P2T/cache/iso13365/job-$L-$(date +%s); trap 'rm -rf "$P2_CACHE_ROOT"' EXIT
R=$TD/results/$L; mkdir -p $R/runs
log "transfer $L start $(hostctx)"
first=1
for tw in $TRANSFER_WORKLOADS; do
  n=${tw%%:*}; w=${tw#*:}; w=${w//,/ }
  stop_check "transfer $L $n"
  WU=(); [ $first = 1 ] && WU=(--warmup parent,fix --warmup-workload "$WARMUP_WORKLOAD"); first=0
  python3 -I $H/pairs.py $TD $R/runs tr$n --plan "ab:fix:parent:$NP" "${WU[@]}" --workload "$w" >> $R/pairs.log 2>&1
  log "transfer $L $n done"
done
python3 -I $H/summarize.py $TD $R/runs > $R/runs_summary.json
python3 -I -c "
import json; s=json.load(open('$R/runs_summary.json'))
for k,v in s['summary'].items(): print(k, v['n'], round(v['median_speedup'],4), v['values'], v['flagged_pairs'])
for p in s['pairs']: print(p['label'], p['pair'], round(p['m_first'],3), round(p['m_second'],3), (p.get('outcheck') or {}).get('pass'), (p.get('outcheck') or {}).get('accepted_timed'), p['flags'])
" > $R/summary.txt 2>&1
log "transfer $L end $(hostctx) | $(head -5 $R/summary.txt | tr '\n' ' ')"
