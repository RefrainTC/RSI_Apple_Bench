#!/bin/bash
# aa.sh <label> [gpu=1] [n_pairs=3] : extra A/A (parent vs parent) pairs under scoring conditions (job-mode caches, one
# untimed warm-up), on whichever host runs it. Results: results/<label>/summary.txt.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
source $ISO_ROOT/ops_iso13365/common.sh
L=$1; G=${2:-1}; NP=${3:-3}; stop_check "aa $L"; gpu_free $G || exit 3
source $TD/task.env; export GPU=$G SETTINGS_IGNORE P2_AUDIT_NEED_END P2_AUDIT_IGNORE   # summarize.py reads these
export P2_CACHE_MODE=job P2_CACHE_ROOT=$P2T/cache/iso13365/job-$L-$(date +%s); trap 'rm -rf "$P2_CACHE_ROOT"' EXIT
R=$TD/results/$L; mkdir -p $R/runs
log "aa $L start $(hostctx)"
python3 -I $H/pairs.py $TD $R/runs aa --plan "aa:parent:parent:$NP" --warmup parent --warmup-workload "$WARMUP_WORKLOAD" >> $R/pairs.log 2>&1
python3 -I $H/summarize.py $TD $R/runs > $R/runs_summary.json
python3 -I -c "
import json; s=json.load(open('$R/runs_summary.json'))
for k,v in s['summary'].items(): print(k, v['n'], round(v['median_speedup'],4), v['values'], v.get('max_abs_dev'), v['flagged_pairs'])
for p in s['pairs']: print(p['label'], p['pair'], round(p['m_first'],4), round(p['m_second'],4), (p.get('outcheck') or {}).get('pass'), p['flags'])
" > $R/summary.txt 2>&1
log "aa $L end $(hostctx) | $(head -1 $R/summary.txt)"
