#!/bin/bash
# final_report.sh : everything the card needs, as text (ops_iso13365/final_report.txt).
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
P2=$ISO_ROOT; TD=$P2/tasks/iso13365; O=$P2/ops_iso13365
{
echo "== calibration"; python3 -I -c "
import json; c=json.load(open('$TD/calib/calib.json'))
print('REF', c['ref_speedup'], 'fix pairs', c['fix_pairs']['values'], 'AA', c['aa']['values'], 'noise_band', c['noise_band'])
print('xv', (c.get('overlay_equivalence_fixvenv_vs_parentvenv') or {}).get('values'), 'witness', c.get('witness'), 'parent_primary', c.get('parent_primary'))
print('flags', c.get('calibration_flags'), 'audit', c.get('audit_events_trusted_runs'))"
python3 -I $O/calib_report.py $TD
echo "== scoring jobs"; python3 -I $O/jobs_summary.py $TD $(cd $TD/results && ls -d m? n1? 2>/dev/null)
for l in $(cd $TD/results && ls -d m? n1? 2>/dev/null); do echo "-- $l"; cat $TD/results/$l/summary.md; done
echo "== transfer"; cat $TD/results/tr1/summary.txt 2>/dev/null
echo "== aa"; for d in $TD/results/aa_*; do echo "-- $d"; cat $d/summary.txt; done
echo "== canaries"; for l in C1_nopropose C2_k3 C3_window C4_window_rt; do echo "-- $l"; cat $TD/results/$l/summary.md 2>/dev/null; python3 -I -c "
import json; s=json.load(open('$TD/results/$l/summary.json')); c=s.get('correctness') or {}
print('correct pass', c.get('pass'), 'unit', c.get('unit'), {g:(v.get('ok'), v.get('core_diff') and list(v['core_diff'])) for g,v in (c.get('groups') or {}).items()})
for p in s.get('pair_checks') or []: print('pair', p['pair'], p['flags'], (p.get('outcheck') or {}).get('core_diff'), (p.get('outcheck') or {}).get('bad_sample'), p.get('settings_diff'))
" 2>&1 | cut -c1-900; done
echo "== microbench"; cat $O/microbench_n1.json
} > $O/final_report.txt 2>&1
wc -c $O/final_report.txt
