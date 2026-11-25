#!/bin/bash
# canaries16436.sh : #16436 readiness batch on GPU0 (after the held remote re-score). Installs upload/v4d (outcheck.py:
# TTFT guard), re-makes the #16436 manifests, scores the human fix again (must pass the guard) and canaries C1-C7g.
# Report: smoke/canaries16436.md
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
P2=$ISO_ROOT; H=$P2/harness; TD=$P2/tasks/iso16436; L=$P2/logs/install_v3.log
cd $P2/upload/v4d && for f in $(find . -type f | sed 's#^\./##'); do cp "$f" "$P2/$f.new" && mv "$P2/$f.new" "$P2/$f" && echo "$(date '+%F %T') installed v4d $f" >> $L; done
bash $H/integrity.sh make $TD > $TD/manifest/manifest_hashes.txt 2>&1; echo "$(date '+%F %T') v4d manifests iso16436: $(tr '\n' ' ' < $TD/manifest/manifest_hashes.txt)" >> $L
O=$P2/smoke/canaries16436.md; echo "# #16436 readiness: human fix + canaries ($(date -u '+%F %T') UTC)" > $O
bash $P2/scripts/make_canaries16436.sh $P2/smoke >> $O.log 2>&1 || echo "canary build failed" >> $O
echo "| score | verdict | speedup | flags | integrity pre/post | outcheck fails | ttft ratios | audit kinds |" >> $O; echo "|---|---|---|---|---|---|---|---|" >> $O
for t in e2e_fix5:$P2/smoke/rtest_iso16436/fix v16436_C7g:$P2/smoke/v16436_C7g v16436_C1:$P2/smoke/v16436_C1 v16436_C2:$P2/smoke/v16436_C2 v16436_C3w:$P2/smoke/v16436_C3w v16436_C4r:$P2/smoke/v16436_C4r v16436_C6:$P2/smoke/v16436_C6 v16436_C7:$P2/smoke/v16436_C7; do
  Lb=${t%%:*}; W=${t#*:}; bash $H/score.sh $TD $Lb $W > /dev/null 2>&1
  python3 -I -c "
import json, sys, glob; R = sys.argv[1]; s = json.load(open(R + '/summary.json'))
ig = []
for ph in ('pre', 'post'):
    try: ig.append(str(sum(v.get('diff_lines', 0) for v in json.load(open(f'{R}/integrity_{ph}.json')).values() if isinstance(v, dict) and 'diff_lines' in v)))
    except Exception: ig.append('-')
pc = s.get('pair_checks') or []
fails = sorted({r.get('fail') for p in pc for r in ((p.get('outcheck') or {}).get('per_rep') or []) if r.get('fail')})
tt = [round(r['ttft_ratio'], 3) for p in pc for r in ((p.get('outcheck') or {}).get('per_rep') or []) if r.get('ttft_ratio')]
kinds = {}
for f in glob.glob(R + '/runs/*/audit.*.jsonl') + glob.glob(R + '/correct/audit.*.jsonl'):
    for l in open(f):
        try: k = json.loads(l).get('kind')
        except Exception: continue
        if k not in ('end', 'exec'): kinds[k] = kinds.get(k, 0) + 1
w = [x for p in pc for x in ((p.get('outcheck') or {}).get('ttft_watch') or [])]
print(f\"| {sys.argv[2]} | {s.get('verdict')} | {s.get('speedup')} | {','.join(s.get('flags') or [])} | {'/'.join(ig)} | {fails} | {tt}{' WATCH>1.03' if w else ''} | {kinds} |\")" $TD/results/$Lb $Lb >> $O 2>/dev/null || echo "| $Lb | (no summary) | | | | | | |" >> $O
done
echo "CANARIES16436 DONE"; cat $O
