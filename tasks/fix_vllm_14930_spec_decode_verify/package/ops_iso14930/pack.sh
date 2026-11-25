#!/bin/bash
# pack.sh : small text results of #14930 into $P2T/tmp/iso14930_pack.tgz (no venvs, weights, out_ref / outputs.json)
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
P2=$ISO_ROOT; cd $P2
D=$(mktemp -d -p $P2T/tmp); trap 'rm -rf "$D"' EXIT
O=$D/iso14930; mkdir -p $O
diff -ru harness harness_iso14930 -x __pycache__ > $O/harness_iso14930_vs_harness.diff 2>&1
for w in canaries/C1 canaries/C2 canaries/C3 alt/V1; do diff -ru tasks/iso14930/workspace/repo ops_iso14930/$w/repo -x .git -x __pycache__ > $O/$(basename $w).diff 2>&1; done
find tasks/iso14930 -maxdepth 1 -type f \( -name '*.py' -o -name '*.sh' -o -name '*.md' -o -name task.env -o -name pristine.sha256 \) > $D/list
find tasks/iso14930/calib -maxdepth 1 -type f \( -name calib.json -o -name calibrate.log -o -name ref_jobs.json -o -name outcheck_calibration.json -o -name summary.json \) >> $D/list
echo tasks/iso14930/calib/oracle_stage/oracle_stage.json >> $D/list
find tasks/iso14930/manifest -name 'scorer.sha256' -o -name 'ref.sha256' -o -name 'made_at.txt' >> $D/list
find tasks/iso14930/results -maxdepth 2 -type f \( -name 'summary.*' -o -name runs_summary.json -o -name gate.txt -o -name changes.patch -o -name score.log -o -name 'integrity_*.json' -o -name phase1.json \) >> $D/list
find tasks/iso14930/results tasks/iso14930/calib/runs ops_iso14930/study ops_iso14930/smoke ops_iso14930/pilot -name passes.json -o -name job_meta.json -o -name settings.json -o -name variant.json | grep -v '\.aborted\.' >> $D/list
find tasks/iso14930/results -path '*correct*' -name job_meta.json >> $D/list
find ops_iso14930 -maxdepth 1 -type f \( -name '*.sh' -o -name '*.py' -o -name '*.log' \) >> $D/list
echo ops_iso14930/pool/screen_g2/pool_screen.json >> $D/list
find ops_iso14930/gatecheck -name gate.txt >> $D/list
sort -u $D/list | tar cf - -T - | tar xf - -C $O
tar czf $P2T/tmp/iso14930_pack.tgz -C $D iso14930; ls -la $P2T/tmp/iso14930_pack.tgz; du -sh $O
