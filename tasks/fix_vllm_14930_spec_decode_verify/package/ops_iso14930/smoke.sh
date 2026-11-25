#!/bin/bash
# smoke.sh <gpu> : parent timed x2, fix timed x1, parent correct x1, fix correct x1 (dev-warm caches), sequentially.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
G=${1:-2}; P2=$ISO_ROOT; H=$P2/harness_iso14930; TD=$P2/tasks/iso14930; O=$P2/ops_iso14930/smoke
source $P2/ops_iso14930/guard.sh; mkdir -p $O
export GPU=$G
for j in p1:parent:timed f1:fix:timed p2:parent:timed pc:parent:correct fc:fix:correct; do
  n=${j%%:*}; r=$(echo $j | cut -d: -f2); m=${j##*:}
  stop_check || exit 0; gpu_free $G || exit 3
  echo "$(date -Is) start $n $r $m"
  GPU=$G bash $H/run_variant.sh $TD $r - "p2_iso14930_smoke_$n" $O/$n $m > $O/$n.log 2>&1; echo "$(date -Is) end $n rc=$?"
done
echo SMOKE DONE
