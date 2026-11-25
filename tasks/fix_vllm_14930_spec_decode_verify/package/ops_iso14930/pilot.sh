#!/bin/bash
# pilot.sh <gpu> : transfer-load pilot (do the loads run on parent and fix?), REPS=3, dev-warm caches. Not scored.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
G=$1; P2=$ISO_ROOT; H=$P2/harness_iso14930; TD=$P2/tasks/iso14930; O=$P2/ops_iso14930/pilot
source $P2/ops_iso14930/guard.sh; mkdir -p $O
for w in t1:TEMP=1.0 k5:K=5 n16:N=16 noprime:PRIME=0; do n=${w%%:*}; kv=${w#*:}
  for r in parent fix; do
    stop_check || exit 0; gpu_free $G || exit 3
    echo "$(date -Is) start $n $r"
    GPU=$G bash $H/run_variant.sh $TD $r - "p2_iso14930_pilot_${n}_$r" $O/${n}_$r timed $kv REPS=3 > $O/${n}_$r.log 2>&1; echo "$(date -Is) end $n $r rc=$?"
  done
done
echo PILOT DONE
