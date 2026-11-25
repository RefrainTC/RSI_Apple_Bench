#!/bin/bash
# calib.sh [gpu=1] : iso13365 calibration (harness_iso13365/calibrate.sh, 5 AB + 5 A/A + 1 venv-equivalence pair) on main GPU1.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
source $ISO_ROOT/ops_iso13365/common.sh
G=${1:-1}; stop_check calibrate; gpu_free $G || exit 3
log "calibrate start $(hostctx)"
GPU=$G bash $H/calibrate.sh $TD 5 5; rc=$?
log "calibrate end rc=$rc $(hostctx) $(cat $TD/calib/calib.json 2>/dev/null | python3 -I -c 'import json,sys; c=json.load(sys.stdin); print(c.get("ref_speedup"), c.get("noise_band"), c.get("fix_pairs",{}).get("values"), c.get("aa",{}).get("values"))')"
