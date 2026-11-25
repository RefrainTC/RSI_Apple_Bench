#!/bin/bash
# pkg_stage.sh : copy the interpreter, both venvs, the model (and #6994's LoRA adapters) to fast local disk under $STAGE,
# using the authors' stage_tmp.sh (interpreter) and b3_stage.sh (the rest). Package addition; b3_caldrv.sh would also
# stage, but staging before the timed jobs keeps the copies away from them. Idempotent.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/../../env.sh"; . "$ISO_ROOT/env/build.env"; P2=$ISO_ROOT; B3=$P2/b3
. $B3/env/${PR}_cal.env
bash $P2/scripts/stage_tmp.sh py || exit 1
bash $B3/scripts/b3_stage.sh venv:$B3/venvs/${PR}_parent venv:$B3/venvs/${PR}_fix ${STAGE_EXTRA:-} hf:$M || exit 1
echo "PKG_STAGE_OK $PR ($(du -sh $STAGE | cut -f1) in $STAGE)"
