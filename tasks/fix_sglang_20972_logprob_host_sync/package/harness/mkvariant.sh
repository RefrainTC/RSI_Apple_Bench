#!/bin/bash
# mkvariant.sh <name> : build a broken / cheating variant venv $STAGE/venvs/<pr>_var_<name> as a hard-linked copy of its
# base venv (parent or fix, see variants/apply.py) with Python-only edits; marker $STAGE/.staged/venv_<pr>_var_<name>.
# Then score it like any candidate: bash harness/run.sh <label> score var_<name>
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/../env.sh"; . "$ISO_ROOT/env/build.env"
N=$1; BASE=$("$SYSPY" -I "$ISO_ROOT/variants/apply.py" $PR $N --base) || exit 1
S=$STAGE/venvs/${PR}_$BASE; D=$STAGE/venvs/${PR}_var_$N
[ -f "$STAGE/.staged/venv_${PR}_$BASE" ] || { echo "base venv not staged: $S"; exit 1; }
rm -rf "$D" "$STAGE/.staged/venv_${PR}_var_$N"; cp -al "$S" "$D" || exit 1
"$SYSPY" -I "$ISO_ROOT/variants/apply.py" $PR $N "$D/lib/python3.10/site-packages" || { rm -rf "$D"; exit 1; }
"$D/bin/python3" -I -c "import sys; assert sys.prefix == '$D', sys.prefix; print('variant venv ok', sys.prefix)" || exit 1
echo "$(date -Is) base=$BASE" > "$STAGE/.staged/venv_${PR}_var_$N"; echo "VARIANT_OK $PR $N (base $BASE)"
