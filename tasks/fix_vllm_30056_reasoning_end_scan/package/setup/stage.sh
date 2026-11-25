#!/bin/bash
# stage.sh : copy the interpreter, both venvs, and the model to fast local disk under $STAGE (file count and
# bytes checked; venvs relocated to the staged interpreter). Idempotent (markers in $STAGE/.staged). Do not stage while
# timed jobs run on the same machine.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/../env.sh"; . "$ISO_ROOT/env/build.env"
PYN=cpython-3.10.20-linux-x86_64-gnu; PYDST=$STAGE/py/$PYN
mkdir -p $STAGE/venvs $STAGE/hf/hub $STAGE/.staged $STAGE/tmp $STAGE/cache; rc=0
sig() { find "$1" \( -type f -o -type l \) ! -name pyvenv.cfg -printf '%y %s\n' | awk '{n++; if ($1 == "f") b += $2} END {printf "%d %.0f\n", n, b}'; }
copy() {  # src dst marker
  local src=$1 dst=$2 name=$3 t0=$(date +%s)
  [ -f "$STAGE/.staged/$name" ] && [ -d "$dst" ] && { echo "skip $name"; return 0; }
  rm -rf "$dst.partial" "$dst"; mkdir -p "$dst.partial"
  tar -C "$src" -cf - . | tar -C "$dst.partial" -xf - || { echo "COPY FAILED $name"; return 1; }
  local a b; a=$(sig "$src"); b=$(sig "$dst.partial")
  [ "$a" = "$b" ] || { echo "VERIFY FAILED $name: $a vs $b"; rm -rf "$dst.partial"; return 1; }
  mv "$dst.partial" "$dst"; echo "$(date -Is) src=$src $a" > "$STAGE/.staged/$name"; echo "staged $name ($a) in $(( $(date +%s) - t0 )) s"
}
copy $ISO_ROOT/.uv-python/$PYN $PYDST py || rc=1
for role in parent fix; do v=$STAGE/venvs/${PR}_$role
  if copy $ISO_ROOT/venvs/${PR}_$role $v venv_${PR}_$role; then
    ln -sfn $PYDST/bin/python3.10 $v/bin/python; sed -i "s|^home = .*|home = $PYDST/bin|" $v/pyvenv.cfg
    grep -l -I "$ISO_ROOT/venvs/${PR}_$role" $v/bin/* 2>/dev/null | xargs -r sed -i "s#$ISO_ROOT/venvs/${PR}_$role#$v#g"
    chk=$($v/bin/python3 -I -c "import sys, sysconfig; print(sys.prefix, sys.base_prefix)")
    [ "$chk" = "$v $PYDST" ] && echo "relocated $role ok" || { echo "RELOCATE FAILED $v: $chk"; rm -f $STAGE/.staged/venv_${PR}_$role; rc=1; }
  else rc=1; fi
done
M=models--${MODEL//\//--}; copy $ISO_ROOT/hf/hub/$M $STAGE/hf/hub/$M hf_$M || rc=1
[ $rc = 0 ] && echo "STAGE_OK $PR ($(du -sh $STAGE | cut -f1) in $STAGE)" || echo "STAGE FAILED"; exit $rc
