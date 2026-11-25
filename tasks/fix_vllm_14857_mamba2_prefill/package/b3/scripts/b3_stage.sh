#!/bin/bash
# b3_stage.sh <item> ... : batch-3 copy of p2-port/scripts/stage_tmp.sh. Copies NAS inputs to this pod's NVMe under
# /tmp/p2/b3 (venvs, hf). Uses the interpreter copy /tmp/p2/uvpy that stage_tmp.sh already staged on this pod (read only).
# Items: venv:<src dir> -> /tmp/p2/b3/venvs/<basename> (relocated to /tmp/p2/uvpy); hf:<org/name> -> /tmp/p2/b3/hf/hub
# (source: b3/hf/hub, else the T5 hub). Idempotent; marker per item in /tmp/p2/b3/.staged.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/../../env.sh"; D=$STAGE/b3; B3=$ISO_ROOT/b3; T5=$ISO_ROOT
PYDST=$STAGE/uvpy/cpython-3.10.20-linux-x86_64-gnu
[ -f $STAGE/.staged/py ] && [ -x $PYDST/bin/python3.10 ] || { echo "no staged interpreter at $PYDST"; exit 1; }
mkdir -p $D/venvs $D/hf/hub $D/cache $D/scratch $D/.staged; LOG=$D/stage.log
sig() { find "$1" \( -type f -o -type l \) ! -name pyvenv.cfg -printf '%y %s\n' | awk '{n++; if ($1 == "f") b += $2} END {printf "%d %.0f\n", n, b}'; }
copy() {
  local src=$1 dst=$2 name=$3 tmp="$2.partial"
  if [ -f "$D/.staged/$name" ] && [ -d "$dst" ]; then echo "skip $name"; return 0; fi
  echo "$(date '+%F %T') copy $name: $src -> $dst" | tee -a $LOG
  local t0=$(date +%s); rm -rf "$tmp" "$dst"; mkdir -p "$tmp"
  tar -C "$src" -cf - . | tar -C "$tmp" -xf - || { echo "COPY FAILED $name" | tee -a $LOG; rm -rf "$tmp"; return 1; }
  local a b; a=$(sig "$src"); b=$(sig "$tmp")
  [ "$a" = "$b" ] || { echo "VERIFY FAILED $name: src=$a dst=$b" | tee -a $LOG; rm -rf "$tmp"; return 1; }
  mv "$tmp" "$dst"; echo "$(date -Is) src=$src files_bytes=$a secs=$(( $(date +%s) - t0 ))" > "$D/.staged/$name"
  echo "$(date '+%F %T') done $name: $a in $(( $(date +%s) - t0 )) s" | tee -a $LOG
}
relocate() {
  local v=$1; ln -sfn $PYDST/bin/python3.10 "$v/bin/python"; sed -i "s|^home = .*|home = $PYDST/bin|" "$v/pyvenv.cfg"
  local chk; chk=$("$v/bin/python3" -I -c "import sys, sysconfig; print(sys.prefix, sys.base_prefix, sysconfig.get_paths()['purelib'])")
  case "$chk" in "$v $PYDST $v/lib/python3.10/site-packages") echo "relocated ok";; *) echo "RELOCATE FAILED $v: $chk" | tee -a $LOG; return 1;; esac
}
rc=0
for it in "$@"; do
  case "$it" in
    venv:*) src=${it#venv:}; n=$(basename "$src")
            if copy "$src" "$D/venvs/$n" "venv_$n"; then
              grep -q "^home = $PYDST/bin" "$D/venvs/$n/pyvenv.cfg" || relocate "$D/venvs/$n" || { rm -rf "$D/venvs/$n" "$D/.staged/venv_$n"; rc=1; }
            else rc=1; fi ;;
    hf:*)   m=${it#hf:}; dn="models--${m//\//--}"; src=$B3/hf/hub/$dn; [ -d $src ] || src=$T5/hf/hub/$dn
            copy "$src" "$D/hf/hub/$dn" "hf_$dn" || rc=1 ;;
    dir:*)  src=${it#dir:}; copy "$src" "$D/$(basename $src)" "dir_$(basename $src)" || rc=1 ;;
    *) echo "bad item $it"; rc=1 ;;
  esac
done
echo "$(date '+%F %T') B3_STAGE DONE rc=$rc ($*)" | tee -a $LOG; exit $rc
