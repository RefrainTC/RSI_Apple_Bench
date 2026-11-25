#!/bin/bash
# stage_n1.sh (iso13365 copy of scripts/stage_tmp.sh with the root as STAGE_ROOT, for n1 /tmp/ag13365) <item> ... : copy re-creatable inputs from the NAS (/home/coder, NFS) to the pod's local NVMe under $P2T.
# /tmp is wiped on pod restart and at the 11-26 maintenance: just run this again (idempotent; finished items are skipped).
# Items (copied strictly one at a time, each as one sequential tar stream, so the NAS is read once):
#   py                 uv CPython 3.10 (the venvs' base interpreter + stdlib)        -> $P2T/uvpy/
#   venv:<src dir>     a venv; relocated: bin/python -> /tmp copy of the interpreter,  -> $P2T/venvs/<basename>
#                      pyvenv.cfg home -> same (nothing else in a uv venv holds its own path that our jobs use:
#                      we always run `python3 -m ...`, never console-script shebangs)
#   hf:<org/name>      one HF hub model dir from $T5/hf                                 -> $P2T/hf/hub/
# After each copy: file count + byte total must match the source, else the item is removed and the script fails.
# Marker per item: $P2T/.staged/<name> (source, counts, time). Jobs refuse to use an item without its marker.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
D=${STAGE_ROOT:?set STAGE_ROOT}; T5=$ISO_ROOT
PYSRC=$ISO_ROOT/.uv-python/cpython-3.10.20-linux-x86_64-gnu; PYDST=$D/uvpy/cpython-3.10.20-linux-x86_64-gnu
mkdir -p $D/venvs $D/hf/hub $D/uvpy $D/cache $D/scratch $D/overlays $D/.staged
LOG=$D/stage.log
# entries + file bytes; pyvenv.cfg is excluded because relocation rewrites its home line (and nothing else)
sig() { find "$1" \( -type f -o -type l \) ! -name pyvenv.cfg -printf '%y %s\n' | awk '{n++; if ($1 == "f") b += $2} END {printf "%d %.0f\n", n, b}'; }
copy() {  # <src> <dst> <marker name>
  local src=$1 dst=$2 name=$3 tmp="$2.partial"
  if [ -f "$D/.staged/$name" ] && [ -d "$dst" ]; then echo "skip $name (staged $(cut -d' ' -f1-3 $D/.staged/$name))"; return 0; fi
  echo "$(date '+%F %T') copy $name: $src -> $dst" | tee -a $LOG
  local t0=$(date +%s); rm -rf "$tmp" "$dst"; mkdir -p "$tmp"
  tar -C "$src" -cf - . | tar -C "$tmp" -xf - || { echo "COPY FAILED $name" | tee -a $LOG; rm -rf "$tmp"; return 1; }
  local a b; a=$(sig "$src"); b=$(sig "$tmp")
  [ "$a" = "$b" ] || { echo "VERIFY FAILED $name: src=$a dst=$b" | tee -a $LOG; rm -rf "$tmp"; return 1; }
  mv "$tmp" "$dst"
  echo "$(date -Is) src=$src files_bytes=$a secs=$(( $(date +%s) - t0 ))" > "$D/.staged/$name"
  echo "$(date '+%F %T') done $name: $a in $(( $(date +%s) - t0 )) s" | tee -a $LOG
}
relocate() {  # <venv dir>: point the venv at the /tmp interpreter
  local v=$1
  ln -sfn $PYDST/bin/python3.10 "$v/bin/python"
  sed -i "s|^home = .*|home = $PYDST/bin|" "$v/pyvenv.cfg"
  local chk; chk=$("$v/bin/python3" -I -c "import sys, sysconfig; print(sys.prefix, sys.base_prefix, sysconfig.get_paths()['purelib'])")
  case "$chk" in "$v $PYDST $v/lib/python3.10/site-packages") echo "relocated ok: $chk";;
    *) echo "RELOCATE FAILED $v: $chk" | tee -a $LOG; return 1;; esac
}
rc=0
if [ "${1:-}" = --reverify ]; then   # re-check finished items against their source (metadata only: entry count + file bytes)
  for m in $D/.staged/*; do n=$(basename $m); src=$(sed -n 's/.* src=\([^ ]*\) .*/\1/p' $m)
    case $n in py) dst=$PYDST;; venv_*) dst=$D/venvs/${n#venv_};; hf_*) dst=$D/hf/hub/${n#hf_};; esac
    a=$(sig "$src"); b=$(sig "$dst")
    # relocation changes one symlink target (bin/python), not the counts or file bytes
    if [ "$a" = "$b" ]; then sed -i "s|files_bytes=[0-9]* [0-9]*|files_bytes=$a|" $m; echo "reverified $n: $a"
    else echo "REVERIFY FAILED $n: src=$a dst=$b"; rm -rf "$dst" "$m"; rc=1; fi
  done; exit $rc
fi
for it in "$@"; do
  case "$it" in
    py)     copy "$PYSRC" "$PYDST" py || rc=1 ;;
    venv:*) src=${it#venv:}; n=$(basename "$src")
            [ -f $D/.staged/py ] || { echo "stage py first"; rc=1; continue; }
            if copy "$src" "$D/venvs/$n" "venv_$n"; then
              grep -q "^home = $PYDST/bin" "$D/venvs/$n/pyvenv.cfg" || relocate "$D/venvs/$n" || { rm -rf "$D/venvs/$n" "$D/.staged/venv_$n"; rc=1; }
            else rc=1; fi ;;
    hf:*)   m=${it#hf:}; dn="models--${m//\//--}"
            copy "$T5/hf/hub/$dn" "$D/hf/hub/$dn" "hf_$dn" || rc=1 ;;
    *) echo "bad item $it"; rc=1 ;;
  esac
done
echo "$(date '+%F %T') STAGE_TMP DONE rc=$rc ($*)" | tee -a $LOG
exit $rc
