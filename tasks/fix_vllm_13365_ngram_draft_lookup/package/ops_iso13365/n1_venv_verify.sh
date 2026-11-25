#!/bin/bash
# n1_venv_verify.sh : compare n1's staged venvs with the main pod's integrity manifests (same hash list as integrity.sh).
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
M=$ISO_ROOT/tasks/iso13365/manifest
hashlist() { (cd "$1" && shift && find "$@" -type f ! -name pyvenv.cfg ! -path '*/__pycache__/*.pyc.*' -print0 2>/dev/null | LC_ALL=C sort -z | xargs -0 -r -P 8 -n 300 sha256sum | LC_ALL=C sort -k2); }
for p in venv_parent:13365_base venv_fix:13365_fix; do
  n=${p%%:*}; v=${p#*:}
  d=$(diff <(hashlist $P2T/venvs/$v .) $M/$n.sha256 | grep -c '^[<>]')
  echo "$n ($v): diff_lines=$d"; [ "$d" = 0 ] || diff <(hashlist $P2T/venvs/$v .) $M/$n.sha256 | head -10
done
