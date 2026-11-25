#!/bin/bash
# n1_venv_clean.sh : remove .pyc files that my CPU microbenchmark wrote into n1's staged master venvs (present on n1,
# absent from the main manifest); any other difference is only reported. Then re-verify.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
M=$ISO_ROOT/tasks/iso13365/manifest
hashlist() { (cd "$1" && shift && find "$@" -type f ! -name pyvenv.cfg ! -path '*/__pycache__/*.pyc.*' -print0 2>/dev/null | LC_ALL=C sort -z | xargs -0 -r -P 8 -n 300 sha256sum | LC_ALL=C sort -k2); }
for p in venv_parent:13365_base venv_fix:13365_fix; do
  n=${p%%:*}; v=$P2T/venvs/${p#*:}
  diff <(hashlist $v .) $M/$n.sha256 | python3 -I -c "
import sys, os
v = sys.argv[1]; L = [l.rstrip('\n') for l in sys.stdin if l[:1] in '<>']
cur = {l[2:].split('  ', 1)[1] for l in L if l[0] == '<'}; ref = {l[2:].split('  ', 1)[1] for l in L if l[0] == '>'}
extra = cur - ref; other = (cur & ref) | (ref - cur)
rm = [p for p in extra if '/__pycache__/' in p and p.endswith('.pyc')]
print(sys.argv[2], 'extra', len(extra), 'removable_pyc', len(rm), 'changed_or_missing', len(other), sorted(other)[:5])
if len(rm) == len(extra):
    for p in rm: os.remove(os.path.join(v, p))
    print(sys.argv[2], 'removed', len(rm))
" $v $n
done
bash $ISO_ROOT/ops_iso13365/n1_venv_verify.sh | grep diff_lines
