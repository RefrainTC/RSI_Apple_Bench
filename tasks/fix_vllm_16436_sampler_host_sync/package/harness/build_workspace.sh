#!/bin/bash
# build_workspace.sh <taskdir> [--force] : agent workspace = $TD/workspace (instruction.md, repo/<pkg>/**.py copied from the
# pristine venv, dev_run.sh, WORKLOAD.md); pristine copy = $TD/pristine (+ pristine.sha256). Own git in workspace/ and repo/.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -euo pipefail
TD=$(readlink -f "$1"); FORCE=${2:-}
source "$TD/task.env"
WS=$TD/workspace; PR=$TD/pristine
if [ -e "$WS" ] && [ "$FORCE" != "--force" ]; then echo "workspace exists (use --force to rebuild; wipes agent changes)"; exit 1; fi
[ -d "$PR" ] && chmod -R u+w "$PR"   # the pristine copy is read-only
rm -rf "$WS" "$PR"; mkdir -p "$WS/repo"
SITE=$("$VENV/bin/python3" -I -c "import sysconfig;print(sysconfig.get_paths()['purelib'])")
( cd "$SITE" && find "$PKG" -name '*.py' -not -path '*/__pycache__/*' -print0 | xargs -0 cp --parents -t "$WS/repo" )
cp "$TD/instruction.md" "$WS/instruction.md"
cp "$TD/WORKLOAD.md" "$WS/WORKLOAD.md"
cat > "$WS/dev_run.sh" <<EOW
#!/bin/bash
# dev_run.sh [--baseline] [--correct] [VAR=VALUE ...] : test your current workspace on the GPU (see instruction.md).
exec bash $ISO_ROOT/harness/dev_run_impl.sh $TD "\$@"
EOW
chmod +x "$WS/dev_run.sh"
( cd "$WS/repo" && git init -q && git add -A && git -c user.name=p2 -c user.email=p2@local commit -qm pristine )
( cd "$WS" && printf 'dev_runs/\nrepo/\n' > .gitignore && git init -q && git add -A && git -c user.name=p2 -c user.email=p2@local commit -qm pristine )
cp -a "$WS" "$PR"; rm -rf "$PR/.git" "$PR/repo/.git"
( cd "$PR" && find . -type f -print0 | sort -z | xargs -0 sha256sum ) > "$TD/pristine.sha256"
chmod -R a-w "$PR"
echo "workspace: $WS ($(find "$WS/repo" -name '*.py' | wc -l) .py files); pristine manifest: $TD/pristine.sha256"
