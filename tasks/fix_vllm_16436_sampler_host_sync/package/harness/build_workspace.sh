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
# instruction.md (package change 2026-11-25, acceptance acc7): when this package sits as package/ inside a public task
# directory (../instruction.md and ../task.toml next to $ISO_ROOT), the workspace gets that public instruction, the task
# statement an agent is given there; otherwise (stand-alone package, internal trials) the package's own task text.
PUBI=$(readlink -f "$ISO_ROOT/..")/instruction.md
if [ -f "$PUBI" ] && [ -f "$(dirname "$PUBI")/task.toml" ]; then cp "$PUBI" "$WS/instruction.md"; INSTR_SRC=$PUBI
else cp "$TD/instruction.md" "$WS/instruction.md"; INSTR_SRC=$TD/instruction.md; fi
cp "$TD/WORKLOAD.md" "$WS/WORKLOAD.md"
# dev_run.sh measures the workspace it lives in (package change 2026-11-25, acceptance acc7: it used to measure
# $TD/workspace, so run from a copy it timed the untouched workspace and wrote dev_runs/ there).
cat > "$WS/dev_run.sh" <<EOW
#!/bin/bash
# dev_run.sh [--baseline] [--correct] [VAR=VALUE ...] : test the workspace this script is in on the GPU (see instruction.md).
exec bash $ISO_ROOT/harness/dev_run_impl.sh $TD --ws "\$(dirname "\$(readlink -f "\$0")")" "\$@"
EOW
chmod +x "$WS/dev_run.sh"
( cd "$WS/repo" && git init -q && git add -A && git -c user.name=p2 -c user.email=p2@local commit -qm pristine )
( cd "$WS" && printf 'dev_runs/\nrepo/\n' > .gitignore && git init -q && git add -A && git -c user.name=p2 -c user.email=p2@local commit -qm pristine )
cp -a "$WS" "$PR"; rm -rf "$PR/.git" "$PR/repo/.git"
( cd "$PR" && find . -type f -print0 | sort -z | xargs -0 sha256sum ) > "$TD/pristine.sha256"
chmod -R a-w "$PR"
echo "workspace: $WS ($(find "$WS/repo" -name '*.py' | wc -l) .py files); instruction.md from $INSTR_SRC; pristine manifest: $TD/pristine.sha256"
