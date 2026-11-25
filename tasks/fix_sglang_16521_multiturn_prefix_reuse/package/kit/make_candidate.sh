#!/bin/bash
# make_candidate.sh <name> <agent_tree | patch.diff> : build candidate venv $STAGE/venvs/<pr>_<name> for harness/run.sh.
#   <agent_tree>  the agent's edited engine package directory (…/site-packages/sglang or …/vllm), a directory that
#                 contains it, or the agent's whole venv; or a unified diff against the parent engine package.
# Steps: verify the parent venv's engine files against the setup's hash record (and the kit seal, if made) -> fail loudly
# on any mismatch; check that only Python source (*.py) differs from the parent; copy the parent venv as a REAL copy
# (new inodes; nothing the candidate does can write through into the parent) and put the agent's files in; byte-compile
# the changed files (whole engine for vLLM packages, whose harness checks that runs do not write into the venv);
# relocate; touch $STAGE/.staged/venv_<pr>_<name>; record the change list in $ISO_ROOT/kit_state/cand_<name>.json.
# KIT_LINK_DEPS=1: hard-link everything outside the engine package instead (faster, ~10 GB less disk; the candidate
# could then write through into the parent's other packages: test.sh's checks would see it, but only after the fact).
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/lib/common.sh"; . "$KIT/lib/integrity.sh"
NAME=${1:-}; SRC=${2:-}
[ -n "$NAME" ] && [ -n "$SRC" ] || kit_die "usage: make_candidate.sh <name> <agent engine tree | venv | patch.diff>"
[[ "$NAME" =~ ^[a-z0-9][a-z0-9_]{0,40}$ ]] || kit_die "name must match [a-z0-9][a-z0-9_]*: $NAME"
case $NAME in parent|fix|var_*) kit_die "name '$NAME' is reserved";; esac
SRC=$(readlink -f "$SRC") || kit_die "no such path: $2"
D=$STAGE/venvs/${PR}_$NAME; M=$STAGE/.staged/venv_${PR}_$NAME; SP=$PV/$SPREL
case "$SRC/" in "$PV/"*|"$STAGE/venvs/${PR}_fix/"*) kit_die "the agent tree must not be the parent or fix venv itself";; esac
r=$(verify_parent_engine) || kit_die "$r"; kit_log "$r"
if [ -f "$KSTATE/seal.sha256" ]; then r=$(verify_seal) || kit_die "$r"; kit_log "$r"; fi
T=$(mktemp -d "$STAGE/tmp/kitc_${NAME}.XXXXXX") || kit_die "mktemp"; trap 'rm -rf "$T"' EXIT
export GIT_CEILING_DIRECTORIES=$T   # git apply must not find a repository above the scratch dir
mkdir -p "$T/new"
# 1. the agent's engine tree -> $T/new/$ENGINE (fresh files; bytecode left out)
if [ -f "$SRC" ]; then
  cp -a "$SP/$ENGINE" "$T/new/$ENGINE" || kit_die "copy parent engine"
  find "$T/new/$ENGINE" -name __pycache__ -type d -prune -exec rm -rf {} +
  ok=""; for k in 1 0 2 3 4 5; do
    if (cd "$T/new" && git apply --check -p$k "$SRC" 2>/dev/null); then ok=$k; break; fi
    if (cd "$T/new" && git apply --check -p$k --directory="$ENGINE" "$SRC" 2>/dev/null); then ok="$k --directory=$ENGINE"; break; fi
  done
  [ -n "$ok" ] || kit_die "the diff does not apply to the parent's $ENGINE package (tried -p0..-p5)"
  (cd "$T/new" && git apply -p$ok "$SRC") || kit_die "git apply failed"
  [ -z "$(find "$T/new" -mindepth 1 -maxdepth 1 ! -name "$ENGINE")" ] || kit_die "the diff touches files outside $ENGINE/"
  SRCKIND="diff (git apply -p$ok)"
else
  [ -d "$SRC" ] || kit_die "not a file or directory: $SRC"
  if [ -f "$SRC/__init__.py" ] && [ "$(basename "$SRC")" = "$ENGINE" ]; then E=$SRC
  elif [ -d "$SRC/$ENGINE" ]; then E=$SRC/$ENGINE
  elif [ -d "$SRC/$SPREL/$ENGINE" ]; then E=$SRC/$SPREL/$ENGINE
  else kit_die "no $ENGINE package in $SRC"; fi
  bad=$(find "$E" ! -type f ! -type d | head -3); [ -z "$bad" ] || kit_die "the agent tree may hold only regular files: $bad"
  (cd "$(dirname "$E")" && tar -cf - --exclude=__pycache__ --exclude='*.pyc' "$(basename "$E")") | (cd "$T/new" && tar -xf -) \
    || kit_die "copy of the agent tree failed"
  [ "$(basename "$E")" = "$ENGINE" ] || mv "$T/new/$(basename "$E")" "$T/new/$ENGINE"
  SRCKIND="tree $E"
fi
# 2. what differs from the parent: only *.py may differ
lst() { (cd "$1" && find "$ENGINE" -type f ! -name '*.pyc' ! -path '*/__pycache__/*' | LC_ALL=C sort); }
lst "$SP" > "$T/p.lst"; lst "$T/new" > "$T/n.lst"
comm -23 "$T/p.lst" "$T/n.lst" > "$T/removed"; comm -13 "$T/p.lst" "$T/n.lst" > "$T/added"
comm -12 "$T/p.lst" "$T/n.lst" | while read -r f; do cmp -s "$SP/$f" "$T/new/$f" || echo "$f"; done > "$T/changed"
nonpy=$(cat "$T/removed" "$T/added" "$T/changed" | grep -v '\.py$' | head -5)
[ -z "$nonpy" ] || kit_die "only Python source (*.py) may differ from the parent; these do not: $(echo $nonpy)"
nch=$(cat "$T/changed" "$T/added" "$T/removed" | wc -l)
kit_log "agent tree: $SRCKIND; $(wc -l < "$T/changed") changed, $(wc -l < "$T/added") added, $(wc -l < "$T/removed") removed .py files"
# 3. the candidate venv: a real copy of the parent, the agent's files put in
rm -rf "$D" "$M"
if [ "${KIT_LINK_DEPS:-0}" = 1 ]; then
  cp -al "$PV" "$D" || kit_die "hard-link copy"; rm -rf "$D/$SPREL/$ENGINE"; cp -a "$SP/$ENGINE" "$D/$SPREL/$ENGINE" || kit_die "engine copy"
else
  cp -a "$PV" "$D" || kit_die "copy of the parent venv failed"
fi
DS=$D/$SPREL
rmpyc() { local b; b=$(basename "${1%.py}"); rm -f "$DS/$(dirname "$1")/__pycache__/$b".*.pyc; }
while read -r f; do [ -n "$f" ] || continue; rm -f "$DS/$f"; rmpyc "$f"; done < "$T/removed"
cat "$T/changed" "$T/added" | while read -r f; do [ -n "$f" ] || continue
  mkdir -p "$DS/$(dirname "$f")"; rm -f "$DS/$f"; cp "$T/new/$f" "$DS/$f"; chmod 644 "$DS/$f"; rmpyc "$f"; done
# relocate (as setup/stage.sh does for the parent): interpreter link and pyvenv.cfg are absolute and stay valid
grep -l -I "$PV" "$D"/bin/* 2>/dev/null | xargs -r sed -i "s#$PV#$D#g"
chk=$("$D/bin/python3" -I -c "import sys; print(sys.prefix)") ; [ "$chk" = "$D" ] || kit_die "relocation failed: sys.prefix=$chk"
# 4. byte-compile: the changed files always (a syntax error fails here); the whole engine for vLLM packages
cat "$T/changed" "$T/added" | grep . | sed "s#^#$DS/#" > "$T/compile.lst"
if [ -s "$T/compile.lst" ]; then
  "$D/bin/python3" -I -c "import py_compile, sys
for f in open(sys.argv[1]).read().split():
    py_compile.compile(f, doraise=True)" "$T/compile.lst" || { rm -rf "$D"; kit_die "candidate does not byte-compile (syntax error)"; }
fi
if [ "$ENGINE" = vllm ]; then "$D/bin/python3" -I -m compileall -q -j 8 "$DS/$ENGINE" > /dev/null 2>&1; fi
"$D/bin/python3" -I -c "import sys; assert sys.prefix == '$D'; import importlib.util as u; assert u.find_spec('$ENGINE').origin.startswith('$DS/'), 'engine not from the candidate'" \
  || { rm -rf "$D"; kit_die "import check of the candidate failed"; }
# 5. the parent must be untouched by all of this
r=$(verify_parent_engine) || kit_die "after build: $r"
mkdir -p "$KSTATE"
"$SYSPY" -I - "$T" "$KSTATE/cand_$NAME.json" "$NAME" "$SRCKIND" "$DS" <<'PY'
import hashlib, json, os, sys, time
t, out, name, kind, ds = sys.argv[1:]
rd = lambda f: [l for l in open(os.path.join(t, f)).read().split("\n") if l]
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
json.dump({"name": name, "source": kind, "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "changed": {f: h(os.path.join(ds, f)) for f in rd("changed")}, "added": {f: h(os.path.join(ds, f)) for f in rd("added")},
           "removed": rd("removed")}, open(out, "w"), indent=1)
PY
echo "$(date -Is) kit make_candidate: $SRCKIND; $nch .py files differ" > "$M"
echo "CANDIDATE_OK $PR $NAME ($nch .py files differ from the parent; venv $D)"
