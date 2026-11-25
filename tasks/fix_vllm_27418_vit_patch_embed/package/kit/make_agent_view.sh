#!/bin/bash
# make_agent_view.sh <out_dir> : build what the agent under test gets, and nothing that reveals the human fix.
#   <out>/venv        real copy of the staged parent venv (no hard links), relocated to <out>/st/py
#   <out>/st/         copies of the staged interpreter, model and CUDA tree (dev_bench.sh's STAGE)
#   <out>/bench, loads  the benchmark and the loads (all but tf/transfer), comments and docstrings removed (the code is
#                     unchanged: AST-checked); env.sh, site.env (GPU, CPUS, PORT only), harness/job.sh (verbatim),
#                     harness/dev_env.sh (the timed runs' job environment, taken from harness/run.sh)
#   <out>/dev_bench.sh, README.md
# Not in the view: the fix venv or its sources, variants/, env/ (files_differing.txt, commits), RUN.md, PROVENANCE.md,
# gate.json, REF, check.py, run.sh. Ends with a leak scan (report in kit_state/, not in the view): hard terms (fix
# commit, changed file names, REF values, gate/variant/RUN.md names) must not occur; soft terms (kit/leakterms/<pr>.txt
# and the mechanism phrase of RUN.md's title) are listed for review. Needs ~15 GB under <out_dir>. CPU only.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/lib/common.sh"
O=${1:-}; [ -n "$O" ] || kit_die "usage: make_agent_view.sh <out_dir>"
[ -e "$O" ] && [ -n "$(ls -A "$O" 2>/dev/null)" ] && kit_die "$O exists and is not empty"
r=$(verify_parent_engine) || kit_die "$r"; kit_log "$r"
mkdir -p "$O" && O=$(readlink -f "$O") || kit_die "cannot create $O"
case "$O/" in "$ISO_ROOT/"*|"$STAGE/"*) kit_die "put the view outside the package and STAGE";; esac
t0=$(date +%s)
mkdir -p "$O/st" "$O/bench" "$O/loads" "$O/harness" "$O/runs"
# 1. interpreter, venv, model, CUDA: real copies
PYN=$(cd "$STAGE/py" && ls -d cpython-* | head -1); [ -n "$PYN" ] || kit_die "no staged interpreter in $STAGE/py"
cp -a "$STAGE/py" "$O/st/py" || kit_die "copy interpreter"
cp -a "$PV" "$O/venv" || kit_die "copy venv"
[ -d "$STAGE/hf" ] && { cp -a "$STAGE/hf" "$O/st/hf" || kit_die "copy model"; }
[ -d "$STAGE/cuda" ] && { cp -a "$STAGE/cuda" "$O/st/cuda" || kit_die "copy CUDA tree"; }
mkdir -p "$O/st/tmp" "$O/st/cache"
ln -sfn "$O/st/py/$PYN/bin/python3.10" "$O/venv/bin/python"
sed -i -e "s|^home = .*|home = $O/st/py/$PYN/bin|" -e '/^prompt = /d' "$O/venv/pyvenv.cfg"
grep -l -I "$PV" "$O"/venv/bin/* 2>/dev/null | xargs -r sed -i "s#$PV#$O/venv#g"
# build paths in text files (direct_url.json, sysconfig data, ...) name the package directory: neutralise them.
# Bytecode records the path it was compiled at (the package's venv name): drop it and recompile in place.
# (the interpreter's sysconfig data points at the setup's install dir: point it at the view's copy instead)
grep -rlI -F "$ISO_ROOT" "$O/venv" "$O/st/py" 2>/dev/null \
  | xargs -r sed -i -e "s#$ISO_ROOT/.uv-python/$PYN#$O/st/py/$PYN#g" -e "s#$ISO_ROOT#/build/pkg#g"
find "$O/venv" "$O/st/py" -name __pycache__ -type d -prune -exec rm -rf {} +
"$O/st/py/$PYN/bin/python3.10" -I -m compileall -q -j 8 "$O/st/py/$PYN/lib/python3.10" > /dev/null 2>&1
chk=$("$O/venv/bin/python3" -I -c "import sys; print(sys.prefix, sys.base_prefix)")
[ "$chk" = "$O/venv $O/st/py/$PYN" ] || kit_die "venv relocation failed: $chk"
"$O/venv/bin/python3" -I -m compileall -q -j 8 "$O/venv/$SPREL" > /dev/null 2>&1
# 2. benchmark and loads, without comments
for f in "$ISO_ROOT"/bench/*; do b=$(basename "$f")
  case $b in *.py) "$SYSPY" -I "$KIT/lib/strip.py" py "$f" "$O/bench/$b" || kit_die "strip $b";;
             *.sh) "$SYSPY" -I "$KIT/lib/strip.py" sh "$f" "$O/bench/$b" || kit_die "strip $b";;
             __pycache__) ;; *) cp "$f" "$O/bench/$b";; esac; done
for f in "$ISO_ROOT"/loads/*.env; do b=$(basename "$f")
  case $b in tf.env|tr_*) continue;; esac
  "$SYSPY" -I "$KIT/lib/strip.py" sh "$f" "$O/loads/$b" || kit_die "strip $b"; done
grep -ho '\$ISO_ROOT/[A-Za-z0-9_./-]*' "$O"/loads/*.env "$O"/bench/* | sort -u | while read -r ref; do
  [ -e "$O/${ref#\$ISO_ROOT/}" ] || kit_die "the view's loads/bench refer to ${ref} which is not in the view: needs an adapter"; done || exit 2
# 3. job environment: env.sh and job.sh verbatim (generic), dev_env.sh from harness/run.sh, site.env
cp "$ISO_ROOT/env.sh" "$O/env.sh"; cp "$ISO_ROOT/harness/job.sh" "$O/harness/job.sh"
printf 'GPU=%s\nCPUS=%s\nPORT=%s\n' "$GPU" "$CPUS" "$PORT" > "$O/site.env"
"$SYSPY" -I - "$ISO_ROOT/harness/run.sh" "$O/harness/dev_env.sh" <<'PY' || kit_die "harness/run.sh has an unknown job() layout: needs an adapter"
import re, sys
s = open(sys.argv[1]).read().split("\n")
C = [l.strip() for l in s if re.match(r"\s*local C=\S+", l)]
MK = [l.strip() for l in s if re.match(r'\s*mkdir -p "\$C"', l)]
EX = [re.sub(r"\s*\\$", "", l.strip()) for l in s if re.match(r'\s*EXTRA_ENV="[^"]*\$extra"\s*\\$', l)]
TO = re.findall(r"JOB_TIMEOUT=\$\{JOB_TIMEOUT:-(\d+)\}", "\n".join(s))
if not (len(C) == 1 and len(MK) == 1 and len(EX) == 1 and len(TO) == 1): sys.exit(f"found C={len(C)} mkdir={len(MK)} EXTRA_ENV={len(EX)} timeout={len(TO)}")
cold = "KEEP=0" in C[0]
c = re.sub(r"^local ", "", C[0]).replace(" KEEP=0", "")
open(sys.argv[2], "w").write(
    "# job environment of a timed run (taken from the scoring harness): compile caches, ports, CUDA home\n"
    f"{c}\nDEV_COLD={int(cold)}\nDEV_MKDIR='{MK[0]}'\nDEV_TIMEOUT={TO[0]}\n{EX[0]}\n")
print(("cold" if cold else "warm") + " compile caches")
PY
cp "$KIT/view/dev_bench.sh" "$O/dev_bench.sh"; chmod +x "$O/dev_bench.sh"; cp "$KIT/view/README.md" "$O/README.md"
# 4. the view's engine must be the parent's, byte for byte (so it holds none of the fix)
(cd "$O/venv/$SPREL" && sha256sum --quiet -c "$HREC") || kit_die "the view's engine differs from the parent's"
# 5. leak scan
mkdir -p "$KSTATE"; REP=$KSTATE/view_scan_$(basename "$O").txt
"$SYSPY" -I - "$O" "$ISO_ROOT" "$ENGINE" "$SPREL" "$KIT/leakterms/$PR.txt" "$PR" "$REP" <<'PY'
import json, os, re, sys
O, root, eng, sprel, softf, pr, rep = sys.argv[1:8]
be = dict(l.split("=", 1) for l in open(os.path.join(root, "env/build.env")).read().split("\n") if re.match(r"^[A-Z_a-z]+=", l))
fix = be["FIX"].split()[0]
changed = [l.strip() for l in open(os.path.join(root, "env/files_differing.txt")) if l.strip() and not l.strip().endswith("/_version.py")]
G = json.load(open(os.path.join(root, "harness/gate.json")))
refs = {str(G.get("ref"))} | {str((G.get("reference_rtx4090") or {}).get("ref"))} - {"None"}
hard = [fix, fix[:7], fix[:8], fix[:10], fix[:12]] + changed + [os.path.basename(c) for c in changed] + sorted(refs - {"None"}) + \
       ["gate.json", "files_differing", "PROVENANCE", "RUN.md", "mkvariant", "variants/", "_fix", "ref_correct", "human fix", "maintainer"]
soft = [l.strip() for l in open(softf)] if os.path.exists(softf) else []
m = re.search(r"\(([^)]*)\)", open(os.path.join(root, "RUN.md")).readline())
if m: soft.append(re.escape(m.group(1)))
venv = os.path.join(O, "venv"); engdir = os.path.join(venv, sprel, eng)
A, B = [], []   # A: text the kit wrote or copied (bench, loads, harness, docs); B: venv outside the engine + interpreter
for dp, dns, fns in os.walk(O):
    if dp.startswith(engdir) or dp.startswith(os.path.join(O, "st", "hf")) or dp.startswith(os.path.join(O, "st", "cuda")): continue
    for n in fns:
        p = os.path.join(dp, n)
        if os.path.islink(p): continue
        (B if dp.startswith(venv) or dp.startswith(os.path.join(O, "st")) else A).append(p)
def scan(files, terms, nocase=True, text_only=True, limit=2_000_000):
    hits = []
    for p in files:
        try:
            if os.path.getsize(p) > limit: continue
            b = open(p, "rb").read()
        except OSError: continue
        if text_only and b"\0" in b[:4096]: continue
        t = b.decode("utf-8", "replace"); tl = t.lower() if nocase else t
        for term in terms:
            if (term.lower() if nocase else term) in tl:
                ln = tl[:tl.find(term.lower() if nocase else term)].count("\n") + 1
                hits.append((os.path.relpath(p, O), ln, term))
    return hits
hA = scan(A, hard)
hB = scan(B, [fix, fix[:10], root, f"{pr}_fix", f"{pr}_parent"], text_only=False, limit=50_000_000)
soft_hits = []
for p in A:
    t = open(p, "rb").read().decode("utf-8", "replace")
    for s in soft:
        for mm in re.finditer(s, t, re.I):
            ln = t[:mm.start()].count("\n") + 1
            soft_hits.append((os.path.relpath(p, O), ln, s, t.split("\n")[ln - 1].strip()[:120]))
prA = scan(A, [pr]); prB = scan(B, [pr], text_only=True)
L = [f"view {O}", f"scanned: {len(A)} files written/copied by the kit (all hard terms), {len(B)} venv/interpreter files outside the engine (fix commit, package path, venv names)",
     f"hard terms ({len(hard)}): fix commit and prefixes, {len(changed)} changed file(s) and basenames, REF values {sorted(refs)}, gate/variant/RUN.md/PROVENANCE names",
     f"HARD HITS: {len(hA) + len(hB)}"] + [f"  {h}" for h in hA + hB] + \
    [f"soft terms {soft}: {len(soft_hits)} hit(s) (review)"] + [f"  {h}" for h in soft_hits] + \
    [f"PR number '{pr}' occurrences: {len(prA)} in kit-written files, {len(prB)} in venv text files"] + [f"  {h}" for h in (prA + prB)[:10]]
open(rep, "w").write("\n".join(L) + "\n"); print("\n".join(L))
sys.exit(1 if hA or hB else 0)
PY
rc=$?
[ $rc = 0 ] || kit_die "leak scan found hard terms in the view (report: $REP)"
echo "AGENT_VIEW_OK $O ($(du -sh "$O" | cut -f1), $(( $(date +%s) - t0 )) s; scan report $REP)"
