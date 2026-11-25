#!/bin/bash
# integrity.sh <make|verify> <taskdir> [phase] : sha256 manifest of everything a score depends on (review F4).
#   scorer   : harness/ (incl. audit/; not its *.md docs, not NFS .nfs* placeholders of replaced-but-open files), scripts/p2_job.sh, scripts/p2_cpus.sh, the task's *.py / *.sh / task.env /
#              instruction.md / WORKLOAD.md (not README.md: operator notes), pristine.sha256, the CPU pinning table
#   ref      : calib/{calib.json, correct_ref.json, out_ref*.json, oracle_stage/**} (once they exist)
#   venv     : every file of the staged master venvs (VENV, FIX_VENV), .so / .pyc included, except pyvenv.cfg
#              (rewritten by relocation; checked separately for its home line)
# sealed (2026-11-23, harness/vault.sh): while $TD/SEALED exists the reference files and the human fix are not on the
#          pod, so verify checks scorer + venv_parent + the vault archive's sha256; the ref manifest is checked by
#          `verify-ref` after score_phase2.sh restored the reference files (oracle_stage/ entries excluded).
# make   : writes $TD/manifest/{scorer,ref,venv_parent,venv_fix}.sha256 and prints the sha256 of each manifest file. These
#          hashes are recorded OFF the pod (raw/P2-port-4090/manifests/ on the workstation); the pod copy is only used to
#          find what changed. Run make after a calibration, before any candidate code runs.
# verify : recomputes and compares; writes $TD/manifest/verify_<phase>.json; exit 1 on any difference. score.sh and
#          calibrate.sh call it at start and end; the sha256 of each manifest is copied into the summary, so the
#          workstation can check it against its own record.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
OP=$1; TD=$(readlink -f "$2"); PHASE=${3:-manual}; P2=$ISO_ROOT; H=$P2/harness_iso12380
source "$TD/task.env"
MD=$TD/manifest; mkdir -p "$MD"
# hashing ~13 GB of venvs is CPU- and memory-heavy: stay on the task GPU's own CCD (pinning table), 8 workers, low
# priority, so it cannot disturb a timed run on another GPU (it used to run on all cores with 16 workers)
if [ -z "${P2_INTEGRITY_PINNED:-}" ]; then
  C=$(bash $H/p2_cpus.sh "$GPU") || exit 2
  P2_INTEGRITY_PINNED=1 exec taskset -c "$C" nice -n 10 bash "$0" "$@"
fi
hashlist() { (cd "$1" && shift && find "$@" -type f ! -name pyvenv.cfg ! -path '*/__pycache__/*.pyc.*' -print0 2>/dev/null | LC_ALL=C sort -z | xargs -0 -r -P 8 -n 300 sha256sum | LC_ALL=C sort -k2); }
scorer() {
  { (cd $P2 && find harness_iso12380 -type f ! -name '*.pyc' ! -name '*.md' ! -name '.nfs*' ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum)
    (cd "$TD" && find . -maxdepth 1 -type f \( -name '*.py' -o -name '*.sh' -o -name task.env -o -name '*.md' -o -name pristine.sha256 \) ! -name README.md -print0 | LC_ALL=C sort -z | xargs -0 sha256sum)
    true; } | LC_ALL=C sort -k2   # iso12380 copy: job runner, CPU table (cpu_pinning_iso12380.txt) are inside harness_iso12380/
}
ref() { [ -d "$TD/calib" ] && (cd "$TD/calib" && find . -maxdepth 1 -type f \( -name calib.json -o -name correct_ref.json -o -name 'out_ref*.json' \) -print0; find oracle_stage -type f -print0 2>/dev/null) | (cd "$TD/calib" && LC_ALL=C sort -z | xargs -0 -r sha256sum) | LC_ALL=C sort -k2; }
gen() {  # <name> -> stdout
  case $1 in scorer) scorer;; ref) ref;; venv_parent) hashlist "$VENV" .;; venv_fix) hashlist "$FIX_VENV" .;; esac
}
NAMES="scorer ref venv_parent venv_fix"; SEALED=0; [ -e "$TD/SEALED" ] && SEALED=1
if [ "$OP" = verify-ref ]; then
  d=$(diff <(ref | grep -v ' oracle_stage/') <(grep -v ' oracle_stage/' "$MD/ref.sha256") | grep -c '^[<>]')
  out="{\"phase\":\"$PHASE\",\"time\":\"$(date -Is)\",\"ref\":{\"manifest_sha256\":\"$(sha256sum < "$MD/ref.sha256" | cut -c1-64)\",\"diff_lines\":$d,\"oracle_stage_excluded\":true},\"ok\":$([ "$d" = 0 ] && echo true || echo false)}"
  echo "$out" > "$MD/verify_$PHASE.json"; echo "$out"; [ "$d" = 0 ]; exit
fi
[ "$OP" = make ] && [ $SEALED = 1 ] && { echo "integrity: task is sealed, unseal before make"; exit 2; }
[ $SEALED = 1 ] && NAMES="scorer venv_parent"
if [ "$OP" = make ]; then
  for n in $NAMES; do gen $n > "$MD/$n.sha256"; echo "$n $(wc -l < "$MD/$n.sha256") files manifest_sha256=$(sha256sum < "$MD/$n.sha256" | cut -c1-64)"; done
  grep -h '^home = ' "$VENV/pyvenv.cfg" "$FIX_VENV/pyvenv.cfg" > "$MD/pyvenv_home.txt"
  date -Is > "$MD/made_at.txt"; exit 0
fi
ok=1; out="{\"phase\":\"$PHASE\",\"time\":\"$(date -Is)\""
for n in $NAMES; do
  [ -f "$MD/$n.sha256" ] || { out="$out,\"$n\":\"no manifest\""; ok=0; continue; }
  d=$(diff <(gen $n) "$MD/$n.sha256" | grep -c '^[<>]')
  out="$out,\"$n\":{\"manifest_sha256\":\"$(sha256sum < "$MD/$n.sha256" | cut -c1-64)\",\"diff_lines\":$d}"
  if [ "$d" != 0 ]; then ok=0; diff <(gen $n) "$MD/$n.sha256" | head -40 > "$MD/verify_${PHASE}_$n.diff"; fi
done
if [ $SEALED = 1 ]; then
  grep -qxF "$(grep -h '^home = ' "$VENV/pyvenv.cfg")" "$MD/pyvenv_home.txt" || ok=0
  vs=$(sha256sum "$P2/vault/$TASK.tar.enc" 2>/dev/null | cut -c1-64); want=$(python3 -I -c "import json,sys;print(json.load(open(sys.argv[1]))['sha256'])" "$TD/SEALED")
  out="$out,\"sealed\":{\"vault_sha256\":\"$vs\",\"match\":$([ "$vs" = "$want" ] && echo true || echo false)}"
  [ "$vs" = "$want" ] || ok=0
else
  diff <(grep -h '^home = ' "$VENV/pyvenv.cfg" "$FIX_VENV/pyvenv.cfg") "$MD/pyvenv_home.txt" > /dev/null || ok=0
fi
out="$out,\"ok\":$([ $ok = 1 ] && echo true || echo false)}"
echo "$out" > "$MD/verify_$PHASE.json"; echo "$out"
[ $ok = 1 ]
