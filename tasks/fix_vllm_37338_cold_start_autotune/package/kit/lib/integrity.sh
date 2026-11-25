# integrity.sh : sourced after common.sh. Seal, verify and snapshot the trees the scoring trusts.
#   seal_parent    byte-compile the parent venv and the staged interpreter once (so runs never write bytecode into
#                  them), then record the sha256 of every file of the trusted trees in $KSTATE/seal.sha256. Refuses
#                  unless the parent's engine files hash as the setup recorded. Trust-on-first-use: run it right after
#                  calibration (test.sh runs it if the seal is missing).
#   verify_seal    every file of the trusted trees hashes as sealed and no file was added or removed.
_hash_trees() {  # -> "sha256  path" lines, sorted, for every regular file and symlink target of the trusted trees
  local d
  for d in $(trusted_trees); do
    find "$d" -type f -print0
  done | LC_ALL=C sort -z | taskset -c "${KIT_AUX_CPUS:-0-$(( $(nproc --all) - 1 ))}" xargs -0 -n 256 -P 8 sha256sum | LC_ALL=C sort -k2
  for d in $(trusted_trees); do find "$d" -type l -printf 'link:%l  %p\n'; done | LC_ALL=C sort -k2
}
seal_parent() {
  local r t0=$(date +%s); mkdir -p "$KSTATE"
  r=$(verify_parent_engine) || { echo "$r"; return 1; }
  kit_log "sealing: byte-compiling the parent venv and the staged interpreter (idempotent)"
  "$PV/bin/python3" -I -m compileall -q -j 8 "$PV/$SPREL" > "$KSTATE/compileall_parent.log" 2>&1
  local pyd; for pyd in "$STAGE"/py/*/lib/python3.10; do
    "$PV/bin/python3" -I -m compileall -q -j 8 "$pyd" >> "$KSTATE/compileall_parent.log" 2>&1; done
  r=$(verify_parent_engine) || { echo "$r"; return 1; }
  _hash_trees > "$KSTATE/seal.sha256.tmp" && mv "$KSTATE/seal.sha256.tmp" "$KSTATE/seal.sha256"
  echo "$(date -u -Is) $(trusted_trees | tr '\n' ' ')" > "$KSTATE/seal.info"
  echo "sealed $(wc -l < "$KSTATE/seal.sha256") files of $(trusted_trees | wc -l) trusted trees in $(( $(date +%s) - t0 )) s"
}
verify_seal() {
  [ -f "$KSTATE/seal.sha256" ] || { echo "no seal ($KSTATE/seal.sha256)"; return 1; }
  local t0=$(date +%s) d
  d=$(diff <(_hash_trees) "$KSTATE/seal.sha256" | grep '^[<>]' | head -8)
  [ -z "$d" ] || { echo "TRUSTED TREES DIFFER FROM THE SEAL: $(echo "$d" | cut -c1-200 | tr '\n' ';')"; return 1; }
  echo "seal ok: $(wc -l < "$KSTATE/seal.sha256") files unchanged ($(( $(date +%s) - t0 )) s)"
}
