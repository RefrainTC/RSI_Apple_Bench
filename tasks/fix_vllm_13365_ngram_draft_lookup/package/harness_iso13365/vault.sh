#!/bin/bash
# vault.sh <seal|open-refs|close-refs|unseal|status> <taskdir> : keeps the human fix and the reference outputs out of
# the pod's plain filesystem while a trial runs (leakage review 2026-11-23: candidate code runs as `coder`, with sudo and
# no container, so anything readable on the pod is readable by it).
#   seal       (key on stdin) : tar every path in $TD/vault_paths.txt (one absolute path per line; must lie under
#                p2-port, t5-isobench or $P2T; typically calib/, results/, staging/, remote_in|out/, the smoke
#                workspaces and runs built from the fix, exact copies of the fix's files found by
#                scripts/leak_surface.py), encrypt it (AES-256-CBC, PBKDF2, key = first stdin line) to
#                $P2/vault/<task>.tar.enc, check the archive decrypts and lists every path, then delete the plaintext and
#                write $TD/SEALED. Prints the archive sha256 (record it off the pod).
#   open-refs  (key on stdin) : restore only calib/{calib.json, correct_ref.json, out_ref.json, out_ref_lp.json}, for
#                the analysis phase of a score (score_phase2.sh), when no candidate process is alive.
#   close-refs : delete those four files again.
#   unseal     (key on stdin) : restore everything (after the trial), remove SEALED.
# The key never touches the pod's disk: it is generated and kept on the workstation (raw/P2-port-4090/vault/) and sent
# over ssh stdin only for open-refs / unseal.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
OP=$1; TD=$(readlink -f "$2"); P2=$ISO_ROOT; T5=$ISO_ROOT
source "$TD/task.env"
VF=$P2/vault/$TASK.tar.enc; mkdir -p $P2/vault; chmod 700 $P2/vault
REFS="$TD/calib/calib.json $TD/calib/correct_ref.json $TD/calib/out_ref.json $TD/calib/out_ref_lp.json"
key() { IFS= read -r KEY || true; [ ${#KEY} -ge 32 ] || { echo "vault: key missing or too short (stdin)"; exit 2; }; }
dec() { openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -pass fd:3 -in "$VF" 3< <(printf '%s\n' "$KEY"); }
case "$OP" in
  seal)
    [ -e "$TD/SEALED" ] && { echo "vault: already sealed"; exit 2; }
    [ -s "$TD/vault_paths.txt" ] || { echo "vault: no $TD/vault_paths.txt"; exit 2; }
    key; L=$(mktemp); : > "$L"
    while IFS= read -r p; do
      [ -z "$p" ] && continue
      case "$p" in $ISO_ROOT/*|$ISO_ROOT/*|$P2T/*) ;; *) echo "vault: refusing path outside our trees: $p"; exit 2;; esac
      [ -e "$p" ] && echo "$p" >> "$L"
    done < "$TD/vault_paths.txt"
    tar -cPf - --files-from="$L" | openssl enc -e -aes-256-cbc -pbkdf2 -iter 200000 -salt -pass fd:3 -out "$VF.tmp" 3< <(printf '%s\n' "$KEY") \
      || { echo "vault: archive failed"; rm -f "$VF.tmp" "$L"; exit 1; }
    mv "$VF.tmp" "$VF"
    n=$(dec | tar -tPf - | sed 's#/$##' | sort -u | grep -cFxf "$L"); m=$(wc -l < "$L")
    [ "$n" = "$m" ] || { echo "vault: verification failed ($n of $m top-level paths listed); nothing deleted"; exit 1; }
    while IFS= read -r p; do chmod -R u+w "$p" 2>/dev/null; rm -rf "$p"; done < "$L"
    S=$(sha256sum "$VF" | cut -c1-64)
    printf '{"sealed_at":"%s","archive":"%s","sha256":"%s","paths":%s}\n' "$(date -Is)" "$VF" "$S" "$m" > "$TD/SEALED"
    cp "$L" "$P2/vault/$TASK.paths"; rm -f "$L"; echo "SEALED $TASK paths=$m archive_sha256=$S" ;;
  open-refs)
    [ -e "$TD/SEALED" ] || { echo "vault: not sealed (refs are in place)"; exit 0; }
    key; dec | tar -xPf - $REFS 2>/dev/null; [ -f "$TD/calib/calib.json" ] || { echo "vault: open-refs failed"; exit 1; }
    echo "REFS OPEN $TASK" ;;
  close-refs)
    [ -e "$TD/SEALED" ] || exit 0
    rm -f $REFS; echo "REFS CLOSED $TASK" ;;
  unseal)
    [ -e "$TD/SEALED" ] || { echo "vault: not sealed"; exit 0; }
    key; dec | tar -xPf - || { echo "vault: unseal failed"; exit 1; }
    mv "$TD/SEALED" "$P2/vault/$TASK.unsealed_$(date +%Y%m%d%H%M%S).json"; echo "UNSEALED $TASK" ;;
  status)
    if [ -e "$TD/SEALED" ]; then cat "$TD/SEALED"; ls -l $REFS 2>/dev/null | sed 's/^/open ref: /'; else echo "not sealed"; fi ;;
  *) echo "usage: vault.sh <seal|open-refs|close-refs|unseal|status> <taskdir>"; exit 2 ;;
esac
