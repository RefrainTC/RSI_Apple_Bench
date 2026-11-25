#!/bin/bash
# mk_overlay.sh <site-packages dir> <pkg> <stage dir|-> <out root>
# Builds an import overlay: <out root>/<pkg> is a symlink tree of the pristine package (real dirs, file symlinks),
# with the stage's changed/added files copied in as real files and its deleted files removed.
# Stage layout = kernel_gate.py --stage output: <stage>/repo_overlay/<pkg>/..., <stage>/repo_deleted.txt (paths rel. to repo/).
set -euo pipefail
SITE=$1; PKG=$2; STAGE=$3; OUT=$4
rm -rf "$OUT"; mkdir -p "$OUT"
cp -rs "$SITE/$PKG" "$OUT/$PKG"
n=0
if [ "$STAGE" != "-" ]; then
  if [ -d "$STAGE/repo_overlay" ]; then
    while IFS= read -r -d '' f; do
      rel=${f#"$STAGE/repo_overlay/"}
      case "$rel" in "$PKG"/*) ;; *) echo "[mk_overlay] skip path outside $PKG: $rel" >&2; continue;; esac
      t="$OUT/$rel"; mkdir -p "$(dirname "$t")"; rm -f "$t"; cp "$f" "$t"
      b=$(basename "$rel" .py); rm -f "$(dirname "$t")/__pycache__/$b".*.pyc
      n=$((n+1))
    done < <(find "$STAGE/repo_overlay" -type f -print0)
  fi
  if [ -s "$STAGE/repo_deleted.txt" ]; then
    while IFS= read -r rel; do [ -n "$rel" ] && rm -f "$OUT/$rel" && n=$((n+1)); done < "$STAGE/repo_deleted.txt"
  fi
fi
echo "[mk_overlay] $OUT ($n changed paths)"
