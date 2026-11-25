#!/bin/bash
# get_model.sh : download the task's model at the authors' exact revision into $ISO_ROOT/hf (HF hub cache layout), check
# every LFS blob against its sha256 name, and point refs/main at that revision (jobs run with HF_HUB_OFFLINE=1 and resolve
# the repo id through refs/main). Uses huggingface_hub from the parent venv (run setup_venvs.sh first). Public repo, no token.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/../env.sh"
. "$ISO_ROOT/env/pkg.conf"
export HF_HOME=$ISO_ROOT/hf HF_HUB_DISABLE_TELEMETRY=1
t0=$(date +%s)
# huggingface.co rate-limits anonymous IPs (HTTP 429); retry with back-off. Set HF_TOKEN in your shell to avoid it.
ok=0; for i in 1 2 3 4 5 6 7 8 9 10; do
  "$ISO_ROOT/venvs/${PR}_parent/bin/python3" -I -c "
import sys; from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "$MODEL" "$MODEL_REV" 2> "$ISO_ROOT/setup/get_model.err" && { ok=1; break; }
  tail -1 "$ISO_ROOT/setup/get_model.err"; w=$(( i * 60 )); [ $w -gt 300 ] && w=300; echo "attempt $i failed; retry in $w s"; sleep $w
done
[ $ok = 1 ] || { echo "download failed (see setup/get_model.err)"; exit 1; }
M=$HF_HOME/hub/models--${MODEL//\//--}
mkdir -p "$M/refs"; printf '%s' "$MODEL_REV" > "$M/refs/main"
bad=0; for b in "$M"/blobs/*; do n=$(basename "$b"); [ ${#n} = 64 ] || continue
  [ "$(sha256sum < "$b" | cut -c1-64)" = "$n" ] || { echo "BLOB MISMATCH $n"; bad=1; }; done
echo "model $MODEL@$MODEL_REV: $(ls "$M/snapshots/$MODEL_REV" | wc -l) files, $(du -shL "$M/snapshots/$MODEL_REV" | cut -f1), LFS blobs ok=$([ $bad = 0 ] && echo yes || echo NO), $(( $(date +%s) - t0 )) s"
exit $bad
