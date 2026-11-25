#!/bin/bash
# get_model.sh : download the model at the pinned revision into $ISO_ROOT/hf (HF hub cache layout) and check every LFS
# blob's sha256. Needs the parent venv (huggingface_hub). Retries with back-off: huggingface.co rate-limits anonymous
# downloads (HTTP 429); setting HF_TOKEN avoids it.
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/../env.sh"; . "$ISO_ROOT/env/build.env"
export HF_HOME=$ISO_ROOT/hf HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_ENABLE_HF_TRANSFER=0
PY=$ISO_ROOT/venvs/${PR}_parent/bin/python3; [ -x "$PY" ] || { echo "build the venvs first (setup/setup_lock.sh)"; exit 1; }
fail() { echo "GET_MODEL FAILED: $*"; exit 1; }
t0=$(date +%s)
for i in 1 2 3 4 5 6 7 8 9 10; do
  (cd $ISO_ROOT && "$PY" -I -c "import sys; from huggingface_hub import snapshot_download as s; print(s(sys.argv[1], revision=sys.argv[2]))" "$MODEL" "$MODEL_REV") && break
  [ $i = 10 ] && fail download; echo "download attempt $i failed (HTTP 429 = rate limit); retrying in $((30 * i)) s"; sleep $((30 * i))
done
H=$HF_HOME/hub/models--${MODEL//\//--}
mkdir -p $H/refs; printf '%s' "$MODEL_REV" > $H/refs/main   # the engine resolves the model offline through refs/main
for b in $H/blobs/*; do n=$(basename $b)
  if [[ $n =~ ^[0-9a-f]{64}$ ]]; then [ "$(sha256sum < $b | cut -c1-64)" = "$n" ] || fail "blob $n corrupt"; fi; done
for x in $MODEL_LFS; do [ -f $H/blobs/$x ] || fail "missing LFS blob $x"; done
echo "model ok: $MODEL @ $MODEL_REV, $(du -shL $H | cut -f1), $(( $(date +%s) - t0 )) s"
echo GET_MODEL_OK
