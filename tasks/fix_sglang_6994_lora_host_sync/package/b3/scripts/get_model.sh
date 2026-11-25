#!/bin/bash
# get_model.sh : download the model at the pinned revision into $ISO_ROOT/b3/hf (HF hub cache layout) and check every LFS
# blob's sha256; for #6994 also make the two random-weight LoRA adapters and check them. Package addition (the authors
# downloaded with b3/scripts/dl.sh and ran make_lora.py by hand). Needs the parent venv (huggingface_hub, torch).
set -uo pipefail
. "$(dirname "$(readlink -f "$0")")/../../env.sh"; . "$ISO_ROOT/env/build.env"
export HF_HOME=$ISO_ROOT/b3/hf HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_ENABLE_HF_TRANSFER=0
PY=$ISO_ROOT/b3/venvs/${PR}_parent/bin/python3; [ -x "$PY" ] || { echo "build the venvs first (setup_lock.sh)"; exit 1; }
fail() { echo "GET_MODEL FAILED: $*"; exit 1; }
t0=$(date +%s)
# retried with back-off: huggingface.co rate-limits anonymous downloads (HTTP 429); setting HF_TOKEN avoids it
for i in 1 2 3 4 5 6; do
  "$PY" -I -c "import sys; from huggingface_hub import snapshot_download as s; p=sys.argv[3].split() or None; print(s(sys.argv[1], revision=sys.argv[2], allow_patterns=p))" "$MODEL" "$MODEL_REV" "$MODEL_PATTERNS" && break
  [ $i = 6 ] && fail download; echo "download attempt $i failed; retrying in $((30 * i)) s"; sleep $((30 * i))
done
H=$HF_HOME/hub/models--${MODEL//\//--}
# the engines resolve the model offline through refs/main (as in the authors' cache, where it holds this revision)
mkdir -p $H/refs; printf '%s' "$MODEL_REV" > $H/refs/main
for b in $H/blobs/*; do n=$(basename $b)
  if [[ $n =~ ^[0-9a-f]{64}$ ]]; then [ "$(sha256sum < $b | cut -c1-64)" = "$n" ] || fail "blob $n corrupt"; fi; done
for x in $MODEL_LFS; do [ -f $H/blobs/$x ] || fail "missing LFS blob $x"; done
echo "model ok: $MODEL @ $MODEL_REV, $(du -sh $H | cut -f1), $(( $(date +%s) - t0 )) s"
for spec in ${LORA_SPECS:-}; do IFS=: read n seed sha <<< "$spec"
  "$PY" -I $ISO_ROOT/b3/bench/make_lora.py $ISO_ROOT/b3/lora/$n $LORA_ARGS $seed || fail "make_lora $n"
  [ "$(sha256sum < $ISO_ROOT/b3/lora/$n/adapter_model.safetensors | cut -c1-64)" = "$sha" ] || fail "LoRA $n differs from the authors'"
  echo "lora ok: $n (seed $seed) sha256 matches the authors'"; done
echo GET_MODEL_OK
