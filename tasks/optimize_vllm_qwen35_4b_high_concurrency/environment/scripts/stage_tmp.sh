#!/usr/bin/env bash
# stage_tmp.sh: copy re-creatable inputs from $P3_ROOT to local NVMe ($P3_STAGE), once, sequentially.
# Idempotent (marker per item). /tmp may be wiped on reboot -> every scoring job calls this first.
# Staged: uv python + venv (pyvenv.cfg re-pointed), model weights. Results/logs stay under $P3_ROOT.
set -euo pipefail
# v5d: P3_ROOT / P3_STAGE move the NAS root / the local staging root (default /tmp/taskc_stage)
P=${P3_ROOT:?set P3_ROOT}; T=${P3_STAGE:-/tmp/taskc_stage}; P=${P%/}; T=${T%/}
mkdir -p $T/models $T/run
exec 9>$T/.stage.lock; flock 9
stage_dir() { # src dst-parent name
  local src=$1 dstp=$2 name=$3
  if [ -f $dstp/.$name.ok ] && [ -d $dstp/$name ]; then return 0; fi
  rm -rf $dstp/$name; t0=$(date +%s)
  tar -C $(dirname $src) -cf - $(basename $src) | tar -C $dstp -xf -
  [ "$(basename $src)" = "$name" ] || mv $dstp/$(basename $src) $dstp/$name
  echo "$(date +%FT%T) staged $src -> $dstp/$name in $(( $(date +%s) - t0 ))s ($(du -sh $dstp/$name | cut -f1))" | tee -a $T/STAGE_LOG
  touch $dstp/.$name.ok
}
stage_dir $P/uv-python $T uv-python
stage_dir $P/venv $T venv
# re-point the venv at the staged interpreter
if ! grep -q "^home = $T/uv-python" $T/venv/pyvenv.cfg; then
  sed -i "s#$P/uv-python#$T/uv-python#g" $T/venv/pyvenv.cfg
  for f in $T/venv/bin/*; do [ -f "$f" ] && head -c 2 "$f" | grep -q '#!' && sed -i "1s#$P/venv#$T/venv#" "$f" || true; done
  ln -sfn $(ls -d $T/uv-python/cpython-3.12*/bin/python3.12 | head -1) $T/venv/bin/python 2>/dev/null || true
fi
# models that were not downloaded (e.g. only Qwen3.5-4B for tasks C/B5) are skipped
for m in gpt-oss-20b qwen3.5-4b; do [ -d $P/src/models/$m ] || { echo "skip $m (not downloaded)"; continue; }; stage_dir $P/src/models/$m $T/models $m; done
$T/venv/bin/python -I -c "import sys, vllm, torch; assert sys.prefix=='$T/venv', sys.prefix; print('staged venv ok', vllm.__version__, torch.__version__, vllm.__file__)"
du -sh $T 2>/dev/null | tee -a $T/STAGE_LOG
