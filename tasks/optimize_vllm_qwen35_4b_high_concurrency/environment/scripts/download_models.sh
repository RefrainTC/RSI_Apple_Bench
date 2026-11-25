#!/usr/bin/env bash
# download ungated models into their own src/ dirs (untrusted data). Uses a tiny separate venv.
set -euo pipefail
P=${P3_ROOT:?set P3_ROOT}
QWEN35_4B_REV=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a  # the Qwen3.5-4B snapshot the references were made with
export UV_CACHE_DIR=$P/uv-cache UV_PYTHON_INSTALL_DIR=$P/uv-python HF_HOME=$P/hf-home HF_HUB_DISABLE_TELEMETRY=1
cd $P
if [ ! -x $P/dlvenv/bin/hf ]; then
  uv python install 3.12
  uv venv --python 3.12 $P/dlvenv
  uv pip install --python $P/dlvenv/bin/python "huggingface_hub[hf_xet]" 2>&1 | tail -2
fi
dl() { # repo dir revision extra-args...   (revision "" = the repo's current head)
  local repo=$1 dir=$2 rev=$3; shift 3
  mkdir -p $P/src/models/$dir
  $P/dlvenv/bin/hf download $repo ${rev:+--revision "$rev"} --local-dir $P/src/models/$dir "$@" 2>&1 | tail -3
  # the commit actually downloaded: the pinned revision, else the head at this moment
  $P/dlvenv/bin/python -I -c "import huggingface_hub as h; print('$repo', h.model_info('$repo', revision='$rev' or None).sha)" > $P/src/models/$dir.REVISION
  du -sh $P/src/models/$dir
}
dl Qwen/Qwen3.5-4B qwen3.5-4b "$QWEN35_4B_REV"
echo DL_DONE
