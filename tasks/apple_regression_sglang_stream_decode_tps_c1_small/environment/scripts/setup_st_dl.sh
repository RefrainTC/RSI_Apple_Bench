#!/bin/bash
# setup_st_dl.sh : builds the ST environment from pinned downloads (CPU + network only, no root).
# Why: on our 4090 server, LONG-LIVED pooled HTTPS connections stall (pip: wheels at 13-21 kB/s; uv: 8 sockets idle, host rx
# 19 KiB/s), while a FRESH connection gets 17-23 MB/s (curl: 188 MB pytorch wheel, 322 MB PyPI nccl wheel).
# So: one short-lived `pip download --no-deps <pin>` process per lock pin (fresh connections each, 12 in parallel),
# same three indexes as the Dockerfile -> pip itself picks the same wheel it would pick in `pip install`; then the
# Dockerfile's install with pip, from that local dir only (--no-index --find-links), --no-deps, same lock.
# aiperf venv: resolve once with `pip install --dry-run --report`, then the same per-pin download + local install.
# Then check_lock.py (= the Dockerfile's post-install check), pip check, engine source, models. CPU + network only.
set -uo pipefail
# Step 0 = fetch_base.sh (pinned uv, CPython 3.12.13, CUDA 12.8 redist, Apple repo @1a38072f, all under $RSI_ROOT).
# Roots: RSI_ROOT (required); RSI_AIPERF_PINS (optional: install these exact aiperf pins instead of resolving); RSI_TASK_DIR (default $RSI_ROOT/apple/tasks/regression_sglang_stream_decode_tps_c1).
# Last step builds the harnesses: Apple tests/ + make_harness.py (edits 1-3) -> st/harness, + make_harness_nl.py (edit 4)
# -> st/harness_nl, + make_harness_sc.py (edit 5) -> st/harness_sc (the one to score with).
ROOTDIR=${RSI_ROOT:?set RSI_ROOT}; D=$ROOTDIR/st; PY=${RSI_PY:-$ROOTDIR/uvpy/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12}
TASK=${RSI_TASK_DIR:-$ROOTDIR/apple/tasks/regression_sglang_stream_decode_tps_c1}; S="$(cd "$(dirname "$0")" && pwd)"
bash $S/fetch_base.sh || { echo "fetch_base.sh failed"; exit 1; }
mkdir -p $D/logs/dl $D/hf $D/tmp $D/wheels $D/wheels_aiperf
export TMPDIR=$D/tmp PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_RETRIES=10 PIP_TIMEOUT=60 PIP_NO_CACHE_DIR=1
say() { echo "== $(date '+%F %T') $*"; }
IDX="--index-url https://pypi.org/simple --extra-index-url https://docs.sglang.ai/whl/cu129/ --extra-index-url https://download.pytorch.org/whl/cu128"
fetch() {  # fetch <venv python> <wheel dir> <pins file> <index args...>
  local pins=$3
  export vpy=$1 wd=$2 LOGD=$D/logs/dl; shift 3; export IDXARGS="$*"
  rm -f $wd.fail
  xargs -P 12 -n 1 -a "$pins" bash $S/fetch_one.sh
  [ ! -s $wd.fail ]
}
if [ ! -f $D/.venv_ok ]; then
  say "1. engine venv: per-pin download"; rm -rf $D/venv; $PY -m venv $D/venv || exit 1
  grep -E '^[A-Za-z0-9._-]+\s*={2,3}' $TASK/environment/requirements.lock | sed 's/ #.*//; s/[[:space:]]//g' > $D/logs/pins.txt
  fetch $D/venv/bin/python $D/wheels $D/logs/pins.txt $IDX || { say "downloads failed:"; cat $D/wheels.fail; }
  say "   $(ls $D/wheels | wc -l) files, $(du -sh $D/wheels | cut -f1); installing with pip from the local dir"
  $D/venv/bin/python -m pip install --no-deps --no-index --find-links $D/wheels -r $TASK/environment/requirements.lock > $D/logs/pip_install_local.log 2>&1 \
    || { say "local install failed"; tail -20 $D/logs/pip_install_local.log; }
  $D/venv/bin/python -m pip check > $D/logs/pip_check.txt 2>&1
  $D/venv/bin/python -I $S/check_lock.py $TASK/environment/requirements.lock | tee $D/logs/check_lock.txt
  $D/venv/bin/python -m pip freeze > $D/logs/venv_freeze.txt 2>&1
  grep -q "missing \[\], wrong \[\]" $D/logs/check_lock.txt && grep -q "sglang installed as dist: False" $D/logs/check_lock.txt && touch $D/.venv_ok
fi
if [ ! -f $D/.aiperf_ok ]; then
  say "2. aiperf venv (resolved now, as the Dockerfile does at build time)"; rm -rf $D/aiperf-venv; $PY -m venv $D/aiperf-venv
  if [ -n "${RSI_AIPERF_PINS:-}" ]; then   # 5090 run: the resolver stalled, so the 4090 run's exact freeze was used as the pins
    cp "$RSI_AIPERF_PINS" $D/logs/aiperf_pins.txt
  else
    $D/aiperf-venv/bin/python -m pip install --dry-run --ignore-installed --quiet --report $D/logs/aiperf_report.json "aiperf==0.12.0" > $D/logs/aiperf_resolve.log 2>&1
    $D/aiperf-venv/bin/python -I -c "import json,sys; r=json.load(open(sys.argv[1])); print('\n'.join(f\"{i['metadata']['name']}=={i['metadata']['version']}\" for i in r['install']))" $D/logs/aiperf_report.json > $D/logs/aiperf_pins.txt
  fi
  fetch $D/aiperf-venv/bin/python $D/wheels_aiperf $D/logs/aiperf_pins.txt --index-url https://pypi.org/simple || { say "aiperf downloads failed"; cat $D/wheels_aiperf.fail; }
  $D/aiperf-venv/bin/python -m pip install --no-deps --no-index --find-links $D/wheels_aiperf -r $D/logs/aiperf_pins.txt > $D/logs/aiperf_install.log 2>&1
  $D/aiperf-venv/bin/python -c "import importlib.metadata as m; assert m.version('aiperf')=='0.12.0'" && $D/aiperf-venv/bin/aiperf --version > $D/logs/aiperf_version.txt 2>&1 && touch $D/.aiperf_ok
  $D/aiperf-venv/bin/python -m pip freeze > $D/logs/aiperf_freeze.txt 2>&1
fi
if [ ! -f $D/.src_ok ]; then
  say "3. engine source 163b970"; rm -rf $D/sglang; mkdir -p $D/sglang; cd $D/sglang
  git init -q . && git remote add origin https://github.com/sgl-project/sglang.git \
   && timeout 1200 git fetch -q --depth 1 origin 163b97012707d712e141e7ee486b72e2b0d8e60d && git checkout -q FETCH_HEAD \
   && [ "$(git rev-parse HEAD)" = 163b97012707d712e141e7ee486b72e2b0d8e60d ] && touch $D/.src_ok
fi
if [ ! -f $D/.model_ok ] && [ -z "${RSI_SKIP_GPTOSS:-}" ]; then  # gpt-oss-20b = load probe only; RSI_SKIP_GPTOSS=1 skips it
  say "4. model openai/gpt-oss-20b@6cee5e81 (fresh from HF)"
  HF_HOME=$D/hf timeout 3600 $D/venv/bin/hf download openai/gpt-oss-20b --revision 6cee5e81ee83917806bbde320786a8fb61efebee --exclude "original/*" --exclude "metal/*" > $D/logs/hf_download.log 2>&1 \
    && touch $D/.model_ok
  tail -2 $D/logs/hf_download.log
fi
if [ ! -f $D/.qwen_ok ]; then
  say "5. substitute model Qwen/Qwen2.5-1.5B-Instruct@989aa798 (fresh from HF)"
  HF_HOME=$D/hf timeout 1800 $D/venv/bin/hf download Qwen/Qwen2.5-1.5B-Instruct --revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 > $D/logs/hf_qwen25.log 2>&1 \
    && touch $D/.qwen_ok
  tail -1 $D/logs/hf_qwen25.log
fi
say "6. harness: Apple tests + edits 1-3 (harness), + edit 4 (harness_nl), + edit 5 (harness_sc)"
$D/venv/bin/python -I $S/make_harness.py $TASK/tests $D/harness && $D/venv/bin/python -I $S/make_harness_nl.py $D/harness $D/harness_nl \
  && $D/venv/bin/python -I $S/make_harness_sc.py $D/harness_nl $D/harness_sc && touch $D/.harness_ok
say "SETUP_ST_DL DONE venv=$([ -f $D/.venv_ok ] && echo ok) aiperf=$([ -f $D/.aiperf_ok ] && echo ok) src=$([ -f $D/.src_ok ] && echo ok) gptoss=$([ -f $D/.model_ok ] && echo ok) qwen=$([ -f $D/.qwen_ok ] && echo ok) harness=$([ -f $D/.harness_ok ] && echo ok)"
