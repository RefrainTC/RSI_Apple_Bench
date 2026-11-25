#!/bin/bash
# setup.sh : T5 ISO-Bench environments (CPU / network only). One venv per commit (parent / fix) per task.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -uo pipefail
R=$ISO_ROOT; T5=$R/t5-isobench
export UV_PYTHON_INSTALL_DIR=$R/.uv-python UV_PYTHON_BIN_DIR=$R/.uv-bin UV_CACHE_DIR=$R/.uv-cache
UV=$HOME/.local/bin/uv; SYSPY=$HOST_PY_BIN/python3
PY=$($UV python find 3.10)
declare -A P F X
P[16436]=f728ab8e3578c22b42ed53e51b5e8ec35328d8b9; F[16436]=299ebb62b269ce167eb1c71b5e39a1dc1f65ce1c; X[16436]=2025-04-23
P[7822]=029c71de11bc3bcf84a1b3cf9d91e79ab6949799;  F[7822]=2deb029d115dadd012ce5ea70487a207cb025493;  X[7822]=2024-08-28
P[10733]=8c1e77fb585c4f42783a3d88c1efc7c9e15fd89f; F[10733]=98f47f2a4032f8c395268de80858c64ffcfc60fa; X[10733]=2024-11-30
cd $T5
for pr in 16436 10733 7822; do
  echo "=== PR $pr $(date +%T)"
  for role in parent fix; do
    c=${P[$pr]}; [ $role = fix ] && c=${F[$pr]}
    W=$(ls wheels/${pr}_$role/vllm-*.whl 2>/dev/null | head -1)
    [ -n "$W" ] || W=$($SYSPY -I scripts/fetch.py $pr $role $c wheels/${pr}_$role) || { echo "FETCH FAILED $pr $role"; continue 2; }
    cat wheels/${pr}_$role/SOURCE.txt
  done
  WP=$(ls wheels/${pr}_parent/vllm-*.whl); WF=$(ls wheels/${pr}_fix/vllm-*.whl)
  for role in parent fix; do rm -rf venvs/${pr}_$role; $UV venv -q --python $PY venvs/${pr}_$role; done
  # parent: resolve vLLM's own requirements against PyPI as of the merge date (+2 days)
  VIRTUAL_ENV=$T5/venvs/${pr}_parent $UV pip install -q --exclude-newer ${X[$pr]} "$WP" || { echo "INSTALL FAILED ${pr}_parent"; continue; }
  VIRTUAL_ENV=$T5/venvs/${pr}_parent $UV pip freeze | grep -v -i "^vllm" > results/${pr}_constraints.txt
  # fix: exactly the same dependency set, only the vLLM wheel differs
  VIRTUAL_ENV=$T5/venvs/${pr}_fix $UV pip install -q --exclude-newer ${X[$pr]} -c results/${pr}_constraints.txt "$WF" || { echo "INSTALL FAILED ${pr}_fix"; continue; }
  for role in parent fix; do VIRTUAL_ENV=$T5/venvs/${pr}_$role $UV pip freeze > results/${pr}_${role}_freeze.txt; done
  diff results/${pr}_parent_freeze.txt results/${pr}_fix_freeze.txt > results/${pr}_freeze.diff; echo "freeze diff lines: $(grep -c '^[<>]' results/${pr}_freeze.diff)"; cat results/${pr}_freeze.diff
  grep -i -E "^(torch|xformers|transformers|vllm-flash-attn|flashinfer|triton)==" results/${pr}_parent_freeze.txt
  # what actually differs inside the installed vllm packages (pyc excluded)
  for role in parent fix; do (cd venvs/${pr}_$role/lib/python3.10/site-packages && find vllm -type f ! -name '*.pyc' ! -path '*/__pycache__/*' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > results/${pr}_${role}_vllm.sha256; done
  diff results/${pr}_parent_vllm.sha256 results/${pr}_fix_vllm.sha256 | grep '^[<>]' | awk '{print $1, $3}' | sort -k2 > results/${pr}_vllm_pkg.diff; echo "vllm package files differing:"; awk '{print $2}' results/${pr}_vllm_pkg.diff | sort -u
done
echo "=== benchmark scripts (upstream, at the parent and fix commits)"
mkdir -p bench
for spec in "7822 benchmarks/benchmark_prefix_caching.py" "10733 benchmarks/benchmark_latency.py"; do
  set -- $spec
  for role in parent fix; do c=${P[$1]}; [ $role = fix ] && c=${F[$1]}
    curl -fsSL -o bench/${1}_${role}_$(basename $2) https://raw.githubusercontent.com/vllm-project/vllm/$c/$2 && echo "got $1 $role $2"
  done
  cmp bench/${1}_parent_$(basename $2) bench/${1}_fix_$(basename $2) && echo "$1: benchmark script identical at parent and fix"
done
echo "=== models (HF cache under $T5/hf)"
export HF_HOME=$T5/hf HF_HUB_DISABLE_TELEMETRY=1
HFC=$T5/venvs/16436_parent/bin/huggingface-cli
$HFC download Qwen/Qwen2.5-1.5B-Instruct > /dev/null && echo "qwen ok"
$HFC download facebook/opt-125m --exclude "*.h5" "*.msgpack" > /dev/null && echo "opt ok"
$HFC download RedHatAI/Meta-Llama-3-8B-Instruct-FP8 > /dev/null && echo "llama fp8 ok"
for m in models--Qwen--Qwen2.5-1.5B-Instruct models--facebook--opt-125m models--RedHatAI--Meta-Llama-3-8B-Instruct-FP8; do echo "$m $(cat hf/hub/$m/refs/main 2>/dev/null) $(du -sh hf/hub/$m | cut -f1)"; done | tee results/model_revisions.txt
echo SETUP_DONE
