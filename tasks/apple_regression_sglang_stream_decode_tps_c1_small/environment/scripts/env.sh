# ST environment (sourced by stage_ws.sh and run_phase.sh). Mirrors the task Dockerfile's ENV block; everything on
# local disk under $RSI_ROOT/st.
# Variables (paths are placeholders you set; nothing defaults to an author directory):
#   RSI_ROOT      local working root, required
#   RSI_OUT       where finished passes are copied (default $RSI_ROOT/out)
#   RSI_TASK_DIR  Apple task dir (default $RSI_ROOT/apple/tasks/regression_sglang_stream_decode_tps_c1, cloned by fetch_base.sh)
#   RSI_CUDA_HOME CUDA toolkit (default $RSI_ROOT/mm/cuda = the CUDA 12.8 redist from fetch_base.sh; variant F on a
#                 5090 points it at the CUDA 12.9 tree made by mk_cuda129.sh)
#   RSI_GPU       GPU index, required;  RSI_ARCH  TORCH_CUDA_ARCH_LIST (default 8.9 = RTX 4090; 12.0 on an RTX 5090)
#   RSI_CPU_MAP   file with lines '<gpu> <taskset cpu list>' (default: cpu_map.txt next to this file)
ROOTDIR=${RSI_ROOT:?set RSI_ROOT}; W=$ROOTDIR/st; OUTDIR=${RSI_OUT:-$ROOTDIR/out}
export VIRTUAL_ENV=$W/venv VPY=$W/venv/bin/python INFB_VLLM_PYTHON=$W/venv/bin/python3 INFB_VENV=$W/venv
export INFB_AIPERF_BIN=$W/aiperf-venv/bin/aiperf
export HF_HOME=$W/hf HOME=$W/home TMPDIR=$W/t
export CUDA_HOME=${RSI_CUDA_HOME:-$ROOTDIR/mm/cuda}   # NVIDIA redist CUDA 12.8 (image: nvidia/cuda:12.8.1-devel); read-only
export TORCH_CUDA_ARCH_LIST=${RSI_ARCH:-8.9}          # image: 10.0+PTX (B200)
export PATH=$W/venv/bin:$CUDA_HOME/bin:/usr/local/bin:/usr/bin:/bin LD_LIBRARY_PATH=$CUDA_HOME/lib64
export LANG=C.UTF-8 SHELL=/bin/bash TERM=dumb PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 SGLANG_DISABLE_REQUEST_LOGGING=1
RSI_GPU=${RSI_GPU:?set RSI_GPU}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$RSI_GPU
export INFB_BENCH_PORT=$((31418 + 1000 * RSI_GPU))  # task default 31418; 1000 apart per GPU
export RSI_EXTRA_SERVE_ARGS=${RSI_EXTRA_SERVE_ARGS-}
RSI_HARNESS=${RSI_HARNESS_DIR:-$W/harness}; RSI_TASK=${RSI_TASK_DIR:-$ROOTDIR/apple/tasks/regression_sglang_stream_decode_tps_c1}
RSI_SRC=$W/sglang; RSI_WORK=$W/work; RSI_RES=$W/res; RSI_LOGS=$W/logs
mkdir -p "$HOME" "$TMPDIR" "$RSI_WORK" "$RSI_RES" "$RSI_LOGS"
RSI_CPU_MAP=${RSI_CPU_MAP:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cpu_map.txt}
RSI_CPUS=$(awk -v g="$RSI_GPU" '$1==g{print $2}' "$RSI_CPU_MAP" 2>/dev/null)
[ -n "$RSI_CPUS" ] || { echo "no cpu set for gpu $RSI_GPU in $RSI_CPU_MAP (see the comment in env.sh)" >&2; return 1; }
unset PYTHONPATH
