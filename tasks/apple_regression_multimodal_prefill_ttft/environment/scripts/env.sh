# MM environment (sourced by stage_ws.sh and run_phase.sh). Mirrors the ENV block of Apple's Dockerfile, with
# everything on local disk under $RSI_ROOT/mm (venv, interpreter, model, CUDA, harness, workspaces, HOME, TMPDIR).
# Variables (paths are placeholders you set; nothing defaults to an author directory):
#   RSI_ROOT     local working root, required (keep it on local NVMe: see "Storage" in the task README)
#   RSI_OUT      where finished passes are copied (default $RSI_ROOT/out)
#   RSI_GPU      GPU index to use, required
#   RSI_CPU_MAP  file with lines '<gpu> <taskset cpu list>' (default: cpu_map.txt next to this file)
#   RSI_ARCH     TORCH_CUDA_ARCH_LIST (default 8.9 = RTX 4090; use 12.0 on an RTX 5090)
#   RSI_STORE    must be "tmp" (kept so the campaign job format stays the same as in our runs)
ROOTDIR=${RSI_ROOT:?set RSI_ROOT}
OUTDIR=${RSI_OUT:-$ROOTDIR/out}
[ "${RSI_STORE:-tmp}" = tmp ] || { echo "only RSI_STORE=tmp is supported in this copy" >&2; return 1; }
W=$ROOTDIR/mm
export VIRTUAL_ENV=$W/venv HF_HOME=$W/hf CUDA_HOME=$W/cuda HOME=$W/home TMPDIR=$W/t
RSI_HARNESS=$W/harness; RSI_TASK=$W/task; RSI_SRC=$W/sglang; RSI_WORK=$W/work
RSI_RES=$W/res; RSI_LOGS=$W/logs
mkdir -p "$HOME" "$TMPDIR" "$RSI_WORK" "$RSI_RES" "$RSI_LOGS" "$ROOTDIR/locks"
export LANG=C.UTF-8 SHELL=/bin/bash TERM=dumb
export VPY=$VIRTUAL_ENV/bin/python INFB_VLLM_PYTHON=$VIRTUAL_ENV/bin/python INFB_VENV=$VIRTUAL_ENV
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
RSI_GPU=${RSI_GPU:?set RSI_GPU}
export INFB_SERVE_PORT=$((30310 + 1000 * RSI_GPU))  # one port per GPU, 1000 apart: sglang also binds ports just above the main one
export MM_GPU_LOCK=$ROOTDIR/locks/mm_gpu$RSI_GPU.lock MM_BENCH_TIMEOUT_SEC=1500
export PATH=$VIRTUAL_ENV/bin:$CUDA_HOME/bin:/usr/local/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=$CUDA_HOME/lib64
export TORCH_CUDA_ARCH_LIST=${RSI_ARCH:-8.9}       # Apple's image: 9.0+PTX (H100)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$RSI_GPU
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export MM_EXTRA_SERVE_ARGS=${MM_EXTRA_SERVE_ARGS-}
# CPU set = that GPU's line in $RSI_CPU_MAP: '<gpu> <taskset cpu list>', ideally one L3 cache / CCD per GPU.
RSI_CPU_MAP=${RSI_CPU_MAP:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cpu_map.txt}
RSI_CPUS=$(awk -v g="$RSI_GPU" '$1==g{print $2}' "$RSI_CPU_MAP" 2>/dev/null)
[ -n "$RSI_CPUS" ] || { echo "no cpu set for gpu $RSI_GPU in $RSI_CPU_MAP (see the comment in env.sh)" >&2; return 1; }
unset PYTHONPATH
