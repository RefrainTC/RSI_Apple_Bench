#!/usr/bin/env bash
# 5090 config-only sweep candidate: g_s192_a
# source: 5090 grid
# effective: seqs192 tok8192 async
# Stock vLLM 0.10.1.1, launch flags only. gpu-memory-utilization 0.85 (hardware cap on this card).
set -euo pipefail
PORT="${PORT:-30001}"
MODEL_PATH="${MODEL_PATH:-/app/model}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
ARGS=(
    --model "$MODEL_PATH"
    --served-model-name default
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size 1
    --dtype bfloat16
    --gpu-memory-utilization 0.85
    --max-model-len 8192
    --max-num-seqs 192
    --max-num-batched-tokens 8192
    --enable-chunked-prefill
    --enable-prefix-caching
    --disable-log-requests
    --trust-remote-code
    --async-scheduling
)
echo "[sweep] variant=g_s192_a argv: ${ARGS[*]}" >&2
exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
