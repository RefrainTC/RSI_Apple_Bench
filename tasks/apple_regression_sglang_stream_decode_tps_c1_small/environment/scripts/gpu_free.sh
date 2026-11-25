#!/usr/bin/env bash
# gpu_free.sh <gpu> : exit 0 only if nobody (us included) has a compute process on that GPU
UUID=$(nvidia-smi -i $1 --query-gpu=uuid --format=csv,noheader)
n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$UUID")
echo "[gpu_free] gpu $1: $n compute apps"; [ "$n" -eq 0 ]
