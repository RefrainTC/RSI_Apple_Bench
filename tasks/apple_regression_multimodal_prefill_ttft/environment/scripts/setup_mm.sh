#!/usr/bin/env bash
# setup_mm.sh : fresh MM build from pinned downloads only (CPU + network, no GPU, no root). About 11 min on our 4090 server.
# Laid out the way env.sh (RSI_STORE=tmp) reads it:
#   $RSI_ROOT/mm/task      Apple task dir (from the Apple clone made by fetch_base.sh)
#   $RSI_ROOT/mm/harness   Apple tests/ + harness_vs_apple.diff (4 hunks, both arms; see SAMPLE_CARD.md)
#   $RSI_ROOT/mm/venv      the Dockerfile's lock (LOCK_EOF heredoc, 215 pins) installed with the Dockerfile's own pip command.
#                         A real directory, not a symlink (mem_guard.sh matches the venv path in /proc/<pid>/maps).
#   $RSI_ROOT/mm/cuda      CUDA 12.8 redist (fetch_base.sh)
#   $RSI_ROOT/mm/sglang    sglang @01a96720c6f690ac00adeed94b762b193d8472cd (stage_ws.sh checks Apple's python/ digest)
#   $RSI_ROOT/mm/hf        Qwen/Qwen3.5-4B @851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
# Idempotent: finished steps leave a marker in $RSI_ROOT/mm/.setup_*. Exit 0 only if every check passed.
set -uo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
D=${RSI_ROOT:?set RSI_ROOT}; M=$D/mm
bash "$S/fetch_base.sh" || { echo "fetch_base.sh failed"; exit 1; }
PY=$D/uvpy/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12
mkdir -p "$M/logs" "$M/home" "$M/t"
export HOME=$M/home TMPDIR=$M/t PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_RETRIES=10 PIP_TIMEOUT=60 PYTHONNOUSERSITE=1
unset PYTHONPATH PIP_INDEX_URL PIP_EXTRA_INDEX_URL
step() { echo "== $(date '+%F %T') $*"; }
rc=0

step "1. task, lock, harness"
rm -rf "$M/task" "$M/harness"
cp -a "$D/apple/tasks/regression_multimodal_prefill_ttft" "$M/task"
awk '/<<.LOCK_EOF.$/{f=1;next} /^LOCK_EOF$/{f=0} f' "$M/task/environment/Dockerfile" > "$M/requirements.lock"
n=$(grep -cE '^[A-Za-z0-9._-]+[[:space:]]*={2,3}' "$M/requirements.lock"); echo "lock pins: $n"
[ "$n" -eq 215 ] || { echo "LOCK EXTRACT: want 215 pins"; rc=1; }
cp -a "$M/task/tests" "$M/harness"
(cd "$M/harness" && patch -p2 --batch --forward --no-backup-if-mismatch < "$S/harness_vs_apple.diff") || { echo "HARNESS PATCH FAILED"; rc=1; }
h=$(grep -c "local adaptation" "$M/harness/_bench_rounds.py" "$M/harness/_candidate_runner.py" | awk -F: '{s+=$2} END {print s}')
echo "harness adaptation markers: $h (want 4)"; [ "$h" -eq 4 ] || rc=1

step "2. venv (CPython 3.12.13) + lock (Dockerfile command: pip --no-deps, pypi + sglang cu129 + pytorch cu128, 3 tries)"
if [ ! -f "$M/.setup_venv" ]; then
  rm -rf "$M/venv"; "$PY" -m venv "$M/venv"
  k=0
  until "$M/venv/bin/python" -m pip install --no-deps --index-url https://pypi.org/simple \
        --extra-index-url https://docs.sglang.ai/whl/cu129/ --extra-index-url https://download.pytorch.org/whl/cu128 \
        -r "$M/requirements.lock" > "$M/logs/pip_lock_install.log" 2>&1; do
    k=$((k+1)); tail -5 "$M/logs/pip_lock_install.log"
    [ $k -ge 3 ] && { echo "LOCK INSTALL FAILED 3x"; break; }
    echo "attempt $k failed; retry in 45 s"; sleep 45
  done
  { "$M/venv/bin/python" -m pip check || true; } > "$M/logs/pip_check.txt" 2>&1
  "$M/venv/bin/python" -I "$S/check_lock.py" "$M/requirements.lock" | tee "$M/logs/check_lock.txt"
  grep -q "missing \[\], wrong \[\]" "$M/logs/check_lock.txt" && grep -q "sglang installed as dist: False" "$M/logs/check_lock.txt" \
    && touch "$M/.setup_venv" || { echo "LOCK CHECK FAILED"; rc=1; }
fi
echo "pip check lines (4 expected, they come with the lock): $(wc -l < "$M/logs/pip_check.txt")"

step "3. sglang @01a96720"
SG=01a96720c6f690ac00adeed94b762b193d8472cd
if [ "$(git -C "$M/sglang" rev-parse HEAD 2>/dev/null)" != "$SG" ]; then
  rm -rf "$M/sglang"; mkdir -p "$M/sglang"
  git -C "$M/sglang" init -q && git -C "$M/sglang" remote add origin https://github.com/sgl-project/sglang.git \
    && git -C "$M/sglang" fetch -q --depth 1 origin $SG && git -C "$M/sglang" checkout -q FETCH_HEAD
fi
[ "$(git -C "$M/sglang" rev-parse HEAD 2>/dev/null)" = "$SG" ] || { echo "SGLANG FETCH FAILED"; rc=1; }

step "4. model Qwen/Qwen3.5-4B @851bf6e8"
HF_HOME=$M/hf HF_HUB_DISABLE_TELEMETRY=1 "$M/venv/bin/python" -I -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3.5-4B', revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'))" || { echo "MODEL DOWNLOAD FAILED"; rc=1; }
[ -f "$M/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json" ] || rc=1

step "SETUP_MM DONE rc=$rc"
exit $rc
