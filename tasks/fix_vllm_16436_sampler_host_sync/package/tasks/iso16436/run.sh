#!/bin/bash
# run.sh <out> <timed|correct|counter> : ISO-Bench #16436 workload.
# Server = the PR's command minus VLLM_FLASH_ATTN_VERSION=3 (Hopper-only) plus --gpu-memory-utilization 0.85, run with the
# candidate overlay first on PYTHONPATH (P2_OVERLAY). Client = ISO-Bench perf_command, run from the pristine venv with -I
# (the overlay is NOT on its path), so the measuring code is never the candidate's.
# Workload env: RATE (1) NP (100) IN (1000) OL (100) GENCFG (auto|vllm) PORT SEED_BASE (0; warm-ups use 1000) REPS (1; >1 = repeats on the same server, seeds 0..REPS-1; seed 0 = the original perf_command).
OUT=$1; MODE=${2:-timed}; TD=$(dirname "$(readlink -f "$0")")
PORT=${PORT:-$(( 18430 + ${CUDA_VISIBLE_DEVICES:-0} ))}; NP=${NP:-100}; RATE=${RATE:-1}; IN=${IN:-1000}; OL=${OL:-100}; GENCFG=${GENCFG:-auto}
M=Qwen/Qwen2.5-1.5B-Instruct
# port check: SO_REUSEADDR (as uvicorn) so TIME_WAIT left by the previous run's connections does not count; a real
# listener still fails the bind. Wait up to 90 s for one to go away.
pfree() { python3 -I -c "import socket; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('127.0.0.1', $PORT)); s.close()" 2>/dev/null; }
for _w in $(seq 1 30); do pfree && break; sleep 3; done
pfree || { echo "port $PORT in use"; exit 9; }
GC=(); [ "$GENCFG" = vllm ] && GC=(--generation-config vllm)
# record which vllm the server will import (same env and interpreter as the server)
env ${P2_OVERLAY:+PYTHONPATH=$P2_OVERLAY} python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" > "$OUT/import_check.txt" 2>&1
PP=${P2_OVERLAY:-}; [ -n "${P2_AUDIT_DIR:-}" ] && PP=${PP:+$PP:}$P2_AUDIT_DIR   # audit hook (sitecustomize) in the server
SYNC=()
if [ "$MODE" = counter ]; then NP=20; mkdir -p "$OUT/sync"; PP="$T5/scripts/syncdbg${PP:+:$PP}"; SYNC=(R4_SYNC_OUT="$OUT/sync"); fi
env ${PP:+PYTHONPATH=$PP} "${SYNC[@]}" python3 -m vllm.entrypoints.cli.main serve $M --enable-prefix-caching \
   --dtype float16 --disable-log-requests -O3 --gpu-memory-utilization 0.85 "${GC[@]}" --host 127.0.0.1 --port $PORT > "$OUT/server.log" 2>&1 &
SP=$!
ok=0
for i in $(seq 1 240); do
  curl -sf http://127.0.0.1:$PORT/health > /dev/null && { ok=1; break; }
  kill -0 $SP 2>/dev/null || break
  sleep 3
done
[ $ok = 1 ] || { echo "server not ready"; kill -TERM $SP; wait $SP; exit 2; }
echo "server ready after ~$((i*3)) s; MODE=$MODE RATE=$RATE NP=$NP IN=$IN OL=$OL GENCFG=$GENCFG overlay=${P2_OVERLAY:-none}"
case "$MODE" in
  correct) python3 -I "$TD/correct_client.py" --port $PORT --model $M --out "$OUT/correct.json" > "$OUT/correct.log" 2>&1; RC=$? ;;
  *) RC=0
     for r in $(seq 1 ${REPS:-1}); do   # REPS>1: repeat the load on the same server (no extra cold start)
       python3 -I -m vllm.entrypoints.cli.main bench serve --model $M --request-rate $RATE --num-prompts $NP --random-input-len $IN \
         --random-output-len $OL --tokenizer $M --ignore-eos --host 127.0.0.1 --port $PORT --seed $(( ${SEED_BASE:-0} + r - 1 )) \
         --save-result --result-dir "$OUT" --result-filename bench_r$r.json > "$OUT/bench_r$r.log" 2>&1 || RC=$?
     done
     cp "$OUT/bench_r1.json" "$OUT/bench.json" 2>/dev/null; cp "$OUT/bench_r1.log" "$OUT/bench.log" 2>/dev/null ;;
esac
kill -INT $SP; for i in $(seq 1 30); do kill -0 $SP 2>/dev/null || break; sleep 1; done; kill -KILL $SP 2>/dev/null; wait $SP 2>/dev/null
exit $RC
