#!/bin/bash
# run_b3.sh <out> <mode> : run the PR's bench ($BENCH from the env file) in the role's venv (PATH set by b3_job.sh).
OUT=$1; B=$P2/b3/bench
cd "$OUT" && exec env ${OFFLINE_ENV:-} python3 -I $B/${BENCH:-offline_b3.py} "$OUT"
