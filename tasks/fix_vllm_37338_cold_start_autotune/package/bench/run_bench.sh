#!/bin/bash
# run_bench.sh <envfile> <out> : inside the clean job environment, source the load file (export lines only) and run the
# bench with the role's venv python (first on PATH, set by job.sh), isolated (-I), from the output directory.
EF=$1; OUT=$2; . "$EF"; cd "$OUT" && exec python3 -I "$ISO_ROOT/bench/offline_c5y.py" "$OUT"
