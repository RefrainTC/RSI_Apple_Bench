#!/usr/bin/env bash
# Build tasks/<t>/pristine: repo/vllm = the venv's vllm 0.19.1 package minus compiled .so (linked back at run time)
set -euo pipefail
P=${P3_ROOT:?set P3_ROOT}
T=$P/tasks/$1
SP=$P/venv/lib/python3.12/site-packages/vllm
[ -d $T/pristine ] && chmod -R u+w $T/pristine && rm -rf $T/pristine
mkdir -p $T/pristine/repo $T/pristine/submission
mkdir -p $T/pristine/repo/vllm
(cd $SP && tar cf - --exclude='*.so' --exclude='__pycache__' --exclude='*.pyc' .) | (cd $T/pristine/repo/vllm && tar xf -)
cat > $T/pristine/submission/extra_args.txt <<'X'
# Extra `vllm serve` flags, one per line: "--flag" or "--flag value". Locked flags are rejected.
X
cat > $T/pristine/submission/env.txt <<'X'
# Extra environment variables, one per line: NAME=value (VLLM_*, TORCHINDUCTOR_*, NCCL_*, OMP_NUM_THREADS, CUDA_DEVICE_MAX_CONNECTIONS).
X
(cd $T/pristine && find . -type f | LC_ALL=C sort | xargs sha256sum) > $T/pristine.sha256
chmod -R a-w $T/pristine
echo "pristine files: $(wc -l < $T/pristine.sha256)"; du -sh $T/pristine
