#!/usr/bin/env bash
# build the vLLM 0.19.1 venv on the calibration host (CPU/network only). Everything stays under $P.
# v5d: P3_ROOT moves $P; the venv gets the exact pinned set of the reference venv (P3_VENV_FREEZE, default
# $P/results/venv_freeze.txt) and the script fails if the installed set differs from it.
set -euo pipefail
P=${P3_ROOT:?set P3_ROOT}
FREEZE=${P3_VENV_FREEZE:-$P/results/venv_freeze.txt}
[ -s "$FREEZE" ] || { echo "no freeze file $FREEZE"; exit 1; }
for pin in vllm==0.19.1 torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 transformers==5.9.0 \
           triton==3.6.0; do
  grep -qxF "$pin" "$FREEZE" || { echo "$FREEZE does not pin $pin"; exit 1; }
done
export UV_CACHE_DIR=$P/uv-cache UV_PYTHON_INSTALL_DIR=$P/uv-python
cd $P
if [ ! -x $P/venv/bin/python ]; then
  uv python install 3.12
  uv venv --python 3.12 --seed $P/venv
fi
# vllm 0.19.1 (Apple's pin for the gpt-oss task) + torch 2.10.0+cu128 (Apple's lock), with every other package at
# the reference venv's version: an unpinned resolve pulls e.g. transformers 5.24, which breaks vLLM 0.19.1 startup
uv pip install --python $P/venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
  -r "$FREEZE" 2>&1 | tail -30
uv pip install --python $P/venv/bin/python setuptools 2>&1 | tail -2
# byte-compile now: a server must never write __pycache__ into the venv (the scorer's audit flags writes into the
# venv as write_protected, so an uncompiled fresh venv voids every scoring as harness_fault)
$P/venv/bin/python -I -m compileall -q -j 0 $P/venv/lib/python3.12/site-packages > /dev/null 2>&1 || true
$P/venv/bin/python -I -c "import torch,vllm; print('torch',torch.__version__,torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count()); print('vllm',vllm.__version__, vllm.__file__)"
SP=$($P/venv/bin/python -I -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
for so in $SP/_C.abi3.so $SP/_moe_C.abi3.so; do echo "== $so"; ldd $so | grep -i "cudart\|cublas\|not found" || true; done
# the reference stays untouched; the installed set goes next to it and must match it line for line
OUTF=$P/results/venv_freeze.installed.txt
$P/venv/bin/python -I -m pip freeze 2>/dev/null > $OUTF || uv pip freeze --python $P/venv/bin/python > $OUTF
# compare name==version pairs; names normalised as in PEP 503 (pip and uv spell some names differently), and the
# installer's own pip / setuptools / wheel are ignored (pip freeze omits them, uv pip freeze lists them)
$P/venv/bin/python -I - "$FREEZE" "$OUTF" <<'PY' || { echo "installed venv differs from $FREEZE (list above)"; exit 1; }
import re, sys
def load(f):
    d = {}
    for ln in open(f):
        ln = ln.strip()
        if not ln or ln.startswith("#") or "==" not in ln:
            continue
        n, v = ln.split("==", 1)
        n = re.sub(r"[-_.]+", "-", n).lower()
        if n not in ("pip", "setuptools", "wheel"):
            d[n] = v.strip()
    return d
ref, got = load(sys.argv[1]), load(sys.argv[2])
bad = [f"missing {n}=={ref[n]}" for n in sorted(set(ref) - set(got))] + \
      [f"extra {n}=={got[n]}" for n in sorted(set(got) - set(ref))] + \
      [f"{n}: {got[n]} != {ref[n]}" for n in sorted(set(ref) & set(got)) if ref[n] != got[n]]
print("\n".join(bad) if bad else f"venv matches the freeze ({len(ref)} packages)")
sys.exit(1 if bad else 0)
PY
echo SETUP_DONE
