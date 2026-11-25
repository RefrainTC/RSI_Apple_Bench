#!/usr/bin/env bash
# Stage one MM workspace the way Apple's image + prepare_target_env.sh (+ solve.sh) do, plus fault arms: <ws>/repo (candidate), <ws>/baseline_repo (pristine), <ws>/model.
# usage: RSI_GPU=<any> stage_ws.sh <ws name> <fix|noop|f1|f2>
#   fix  = Apple's solution/oracle_fix.patch (what solve.sh applies)
#   noop = pristine (A/A control)
#   f1   = oracle + FAULT F1 (output-buffer reuse in l2norm_fwd: q's output is overwritten by k's) -> outputs_agree must fail
#   f2   = oracle + FAULT F2 (20 ms host delay at the top of TokenizerManager.generate_request, every request)
#          -> both guards must fail (steady-state regression) while the scored points still show a large speedup
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source $S/env.sh
T=$RSI_TASK; SRC=$RSI_SRC
ws=$RSI_WORK/$1; arm=$2
[[ "$(git -C "$SRC" rev-parse HEAD)" == 01a96720c6f690ac00adeed94b762b193d8472cd ]]
rm -rf "$ws"; mkdir -p "$ws"
exp=$(mktemp -d -p "$TMPDIR")
git -C "$SRC" archive HEAD python | tar -x -C "$exp"
want=$(awk 'NR==1{print $1}' "$T/tests/pristine_tree.sha256")
got=$("$VPY" -I "$T/tests/_pristine_digest.py" "$exp/python" 2>/dev/null)
[[ "$want" == "$got" ]] || { echo "pristine digest mismatch: $got != $want" >&2; exit 5; }
echo "[stage] pristine tree digest OK ($got)"
mkdir -p "$ws/repo" "$ws/baseline_repo"
cp -a "$exp/python/." "$ws/repo/"; cp -a "$exp/python/." "$ws/baseline_repo/"; rm -rf "$exp"
git -C "$ws/repo" init -q -b agent_work
git -C "$ws/repo" config gc.auto 0
printf '\n/sglang/_version.py\n/sglang/srt/grpc/_core*.so\n' >> "$ws/repo/.gitignore"
git -C "$ws/repo" add -A
git -C "$ws/repo" -c user.email=infb@example.com -c user.name=infb -c commit.gpgsign=false commit -q -m "engine source baseline"
git -C "$ws/repo" tag inference_base
snap=$HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
[[ -f $snap/config.json ]]; ln -sfn "$snap" "$ws/model"
case "$arm" in
  noop) echo "[stage] repo left pristine (A/A control)" ;;
  fix|f1|f2)
    (cd "$ws/repo" && patch -p2 --batch --forward < "$T/solution/oracle_fix.patch")
    grep -q 'do_not_specialize' "$ws/repo/sglang/kernels/ops/attention/fla/l2norm.py"
    echo "[stage] oracle fix applied"
    if [[ "$arm" != fix ]]; then "$VPY" -I "$S/apply_fault.py" "$arm" "$ws/repo"; fi
    git -C "$ws/repo" diff --stat inference_base ;;
  *) echo "bad arm $arm" >&2; exit 2 ;;
esac
