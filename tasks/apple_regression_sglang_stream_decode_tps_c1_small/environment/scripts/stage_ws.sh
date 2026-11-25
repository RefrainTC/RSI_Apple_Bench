#!/usr/bin/env bash
# Stage one ST workspace the way the image + prepare_target_env.sh (+ solve.sh) do: python/ subtree of the pinned
# tree, checked against tests/pristine_tree.sha256 -> baseline_repo/ (pristine) and repo/ (git, tag inference_base).
# usage: RSI_GPU=<any> RSI_ST_MODEL=<hf id> stage_ws.sh <ws name> <fix|noop|f1|f2>
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source $S/env.sh
ws=$RSI_WORK/$1; arm=$2; T=$RSI_TASK; SRC=$RSI_SRC
[[ "$(git -C "$SRC" rev-parse HEAD)" == 163b97012707d712e141e7ee486b72e2b0d8e60d ]]
rm -rf "$ws"; mkdir -p "$ws"
exp=$(mktemp -d -p "$TMPDIR")
git -C "$SRC" archive HEAD python | tar -x -C "$exp"
want=$(awk 'NR==1{print $1}' "$T/tests/pristine_tree.sha256")
got=$("$VPY" -I "$T/tests/_pristine_digest.py" "$exp/python" 2>/dev/null)
[[ "$want" == "$got" ]] || { echo "pristine digest mismatch: $got != $want" >&2; exit 5; }
echo "[stage] pristine tree digest OK ($got)"
mkdir -p "$ws/repo" "$ws/baseline_repo"
cp -a "$exp/python/." "$ws/repo/"; cp -a "$exp/python/." "$ws/baseline_repo/"; rm -rf "$exp"
git -C "$ws/repo" init -q -b agent_work; git -C "$ws/repo" config gc.auto 0
git -C "$ws/repo" add -A
git -C "$ws/repo" -c user.email=infb@example.com -c user.name=infb -c commit.gpgsign=false commit -q -m "engine source baseline"
git -C "$ws/repo" tag inference_base
# model: $RSI_ST_MODEL = HF repo id (default openai/gpt-oss-20b); the snapshot downloaded on the host
rid=${RSI_ST_MODEL:-openai/gpt-oss-20b}; snap=$(ls -d $HF_HOME/hub/models--${rid//\//--}/snapshots/*/ | head -1); snap=${snap%/}
[[ -f $snap/config.json ]]; ln -sfn "$snap" "$ws/model"
case "$arm" in
  noop) echo "[stage] repo left pristine (A/A control)" ;;
  fix)  INFB_WORKSPACE=$ws INFB_VLLM_PYTHON=$VPY bash "$T/solution/solve.sh"; git -C "$ws/repo" diff --stat inference_base ;;
  f1|f2) INFB_WORKSPACE=$ws INFB_VLLM_PYTHON=$VPY bash "$T/solution/solve.sh"
        "$VPY" -I $S/apply_fault_st.py "$arm" "$ws/repo"
        git -C "$ws/repo" diff --stat inference_base ;;
  *) echo "bad arm $arm" >&2; exit 2 ;;
esac
