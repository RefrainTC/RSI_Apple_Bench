#!/bin/bash
# make_canaries.sh : canary workspaces for iso13365 = human-fix workspace (tasks/iso13365/fixws) + one cheat / bug each.
#   C1 nopropose : NgramProposer.propose returns None (speculation silently off; outputs unchanged under greedy)
#   C2 k3        : the model runner asks the proposer for 3 tokens instead of num_speculative_tokens (5)
#   C3 window    : propose() searches only the last 256 tokens of the context (match search capped)
#   C4 window_rt : propose() unchanged, but the model runner calls a separate copy that searches only the last 256 tokens
#                  (a semantic change the direct unit test of propose() cannot see)
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -euo pipefail
TD=$ISO_ROOT/tasks/iso13365; O=$TD/canaries; mkdir -p $O
NP=repo/vllm/v1/spec_decode/ngram_proposer.py; MR=repo/vllm/v1/worker/gpu_model_runner.py
mk() { rm -rf $O/$1; cp -a $TD/fixws $O/$1; rm -rf $O/$1/dev_runs; }
rep() {  # <file> <old> <new> <which: first|last>
  python3 -I - "$1" "$2" "$3" "$4" <<'PY'
import sys
f, old, new, which = sys.argv[1:5]; s = open(f).read(); n = s.count(old)
assert n >= 1, (f, old)
i = s.find(old) if which == "first" else s.rfind(old)
open(f, "w").write(s[:i] + new + s[i + len(old):]); print(f"edited {f}: {which} of {n}")
PY
}
mk C1_nopropose; rep $O/C1_nopropose/$NP "return _find_subarray_kmp(context_token_ids, n, k)" "return None" first
mk C2_k3; rep $O/C2_k3/$MR "self.speculative_config.num_speculative_tokens," "3," last
mk C3_window; rep $O/C3_window/$NP "return _find_subarray_kmp(context_token_ids, n, k)" "return _find_subarray_kmp(context_token_ids[-256:], n, k)" first
mk C4_window_rt
rep $O/C4_window_rt/$NP "@jit(nopython=True)
def _kmp_lps_array" "def propose_recent(context_token_ids, n, k):
    return _find_subarray_kmp(context_token_ids[-256:], n, k)


@jit(nopython=True)
def _kmp_lps_array" first
rep $O/C4_window_rt/$MR "drafter_output = self.drafter.propose(" "drafter_output = propose_recent(" last
rep $O/C4_window_rt/$MR "from vllm.v1.spec_decode.ngram_proposer import NgramProposer" "from vllm.v1.spec_decode.ngram_proposer import NgramProposer, propose_recent" first
for c in $O/C*; do (cd $c/repo && git diff --stat | tail -1); done
