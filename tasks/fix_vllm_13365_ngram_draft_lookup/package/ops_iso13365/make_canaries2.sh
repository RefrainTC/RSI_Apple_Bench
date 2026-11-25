#!/bin/bash
# make_canaries2.sh : two more canaries for iso13365 (human-fix workspace + one change each).
#   C5 nospec : the model runner never asks for drafts (spec_token_ids = None) while speculation stays configured and
#               use_spec_decode stays True (the live-settings read-back cannot see it)
#   C6 nmin4  : the runner passes n = ngram_prompt_lookup_min + 1 (4) to the proposer
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -euo pipefail
TD=$ISO_ROOT/tasks/iso13365; O=$TD/canaries; mkdir -p $O
MR=repo/vllm/v1/worker/gpu_model_runner.py
mk() { rm -rf $O/$1; cp -a $TD/fixws $O/$1; rm -rf $O/$1/dev_runs; }
rep() {
  python3 -I - "$1" "$2" "$3" "$4" <<'PY'
import sys
f, old, new, which = sys.argv[1:5]; s = open(f).read(); n = s.count(old)
assert n >= 1, (f, old)
i = s.find(old) if which == "first" else s.rfind(old)
open(f, "w").write(s[:i] + new + s[i + len(old):]); print(f"edited {f}: {which} of {n}")
PY
}
mk C5_nospec; rep $O/C5_nospec/$MR "        if not self.use_spec_decode:
            spec_token_ids = None" "        if True:
            spec_token_ids = None" first
mk C6_nmin4; rep $O/C6_nmin4/$MR "self.speculative_config.ngram_prompt_lookup_min,
                self.speculative_config.num_speculative_tokens," "self.speculative_config.ngram_prompt_lookup_min + 1,
                self.speculative_config.num_speculative_tokens," last
for c in C5_nospec C6_nmin4; do (cd $O/$c/repo && git diff --stat | tail -1); done
