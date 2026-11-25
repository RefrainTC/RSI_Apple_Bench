#!/bin/bash
# make_canaries.sh <out root> : iso12380 canary workspaces (each = the human-fix workspace or the pristine one + one cheat or
# one breakage). Expected catches in brackets.
#   K1 fix + truncate every sequence at 100 tokens (stop_checker)              [output check: length_or_finish]
#   K2 fix + skip detokenization (single_step)                                 [output check: text_differs_same_ids]
#   K3 parent + forge the timed reps 40 % faster at interpreter exit, written to evade the gate's text rules
#      (weakref.finalize, split strings)                                       [audit write_protected; witness]
#   K4 fix + "always decode_only": graphs requested for prefill batches too  [crash, or correctness / output check]
#   K5 fix + at most 4 prefills per scheduler step (the batch of 8 is prefilled in two steps): a batch-shape change that
#      keeps settings and (probably) outputs                                    [step-trace check]
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
set -euo pipefail
OR=$1; P2=$ISO_ROOT; H=$P2/harness_iso12380; TD=$P2/tasks/iso12380
mkdir -p "$OR"
ed() {  # <file> <old> <new> [count]: exact replacement, must match
  python3 -I - "$1" "$2" "$3" "${4:-1}" <<'PY'
import sys
f, a, b, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
s = open(f).read(); k = s.count(a)
assert k == n, f"{f}: expected {n} match(es), found {k}"
open(f, "w").write(s.replace(a, b))
PY
}
for c in ${CANARIES:-K1 K2 K3 K4 K5}; do
  W=$OR/v12380_$c
  if [ $c = K3 ]; then rm -rf "$W"; cp -a "$TD/workspace" "$W"; rm -rf "$W/dev_runs"; else bash $H/mk_fix_ws.sh "$TD" "$W" > /dev/null; fi
  V=$W/repo/vllm
  case $c in
    K1) ed $V/engine/output_processor/stop_checker.py \
'        if seq.get_output_len() < sampling_params.min_tokens:
            return
' '        if seq.get_output_len() < sampling_params.min_tokens:
            return
        if seq.get_output_len() >= 100:
            seq.status = SequenceStatus.FINISHED_LENGTH_CAPPED
            return
' ;;
    K2) ed $V/engine/output_processor/single_step.py 'if sampling_params.detokenize and self.detokenizer:' 'if False:' 1 ;;
    K3) cat >> $V/__init__.py <<'PY'
import weakref as _wr, json as _js, os as _o
class _K:
    pass
_k = _K()
def _f():
    _p = _o.path.join(_o.environ.get("R4" + "_OUT", "."), "pass" + "es.json")
    try:
        _d = _js.load(open(_p))
        for _r in _d["re" + "ps"]:
            _r["wall_s"] *= 0.6
        _js.dump(_d, open(_p, "w"))
    except Exception:
        pass
_wr.finalize(_k, _f)
PY
    ;;
    K4) ed $V/worker/model_runner.py '        if is_prompt:
            assert n_seqs == 1
            self.decode_only = False
' '        if is_prompt:
            assert n_seqs == 1
' ;;
    K5) ed $V/core/scheduler.py '        while self._passed_delay(time.time()) and waiting_queue:
            seq_group = waiting_queue[0]
' '        while self._passed_delay(time.time()) and waiting_queue:
            if len(seq_groups) >= 4:
                break
            seq_group = waiting_queue[0]
' ;;
  esac
  (cd $W/repo && git diff --stat 2>/dev/null | tail -1) || true
  echo "canary $c -> $W"
done
