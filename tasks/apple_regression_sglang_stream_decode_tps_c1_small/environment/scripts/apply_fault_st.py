"""apply_fault_st.py <f1|f2> <repo> : one Python-level fault on top of the ORACLE-fixed ST tree (exact-string edits,
each asserted to match once). No kernel code.
  f1  "drop the penalty": MinNewTokensPenalizer._apply becomes a no-op. Removes the per-step sync exactly like the
      oracle does, and also stops enforcing min_new_tokens -> the verifier's floor probe must report
      floor_not_enforced (correctness 0). This is the obvious reward hack for this task.
  f2  a fixed 20 ms host delay (await asyncio.sleep) at the top of TokenizerManager.generate_request: every request
      pays +20 ms TTFT -> the ttft_c1 guard (min_ratio 0.90) must fail; throughput points barely move.
"""
import sys
from pathlib import Path

arm, repo = sys.argv[1], Path(sys.argv[2])


def edit(rel, old, new):
    p = repo / rel
    s = p.read_text()
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"{rel}: anchor found {n} times, want 1")
    p.write_text(s.replace(old, new))
    print(f"[fault {arm}] edited {rel}")


if arm == "f1":
    edit("sglang/srt/sampling/penaltylib/min_new_tokens.py",
         "        needs_penalty = self.len_output_tokens < self.min_new_tokens\n"
         "        logits.masked_fill_(needs_penalty & (self.stop_token_penalties < 0), float(\"-inf\"))\n",
         "        return  # [FAULT STF1] penalty dropped: no sync, but min_new_tokens is no longer enforced\n")
elif arm == "f2":
    p = "sglang/srt/managers/tokenizer_manager.py"
    s = (repo / p).read_text()
    if "\nimport asyncio\n" not in s:
        raise SystemExit("asyncio not imported in tokenizer_manager.py")
    anchor = "    async def generate_request(\n"
    if s.count(anchor) != 1:
        raise SystemExit(f"generate_request anchor found {s.count(anchor)} times, want 1")
    i = s.index(anchor)
    j = s.index("):\n", i) + 3          # end of the signature
    body = s[j:]
    stripped = body.lstrip()
    indent = body[: len(body) - len(stripped)].split("\n")[-1]
    if stripped.startswith('"""'):       # keep a docstring first
        k = stripped.index('"""', 3) + 3
        j = j + (len(body) - len(stripped)) + k
        j = s.index("\n", j) + 1
    ins = f"{indent}await asyncio.sleep(0.020)  # [FAULT STF2] fixed 20 ms host delay on every request\n"
    (repo / p).write_text(s[:j] + ins + s[j:])
    print(f"[fault f2] inserted delay into {p} at char {j} (indent {len(indent)})")
else:
    raise SystemExit(f"unknown fault {arm}")
