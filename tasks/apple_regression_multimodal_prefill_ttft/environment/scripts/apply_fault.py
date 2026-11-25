"""apply_fault.py <f1|f2> <repo> : inject one Python-level fault on top of the oracle-fixed tree (exact-string edits,
each asserted to match once). No kernel code is added or changed.
  f1  output-buffer reuse in l2norm_fwd: when the output shape/dtype/device repeat, return the previous call's buffer.
      chunk.py calls q = l2norm_fwd(q); k = l2norm_fwd(k) with equal shapes, so q's normalised values are overwritten
      by k's (a plausible "avoid an allocation" optimisation that is wrong). Speed unchanged; outputs change.
  f2  a fixed 20 ms host delay (await asyncio.sleep) at the top of TokenizerManager.generate_request: every request,
      image or text, pays +20 ms TTFT. Models a candidate that removes the compile stall but regresses steady state.
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
    edit("sglang/kernels/ops/attention/fla/l2norm.py",
         "    if output_dtype is None:\n        y = torch.empty_like(x)\n    else:\n        y = torch.empty_like(x, dtype=output_dtype)\n",
         "    # [FAULT F1] reuse the previous output buffer when shape/dtype/device repeat (wrong: q and k alias)\n"
         "    global _F1_LAST\n"
         "    _want = output_dtype if output_dtype is not None else x.dtype\n"
         "    _prev = globals().get(\"_F1_LAST\")\n"
         "    if _prev is not None and _prev.shape == x.shape and _prev.dtype == _want and _prev.device == x.device:\n"
         "        y = _prev\n"
         "    else:\n"
         "        y = torch.empty_like(x, dtype=_want)\n"
         "        _F1_LAST = y\n")
elif arm == "f2":
    p = "sglang/srt/managers/tokenizer_manager.py"
    s = (repo / p).read_text()
    if "\nimport asyncio\n" not in s:
        raise SystemExit("asyncio not imported in tokenizer_manager.py")
    edit(p, "    async def generate_request(\n", "    async def generate_request(\n")  # anchor must be unique
    # insert the delay as the first statement of the body (after the signature's closing line)
    s = (repo / p).read_text()
    i = s.index("    async def generate_request(\n")
    j = s.index("):\n", i) + 3
    # skip a docstring if present
    body = s[j:]
    stripped = body.lstrip()
    indent = body[: len(body) - len(stripped)].split("\n")[-1]
    if stripped.startswith('"""'):
        k = stripped.index('"""', 3) + 3
        j = j + (len(body) - len(stripped)) + k
        j = s.index("\n", j) + 1
    ins = f"{indent}await asyncio.sleep(0.020)  # [FAULT F2] fixed 20 ms host delay on every request\n"
    s = s[:j] + ins + s[j:]
    (repo / p).write_text(s)
    print(f"[fault f2] inserted delay into {p} at char {j} (indent {len(indent)})")
else:
    raise SystemExit(f"unknown fault {arm} (this public copy keeps only f1 and f2)")
