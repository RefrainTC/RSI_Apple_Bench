"""make_harness_nl.py <local harness dir> <out dir> : the local harness (make_harness.py edits 1-3) plus edit 4.
  4. _stream_client.py: the min-new-tokens floor probe gets a fixed natural-language paragraph instead of
     prompts[0][:400] (the first scored prompt's first 400 chars, random token ids).
     Why: with the substitute model (Qwen2.5-1.5B-Instruct) the greedy 256-token scout on a random-id prompt
     degenerates into a 2-3 token loop ("0 1 0 1 ..."), so the stop set has 2-3 ids instead of 20. Once the floor
     masks them, the stream diverges and never returns to them -> BOTH honest arms read `never_released` and the
     verifier zeroes every candidate, the oracle included (measured on pristine, oracle; floor_test.py, 2026-11-23).
     With the paragraph the scout gives 20 distinct ids and both honest arms read `floor_enforced_and_released`
     (35 tokens), while the penalty-drop fault (STF1) still reads `floor_not_enforced` (1 token).
     Same probe code, same floor (32) and cap (256), same request shapes; only its prompt. Both arms.
  Also: workloads.json's adaptation note now names the model actually used.
"""
import json, shutil, sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
if out.exists():
    shutil.rmtree(out)
shutil.copytree(src, out, ignore=shutil.ignore_patterns("__pycache__"))
NL = ("The water cycle describes how water moves between the oceans, the air and the land. Heat from the sun "
      "evaporates water from seas and lakes; the vapour rises, cools and condenses into clouds. When the droplets "
      "grow heavy enough they fall as rain or snow, which feeds rivers, soaks into the ground and eventually flows "
      "back to the sea. Along the way, plants take up water through their roots and release it through their leaves.")
p = out / "_stream_client.py"
s = p.read_text()
old = "record = asyncio.run(floor_probe(args.base_url, prompts[0][:400], tokenizer,"
assert s.count(old) == 1, "anchor"
s = s.replace(old, "record = asyncio.run(floor_probe(args.base_url, _LOCAL_FLOOR_PROMPT, tokenizer,")
s = s.replace("\ndef main() -> int:\n",
              "\n# [local adaptation, edit 4] natural-language floor-probe prompt (see make_harness_nl.py)\n"
              f"_LOCAL_FLOOR_PROMPT = {NL!r}\n\n\ndef main() -> int:\n", 1)
assert "_LOCAL_FLOOR_PROMPT = " in s
p.write_text(s)
wl = json.loads((out / "workloads.json").read_text())
wl["_local_adaptation"] = ("LOCAL ADAPTATION: tp_size 4 -> 1 on every point (Qwen2.5-1.5B-Instruct on 1x RTX 4090 "
                           "instead of gpt-oss-120b on 4xB200). Shapes, roles, guard bands unchanged.")
(out / "workloads.json").write_text(json.dumps(wl, indent=2) + "\n")
print(f"[harness_nl] {out}: floor-probe prompt replaced; workloads note updated")
