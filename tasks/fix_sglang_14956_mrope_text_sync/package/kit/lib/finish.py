"""finish.py <checks tsv> <reward file> <ISO_ROOT> <label> <name> <verdict if no job> : write the reward file.
reward = harness/check.py's reward when the scoring job finished, its verdict is not fail:/incomplete, and every kit check
passed; otherwise 0."""
import json, os, sys

chk, rf, root, label, name, v0 = sys.argv[1:7]
checks = {}
for l in open(chk).read().splitlines():
    k, st, d = (l.split("\t") + ["", ""])[:3]
    checks[k] = {"ok": st == "pass", "detail": d[:600]}
res = {"reward": 0.0, "speedup": None, "verdict": v0, "checks": checks}
V = None
if label:
    vf = os.path.join(root, "results", label, "verdict.json")
    try: V = json.load(open(vf))
    except Exception: V = None
if V is not None:
    res["speedup"] = V.get("speedup"); res["verdict"] = V.get("verdict"); res["label"] = label
    res["pairs"] = V.get("pairs"); res["harness_fails"] = V.get("fails")
    gate_ok = not str(V.get("verdict")).startswith(("fail", "incomplete", "uncalibrated")) and V.get("reward") is not None
    meta_ok = V.get("mode") == "score" and V.get("cand") == name
    checks["harness_gate"] = {"ok": gate_ok, "detail": f"verdict {V.get('verdict')}, fails {V.get('fails')}"}
    checks["job_is_this_candidate"] = {"ok": meta_ok, "detail": f"mode {V.get('mode')}, cand {V.get('cand')}"}
    if gate_ok and meta_ok and all(c["ok"] for c in checks.values()):
        res["reward"] = float(V["reward"])
    elif gate_ok and meta_ok:
        res["verdict"] = "fail:kit(" + ",".join(k for k, c in checks.items() if not c["ok"]) + ")"
elif label:
    res["verdict"] = "error:no_verdict"
json.dump(res, open(rf, "w"), indent=1)
