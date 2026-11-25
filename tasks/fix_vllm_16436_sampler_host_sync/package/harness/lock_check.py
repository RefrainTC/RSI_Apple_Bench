#!/usr/bin/env python3
"""lock_check.py --stage DIR --locked "p1 p2 ..." [--json OUT]
Rejects a staged candidate (kernel_gate --stage output) that adds, changes or deletes any path under a locked prefix
(prefixes are relative to repo/, e.g. "vllm/config.py vllm/benchmarks/"). Exit 0 pass, 1 reject."""
import argparse, json, os, sys
ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True); ap.add_argument("--locked", default="")
ap.add_argument("--json"); a = ap.parse_args()
locked = [p for p in a.locked.split() if p]
changed = []
ov = os.path.join(a.stage, "repo_overlay")
for dp, _, fns in os.walk(ov):
    for f in fns:
        changed.append(os.path.relpath(os.path.join(dp, f), ov))
dl = os.path.join(a.stage, "repo_deleted.txt")
if os.path.exists(dl):
    changed += [l.strip() for l in open(dl) if l.strip()]
hits = sorted({c for c in changed for p in locked if c == p or c.startswith(p.rstrip("/") + "/") or (p.endswith("/") and c.startswith(p))})
res = {"verdict": "REJECT" if hits else "PASS", "locked_hits": hits, "n_changed": len(changed), "locked": locked}
if a.json:
    json.dump(res, open(a.json, "w"), indent=1)
print(json.dumps(res))
sys.exit(1 if hits else 0)
