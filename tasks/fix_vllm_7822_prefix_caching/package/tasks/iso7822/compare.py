"""compare.py <ref correct.json> <cand correct.json> <min_exact> <tie_nats> : greedy-output check tolerant to numeric
near-ties. Per sequence: exact match, or first divergence at step j where BOTH sides rank the other side's token within
<tie_nats> of their own choice (top-5 logprobs; a token missing from the top-5 counts as not tied). Pass iff no "real"
divergence, every 'unrelated' prompt (no shared prefix) is exact, and the exact fraction per pass >= <min_exact>.
'seq' = cold-cache pass, 'batch' = warm-cache pass. Old-format files (lists of token-id lists) are compared exactly."""
import json, sys
r, c = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])); MIN, TIE = float(sys.argv[3]), float(sys.argv[4])
def norm(x):
    return [s if isinstance(s, dict) else {"ids": s, "top": []} for s in x]
groups = r.get("groups") or [None] * len(r["seq"])
res = {"pass": True, "tie_nats": TIE, "min_exact": MIN}
for k in ("seq", "batch"):
    R, C = norm(r[k]), norm(c.get(k) or [])
    exact = 0; ties = []; real = []
    for i, (a, b) in enumerate(zip(R, C)):
        if a["ids"] == b["ids"]:
            if "text" in a and "text" in b and a["text"] != b["text"]:   # same tokens, different text: detokenization changed
                real.append({"i": i, "group": groups[i], "reason": "text_differs_same_ids"}); continue
            exact += 1; continue
        j = next((t for t, (x, y) in enumerate(zip(a["ids"], b["ids"])) if x != y), min(len(a["ids"]), len(b["ids"])))
        def gap(side, own, other):
            if j >= len(side["top"]): return None
            d = side["top"][j]; lo = d.get(str(own)); lt = d.get(str(other))
            return None if lo is None or lt is None else lo - lt
        ra = a["ids"][j] if j < len(a["ids"]) else None; cb = b["ids"][j] if j < len(b["ids"]) else None
        g1 = gap(a, ra, cb); g2 = gap(b, cb, ra)
        rec = {"i": i, "group": groups[i], "step": j, "gap_ref": g1, "gap_cand": g2}
        tied = g1 is not None and g2 is not None and g1 <= TIE and g2 <= TIE and groups[i] != "unrelated"
        (ties if tied else real).append(rec)
    n = len(R); fr = exact / n
    res[k] = {"n": n, "exact": exact, "frac": fr, "near_tie_divergences": ties, "real_divergences": real}
    if real or fr < MIN or len(C) != n:
        res["pass"] = False
print(json.dumps(res)); sys.exit(0 if res["pass"] else 1)
