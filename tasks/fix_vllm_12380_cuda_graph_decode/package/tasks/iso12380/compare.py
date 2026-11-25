"""compare.py <ref correct.json> <cand correct.json> <min_exact> <tie_nats> : #12380 greedy-output check, tolerant to
numerical near-ties. Per sequence: exact match (token ids, text and finish reason), or first divergence at step j where
BOTH sides rank the other side's token within <tie_nats> of their own choice (top-5 logprobs; a token missing from the
top-5 counts as not tied). Pass iff the sequence count matches, there is no "real" divergence, and the exactly
identical fraction >= <min_exact>. Prints JSON; exit 0 pass / 1 fail."""
import json, sys
r, c = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])); MIN, TIE = float(sys.argv[3]), float(sys.argv[4])
groups = r.get("groups") or [None] * len(r["seq"])
R, C = r["seq"], c.get("seq") or []
exact = 0; ties = []; real = []
for i, (a, b) in enumerate(zip(R, C)):
    if a["ids"] == b["ids"]:
        if a.get("text") != b.get("text") or a.get("finish") != b.get("finish"):
            real.append({"i": i, "group": groups[i], "reason": "text_or_finish_differs_same_ids"}); continue
        exact += 1; continue
    j = next((t for t, (x, y) in enumerate(zip(a["ids"], b["ids"])) if x != y), min(len(a["ids"]), len(b["ids"])))
    def gap(side, own, other):
        if j >= len(side["top"]) or own is None or other is None: return None
        d = side["top"][j]; lo = d.get(str(own)); lt = d.get(str(other))
        return None if lo is None or lt is None else lo - lt
    ra = a["ids"][j] if j < len(a["ids"]) else None; cb = b["ids"][j] if j < len(b["ids"]) else None
    g1 = gap(a, ra, cb); g2 = gap(b, cb, ra)
    rec = {"i": i, "group": groups[i], "step": j, "gap_ref": g1, "gap_cand": g2}
    (ties if (g1 is not None and g2 is not None and g1 <= TIE and g2 <= TIE) else real).append(rec)
n = len(R); fr = exact / n if n else 0.0
res = {"pass": not real and fr >= MIN and len(C) == n, "n": n, "n_cand": len(C), "exact": exact, "frac": fr,
       "tie_nats": TIE, "min_exact": MIN, "near_tie_divergences": ties, "real_divergences": real}
print(json.dumps(res)); sys.exit(0 if res["pass"] else 1)
