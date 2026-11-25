"""compare.py <ref correct.json> <cand correct.json> <min_exact> <sampled_tol> : #14930 correctness (correct.py output).
Greedy part (48 prompts x 64 tokens): exact token ids vs the parent reference for >= min_exact of the prompts (spec
decoding with greedy acceptance must reproduce the target model's greedy output); same ids -> same text; every request
64 tokens, finish 'length'; tokens per engine step never above K+1 and summing to the output length.
Sampled part (temperature 0.8, seeded, 128 tokens): outputs are random and the fix draws its random numbers differently,
so only statistics are compared: all lengths 128 with finish 'length'; no step above K+1 tokens; mean tokens per
request-step overall within <sampled_tol> and in the 'lure' group within 1.6 x <sampled_tol> (relative) of the reference. A sampler that accepts
every draft (or rejects every draft) moves these means far outside the tolerance.
Also reported (not a pass condition): fraction of greedy requests whose per-step increments equal the reference's."""
import json, sys
r, c = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])); MIN, TOL = float(sys.argv[3]), float(sys.argv[4])
K = r.get("k", 3); G = r["groups"]; res = {"pass": True, "min_exact": MIN, "sampled_tol": TOL}; why = []
def basic(rows, ol, name):
    bad = []
    for i, x in enumerate(rows):
        if len(x["ids"]) != ol or x.get("finish") != "length": bad.append({"i": i, "reason": "length_or_finish", "n": len(x["ids"])})
        elif x["inc"] and (max(x["inc"]) > K + 1 or sum(x["inc"]) != len(x["ids"])): bad.append({"i": i, "reason": "increments", "max": max(x["inc"]), "sum": sum(x["inc"])})
    if bad: why.append(f"{name}: {len(bad)} malformed")
    return bad
RG, CG = r["greedy"], c.get("greedy") or []
if len(CG) != len(RG): why.append("greedy request count")
bad_g = basic(CG, r.get("greedy_ol", 64), "greedy")
exact = sum(a["ids"] == b["ids"] for a, b in zip(RG, CG)); txt = sum(a["ids"] == b["ids"] and a["text"] != b["text"] for a, b in zip(RG, CG))
same_inc = sum(a["inc"] == b["inc"] for a, b in zip(RG, CG))
div = [{"i": i, "group": G[i], "step": next((t for t, (u, v) in enumerate(zip(a["ids"], b["ids"])) if u != v), None)} for i, (a, b) in enumerate(zip(RG, CG)) if a["ids"] != b["ids"]]
fr = exact / max(len(RG), 1)
res["greedy"] = {"n": len(RG), "exact": exact, "frac": fr, "text_differs_same_ids": txt, "same_increments": same_inc, "divergences": div[:10], "malformed": bad_g[:5]}
if fr < MIN: why.append(f"greedy exact {fr:.3f} < {MIN}")
if txt: why.append("text differs for same ids")
def mean_inc(rows, sel=None):
    xs = [v for i, x in enumerate(rows) if sel is None or G[i] == sel for v in x["inc"]]
    return (sum(xs) / len(xs)) if xs else None, len(xs)
RS, CS = r["sampled"], c.get("sampled") or []
if len(CS) != len(RS): why.append("sampled request count")
bad_s = basic(CS, r.get("sampled_ol", 128), "sampled")
st = {}
for sel in (None, "lure", "exact", "random", "text"):
    a, na = mean_inc(RS, sel); b, nb = mean_inc(CS, sel)
    st[sel or "all"] = {"ref": a, "cand": b, "ref_steps": na, "cand_steps": nb, "rel": (b / a - 1) if a and b else None}
for sel, tol in (("all", TOL), ("lure", 1.6 * TOL)):
    if st[sel]["rel"] is None or abs(st[sel]["rel"]) > tol: why.append(f"sampled mean tokens/step ({sel}) rel diff {st[sel]['rel']}")
res["sampled"] = {"mean_tokens_per_step": st, "malformed": bad_s[:5]}
res["pass"] = not why; res["why"] = why
print(json.dumps(res)); sys.exit(0 if res["pass"] else 1)
