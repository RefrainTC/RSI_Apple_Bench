"""outcheck.py : #14930 scored-output check. check(cand_dir, base_dir, taskdir) -> dict.
Reference = calib/out_ref.json (outputs.json of a parent run on the official load; parent greedy runs are expected to be
bit-identical) when the run's workload matches it; otherwise (hidden transfer loads) the paired parent run.
Every rep, every request: exactly OL tokens, finish 'length'; per-step increments (tokens the engine emitted for the
request in one step = accepted drafts + 1) never above K+1 and summing to OL.
Greedy (TEMP 0): token ids equal to the reference for >= min_exact of the requests; same ids -> same text sha256;
  acceptance: mean tokens per request-step within inc_tol (relative) of the reference's and per-request increment
  sequences equal to the reference's for >= min_same_inc of the requests. Output equality alone cannot catch a sampler
  that skips verification on prompts whose drafts are always right; the acceptance counts can.
Sampled (TEMP > 0, transfer only): outputs are random; lengths, increment bounds and the mean tokens per request-step
  within sampled_tol (relative) of the paired parent.
Thresholds: calib.json outcheck_thresholds (from calib_iso14930.py on trusted runs), env OUT_MIN_EXACT etc. override,
else strict defaults (fail closed)."""
import json, os
def _l(p): return json.load(open(p))
def _key(o): w = o.get("workload") or {}; return tuple(w.get(k) for k in ("N", "IN", "OL", "K", "TEMP", "MIX", "PRIME", "SEED_BASE", "REPS"))
def thresholds(td):
    t = {"min_exact": 1.0, "inc_tol": 0.0, "min_same_inc": 1.0, "sampled_tol": 0.03, "source": "strict default (no calibration)"}
    try:
        c = json.load(open(os.path.join(td, "calib", "calib.json")))["outcheck_thresholds"]
        t.update({k: c[k] for k in ("min_exact", "inc_tol", "min_same_inc", "sampled_tol") if k in c}); t["source"] = "calib.json"
    except Exception: pass
    for k, e in (("min_exact", "OUT_MIN_EXACT"), ("inc_tol", "OUT_INC_TOL"), ("min_same_inc", "OUT_MIN_SAME_INC"), ("sampled_tol", "OUT_SAMPLED_TOL")):
        if os.environ.get(e): t[k] = float(os.environ[e]); t["source"] = "env"
    return t
def _mi(rows):
    xs = [v for x in rows for v in x["inc"]]; return (sum(xs) / len(xs)) if xs else None
def rep_stats(C, Rf, ol, k, greedy):
    n = len(Rf); bad = []; exact = same_inc = 0
    if len(C) != n: bad.append({"reason": "request_count", "cand": len(C), "ref": n})
    for i, x in enumerate(C):
        if len(x["ids"]) != ol or x.get("finish") != "length": bad.append({"i": i, "reason": "length_or_finish", "n": len(x["ids"])}); continue
        if not x["inc"] or max(x["inc"]) > k + 1 or sum(x["inc"]) != ol: bad.append({"i": i, "reason": "increments", "max": max(x["inc"] or [0]), "sum": sum(x["inc"])}); continue
        if greedy and i < n:
            y = Rf[i]
            if x["ids"] == y["ids"]:
                if x["text_sha256"] != y["text_sha256"]: bad.append({"i": i, "reason": "text_differs_same_ids"}); continue
                exact += 1
            if x["inc"] == y["inc"]: same_inc += 1
    a, b = _mi(C), _mi(Rf)
    return {"n": n, "bad": len(bad), "bad_sample": bad[:5], "exact": exact, "exact_frac": exact / max(n, 1), "same_inc_frac": same_inc / max(n, 1),
            "mean_tps": a, "ref_mean_tps": b, "tps_rel": (a / b - 1) if a and b else None}
def check(cand, base, td):
    T = thresholds(td); c = _l(os.path.join(cand, "outputs.json")); b = _l(os.path.join(base, "outputs.json"))
    ref, src = b, "paired_parent"; R = os.path.join(td, "calib", "out_ref.json")
    if os.path.exists(R):
        r = _l(R)
        if _key(r) == _key(c): ref, src = r, "calib_ref"
    w = c.get("workload") or {}; greedy = float(w.get("TEMP", 0)) == 0; ol = w.get("OL"); k = w.get("K")
    res = {"ref": src, "pass": True, "workload": list(_key(c)), "thresholds": T, "greedy": greedy}
    if src == "calib_ref" and _key(b) == _key(ref):
        res["parent_matches_ref"] = all(x["ids"] == y["ids"] for rr in range(len(ref["reps"])) for x, y in zip(b["reps"][rr], ref["reps"][rr]))
    if _key(b) != _key(c) or len(c["reps"]) != len(ref["reps"]):
        res["pass"] = False; res["why"] = ["workload or rep count differs from the reference"]; return res
    why = []; ex = []; si = []; tr = []
    for rr in range(len(ref["reps"])):
        s = rep_stats(c["reps"][rr], ref["reps"][rr], ol, k, greedy); res[f"rep{rr}"] = s
        if s["bad"]: why.append(f"rep{rr}: {s['bad']} malformed/text")
        tr.append(s["tps_rel"])
        if greedy:
            ex.append(s["exact_frac"]); si.append(s["same_inc_frac"])
            if s["exact_frac"] < T["min_exact"]: why.append(f"rep{rr}: exact {s['exact_frac']:.3f} < {T['min_exact']}")
            if s["same_inc_frac"] < T["min_same_inc"]: why.append(f"rep{rr}: same increments {s['same_inc_frac']:.3f} < {T['min_same_inc']}")
            if s["tps_rel"] is None or abs(s["tps_rel"]) > T["inc_tol"]: why.append(f"rep{rr}: tokens/step rel {s['tps_rel']}")
    if not greedy:
        allc = [x for rows in c["reps"] for x in rows]; allr = [x for rows in ref["reps"] for x in rows]
        a, bb = _mi(allc), _mi(allr); rel = (a / bb - 1) if a and bb else None; res["sampled_tps_rel"] = rel
        if rel is None or abs(rel) > T["sampled_tol"]: why.append(f"sampled tokens/step rel {rel}")
    res.update(identical_frac=min(ex) if ex else None, min_same_inc_frac=min(si) if si else None, tps_rel=tr)
    res["why"] = why; res["pass"] = not why
    return res
