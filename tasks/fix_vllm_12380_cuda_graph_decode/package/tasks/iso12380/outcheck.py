"""outcheck.py : #12380 scored-run check. check(cand_dir, base_dir, taskdir) -> dict.
1. Outputs (every timed rep, every request). Reference = calib/out_ref.json (outputs.json of a parent run on the official
   load; parent runs are deterministic) + calib/out_ref_lp.json (same load with top-5 logprobs), when the run's workload
   key matches; otherwise (hidden transfer loads) the paired parent run without logprobs: then every divergence counts
   as unexplained (transfer checks are informational, they never decide a verdict).
   Per request: exactly OL tokens and finish_reason 'length' (ignore_eos); ids equal -> text sha256 equal too
   (detokenization is checked); ids differ -> tolerated only as a numerical near-tie: the logprob run followed the same
   tokens up to the first differing step j, has the candidate's token in its top-5 at j, and ranks it at most tie_nats
   below its own choice. Over all reps: no other difference, exactly-identical fraction >= min_exact, median
   first-divergence step >= min_first_div_median.
2. Step trace (batch shapes): for every rep, the sequence of model-runner steps as (num_prefills, num_prefill_tokens,
   num_decode_tokens) must equal the paired parent's, and the trace hook must have worked on both sides (fail closed).
   This rejects candidates that win by changing how the batch is scheduled or split (fewer / merged / multi-token steps,
   dropped sequences). How many decode steps replayed a CUDA graph is reported for both sides, not required.
Thresholds (thresholds()): env OUT_TIE_NATS / OUT_MIN_EXACT / OUT_MIN_FIRST_DIV when set, else calib.json
outcheck_thresholds (calibrated on the human fix's and the parent's own variation by harness_iso12380/calib_outputs.py),
else strict defaults (0.5 nats, 0.5, 0) so a missing calibration fails closed."""
import json, os
def _l(p):
    return json.load(open(p))
def _key(o):
    return tuple(o.get(k) for k in ("workload", "bs", "in", "ol", "reps", "seed_base"))
def thresholds(td):
    t = {"tie_nats": 0.5, "min_exact": 0.5, "min_first_div_median": 0, "source": "strict default (no calibration)"}
    try:
        c = json.load(open(os.path.join(td, "calib", "calib.json")))["outcheck_thresholds"]
        t.update({k: c[k] for k in ("tie_nats", "min_exact", "min_first_div_median")}); t["source"] = "calib.json"
    except Exception:
        pass
    for k, e in (("tie_nats", "OUT_TIE_NATS"), ("min_exact", "OUT_MIN_EXACT"), ("min_first_div_median", "OUT_MIN_FIRST_DIV")):
        if os.environ.get(e):
            t[k] = float(os.environ[e]); t["source"] = "env"
    return t
def pass_stats(C, Rf, L, ol, tie):
    """One list of requests: candidate C vs reference Rf (+ reference logprob run L or None)."""
    n = len(Rf); exact = ties = 0; bad = []; steps = []; gaps = []; notop = 0
    if len(C) != n:
        bad.append({"reason": "request_count", "cand": len(C), "ref": n})
    for i, (x, y) in enumerate(zip(C, Rf)):
        if len(x["ids"]) != ol or x.get("finish") != "length":
            bad.append({"i": i, "reason": "length_or_finish", "n": len(x["ids"]), "finish": x.get("finish")}); continue
        if x["ids"] == y["ids"]:
            if x["text_sha256"] != y["text_sha256"]:
                bad.append({"i": i, "reason": "text_differs_same_ids"}); continue
            exact += 1; continue
        j = next((t for t, (u, v) in enumerate(zip(x["ids"], y["ids"])) if u != v), min(len(x["ids"]), len(y["ids"])))
        steps.append(j); gap = None
        if L is not None and i < len(L) and L[i]["ids"][: j + 1] == y["ids"][: j + 1] and j < len(L[i].get("top") or []):
            top = L[i]["top"][j]; own = top.get(str(y["ids"][j])); alt = top.get(str(x["ids"][j]))
            if own is not None and alt is not None:
                gap = own - alt; gaps.append(gap)
            elif own is not None:
                notop += 1
        if gap is not None and gap <= tie:
            ties += 1
        else:
            bad.append({"i": i, "reason": "divergence", "step": j, "gap_nats": gap})
    st = sorted(steps)
    return {"n": n, "exact": exact, "exact_frac": exact / max(n, 1), "tie_divergences": ties, "bad": len(bad),
            "bad_sample": bad[:5], "diverged": len(steps), "first_div_step_median": st[len(st) // 2] if st else None,
            "max_gap_nats": max(gaps) if gaps else None, "not_in_top5": notop, "gaps": sorted(round(g, 4) for g in gaps)}
def all_stats(o, ref, lp, tie):
    """All reps of one run, flattened (request i of rep r)."""
    if len(o["passes"]) != len(ref["passes"]):
        return {"n": len(ref["passes"]), "exact": 0, "exact_frac": 0.0, "tie_divergences": 0, "bad": 1,
                "bad_sample": [{"reason": "rep_count", "cand": len(o["passes"]), "ref": len(ref["passes"])}], "diverged": 0,
                "first_div_step_median": None, "max_gap_nats": None, "not_in_top5": 0, "gaps": []}
    C = [x for p in o["passes"] for x in p]; R = [x for p in ref["passes"] for x in p]
    L = [x for p in lp["passes"] for x in p] if lp else None
    return pass_stats(C, R, L, o["ol"], tie)
def step_check(cand, base):
    try:
        c = _l(os.path.join(cand, "steps.json")); b = _l(os.path.join(base, "steps.json"))
    except Exception as e:
        return {"pass": False, "error": f"no step trace: {e!r}"}
    res = {"hooks_cand": c.get("hooks"), "hooks_base": b.get("hooks")}
    if "execute_model" not in (c.get("hooks") or []) or "execute_model" not in (b.get("hooks") or []):
        res["pass"] = False; res["error"] = "trace hook missing"; return res
    sc, sb = c.get("steps") or [], b.get("steps") or []
    shape = lambda tr: [tuple(x[:3]) for x in tr]
    mism = [r for r in range(max(len(sc), len(sb))) if r >= len(sc) or r >= len(sb) or shape(sc[r]) != shape(sb[r])]
    def gfrac(S):
        dec = [x for tr in S for x in tr if x[0] == 0]
        return round(sum(1 for x in dec if x[3]) / len(dec), 4) if dec else None
    res.update({"pass": not mism and len(sc) == len(sb) and len(sc) > 0, "reps_mismatched": mism[:10],
                "steps_per_rep_cand": sorted({len(t) for t in sc}), "steps_per_rep_base": sorted({len(t) for t in sb}),
                "graph_decode_frac_cand": gfrac(sc), "graph_decode_frac_base": gfrac(sb)})
    if mism:
        r = mism[0]
        res["first_mismatch"] = {"rep": r, "cand": shape(sc[r])[:6] if r < len(sc) else None, "base": shape(sb[r])[:6] if r < len(sb) else None}
    return res
def check(cand, base, td):
    T = thresholds(td)
    c = _l(os.path.join(cand, "outputs.json")); b = _l(os.path.join(base, "outputs.json"))
    ref, lp, src = b, None, "paired_parent"
    R, RL = os.path.join(td, "calib", "out_ref.json"), os.path.join(td, "calib", "out_ref_lp.json")
    if os.path.exists(R):
        r = _l(R)
        if _key(r) == _key(c):
            ref, src = r, "calib_ref"; lp = _l(RL) if os.path.exists(RL) else None
    res = {"ref": src, "pass": True, "workload": list(_key(c)), "thresholds": T}
    if _key(b) != _key(c):
        res["pass"] = False; res["error"] = "workload differs from the paired parent"
    if src == "calib_ref" and _key(b) == _key(ref):   # determinism sentinel on the parent side (environment, not candidate)
        res["parent_matches_ref"] = all(x["ids"] == y["ids"] for p, q in zip(b["passes"], ref["passes"]) for x, y in zip(p, q))
    s = all_stats(c, ref, lp, T["tie_nats"]); s.pop("gaps")
    res["outputs"] = s
    fd = s["first_div_step_median"]
    if s["bad"] or s["exact_frac"] < T["min_exact"] or (fd is not None and fd < T["min_first_div_median"]):
        res["pass"] = False
    res["steps"] = step_check(cand, base)
    if not res["steps"].get("pass"):
        res["pass"] = False
    return res
