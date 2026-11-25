"""outcheck.py : #7822 scored-output check (independent review F2). check(cand_dir, base_dir, taskdir) -> dict.
Reference = calib/out_ref.json (outputs.json of a parent run on the official load; parent runs are bit-identical) and
calib/out_ref_lp.json (same load with top-5 logprobs), when the run's workload matches them; otherwise (hidden transfer
loads) the paired parent run, without logprobs: then every divergence counts as unexplained (transfer checks are
informational, they never decide a verdict).
Per pass, every request: exactly OL tokens and finish_reason 'length' (ignore_eos); ids equal -> text sha256 equal too
(detokenization is checked). ids differ -> look at the first differing step j: tolerated only as a numerical near-tie,
i.e. the logprob run followed the same tokens up to j, the candidate's token is in the reference's top-5 at j, and the
reference ranks it at most tie_nats below its own choice. Pass iff, in both passes: no other difference, the exactly
identical fraction >= min_exact, and the median first-divergence step >= min_first_div_median.
Thresholds (thresholds()): env OUT_TIE_NATS / OUT_MIN_EXACT / OUT_MIN_FIRST_DIV when set, else calib.json
outcheck_thresholds (calibrated on the human fix's own variation by harness/calib_outputs.py), else strict defaults
(0.5 nats, 0.5, 0) so a missing calibration fails closed."""
import json, os
def _l(p):
    return json.load(open(p))
def _key(o):
    return (o.get("workload"), o.get("repeat"), o.get("ol"), o.get("suffix_base"))
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
    """One pass: candidate requests C vs reference Rf (+ reference logprob run L or None)."""
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
    if src == "calib_ref" and _key(b) == _key(ref):   # determinism sentinel on the parent side (environment, not candidate)
        res["parent_matches_ref"] = all(x["ids"] == y["ids"] for k in (0, 1) for x, y in zip(b["passes"][k], ref["passes"][k]))
    for k in (0, 1):
        s = pass_stats(c["passes"][k], ref["passes"][k], lp["passes"][k] if lp else None, c["ol"], T["tie_nats"])
        s.pop("gaps")
        res[f"pass{k}"] = s
        fd = s["first_div_step_median"]
        if s["bad"] or s["exact_frac"] < T["min_exact"] or (fd is not None and fd < T["min_first_div_median"]):
            res["pass"] = False
    return res
