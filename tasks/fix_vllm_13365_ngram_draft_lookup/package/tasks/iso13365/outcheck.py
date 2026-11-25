"""outcheck.py : #13365 scored-output check, check(cand_dir, base_dir, taskdir) -> dict.
The PR changes only how the n-gram proposer computes its proposals, not what it proposes, so a correct candidate
reproduces the parent EXACTLY: same proposals -> same verified tokens -> same GPU batches -> bit-identical outputs.
Reference = calib/out_ref.json (a parent run on the official load) when the workload key matches, else the paired
parent run (hidden transfer loads; informational).
Pass iff, for every timed rep and every prime: same number of requests, every request exactly OL tokens with
finish_reason 'length', token ids identical, text hash identical (detokenization); AND the engine-core counters of
every generate() call are identical (steps, scheduled / accepted / proposed spec tokens, non-empty proposals, longest
proposal, tokens-per-step histogram, hash of every proposal); AND the front end saw the same number of engine steps.
So switching speculation off, proposing nothing, capping the match search, changing k or n, or any other change of
the proposals fails even when the generated text stays the same (greedy text does not depend on the proposals)."""
import json, os
CORE_KEYS = ("steps", "req_steps", "spec_req_steps", "scheduled_spec", "accepted", "proposed", "proposals_nonempty",
             "max_proposal", "gen_hist", "prop_hash")
def _l(p):
    return json.load(open(p))
def _outs(C, R, ol, tag, bad):
    if len(C) != len(R):
        bad.append({"where": tag, "reason": "request_count", "cand": len(C), "ref": len(R)}); return 0
    ex = 0
    for i, (x, y) in enumerate(zip(C, R)):
        if len(x["ids"]) != ol or x.get("fin") != "length":
            bad.append({"where": tag, "i": i, "reason": "length_or_finish", "n": len(x["ids"]), "fin": x.get("fin")}); continue
        if x["ids"] != y["ids"]:
            j = next((t for t, (u, v) in enumerate(zip(x["ids"], y["ids"])) if u != v), None)
            bad.append({"where": tag, "i": i, "reason": "ids_differ", "first_div_step": j}); continue
        if x["sha"] != y["sha"]:
            bad.append({"where": tag, "i": i, "reason": "text_differs_same_ids"}); continue
        ex += 1
    return ex
def check(cand, base, td):
    c = _l(os.path.join(cand, "outputs.json")); b = _l(os.path.join(base, "outputs.json"))
    ref, src = b, "paired_parent"
    R = os.path.join(td, "calib", "out_ref.json")
    if os.path.exists(R):
        r = _l(R)
        if r.get("key") == c.get("key"):
            ref, src = r, "calib_ref"
    res = {"ref": src, "pass": True, "key": c.get("key")}
    if src == "calib_ref" and b.get("key") == ref.get("key"):   # determinism sentinel on the parent side (environment)
        res["parent_matches_ref"] = (b["passes"] == ref["passes"] and
                                     [{k: x.get(k) for k in CORE_KEYS} for x in b.get("core", [])] == [{k: x.get(k) for k in CORE_KEYS} for x in ref.get("core", [])])
    ol = c["key"]["OL"]; bad = []; exact = 0; total = 0
    if len(c["passes"]) != len(ref["passes"]):
        bad.append({"reason": "rep_count", "cand": len(c["passes"]), "ref": len(ref["passes"])})
    for k, (C, Rr) in enumerate(zip(c["passes"], ref["passes"])):
        exact += _outs(C, Rr, ol, f"rep{k}", bad); total += len(Rr)
    for k, (C, Rr) in enumerate(zip(c.get("primes", []), ref.get("primes", []))):
        _outs(C, Rr, 1, f"prime{k}", bad)
    cc, rc = c.get("core", []), ref.get("core", [])
    core_diff = []
    if len(cc) != len(rc):
        core_diff.append({"reason": "core_line_count", "cand": len(cc), "ref": len(rc)})
    for k, (x, y) in enumerate(zip(cc, rc)):
        if "error" in x:
            core_diff.append({"call": k, "reason": "counter_error", "error": x["error"]})
        d = {f: (x.get(f), y.get(f)) for f in CORE_KEYS if x.get(f) != y.get(f)}
        if d:
            core_diff.append({"call": k, **{f: list(v) for f, v in list(d.items())[:6]}})
    fe = [x.get("steps") for x in c.get("fe_timed", [])]; fr = [x.get("steps") for x in ref.get("fe_timed", [])]
    res.update(exact=exact, total=total, exact_frac=exact / max(total, 1), bad=len(bad), bad_sample=bad[:6],
               core_equal=not core_diff, core_diff=core_diff[:6], fe_steps_equal=(fe == fr), fe_steps=fe,
               accepted_timed=[x.get("accepted") for x in cc[3::2]], proposed_timed=[x.get("proposed") for x in cc[3::2]])
    if bad or core_diff or fe != fr:
        res["pass"] = False
    return res
