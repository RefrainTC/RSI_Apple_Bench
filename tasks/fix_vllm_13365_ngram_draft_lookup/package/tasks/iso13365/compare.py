"""compare.py <ref correct.json> <cand correct.json> [ignored] [ignored] : #13365 correctness check (exact).
Pass iff: for each of the prompt groups (repeat, random, late match, small vocabulary; v2 adds far match), every request's token ids
and text hash equal the reference AND the engine-core counters (proposals, acceptances, steps, proposal hash) are
equal; the live model runner uses speculative decoding with k = 5, n = 3..4; and, when the candidate still has
NgramProposer.propose(context, n, k), it returns the same result as the parent on all 400 unit contexts (if that API
was removed or changed, the unit test is reported as not applicable and the engine-level checks decide)."""
import json, sys
CORE_KEYS = ("steps", "req_steps", "spec_req_steps", "scheduled_spec", "accepted", "proposed", "proposals_nonempty",
             "max_proposal", "gen_hist", "prop_hash")
r, c = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
res = {"pass": True, "groups": {}}
for g, R in r["groups"].items():
    C = c.get("groups", {}).get(g)
    if C is None:
        res["groups"][g] = {"missing": True}; res["pass"] = False; continue
    ids_eq = sum(a == b for a, b in zip(R["ids"], C["ids"])); sha_eq = sum(a == b for a, b in zip(R["sha"], C["sha"]))
    rc, cc = R.get("core") or {}, C.get("core") or {}
    cd = {k: [rc.get(k), cc.get(k)] for k in CORE_KEYS if rc.get(k) != cc.get(k)}
    ok = ids_eq == len(R["ids"]) == len(C["ids"]) and sha_eq == len(R["sha"]) and not cd and "error" not in cc and bool(cc)
    res["groups"][g] = {"n": len(R["ids"]), "ids_equal": ids_eq, "text_equal": sha_eq, "core_diff": cd, "ok": ok,
                        "accepted": cc.get("accepted"), "proposed": cc.get("proposed")}
    res["pass"] &= ok
lv = c.get("live", {})
live_ok = lv.get("use_spec_decode") is True and lv.get("k") == 5 and lv.get("nmin") == 3 and lv.get("nmax") == 4
res["live"] = lv; res["live_ok"] = live_ok; res["pass"] &= live_ok
ur, uc = r.get("unit", {}), c.get("unit", {})
if uc.get("api") != "ok":
    res["unit"] = {"applicable": False, "api": uc.get("api")}
else:
    diff = [i for i, (a, b) in enumerate(zip(ur.get("results", []), uc.get("results", []))) if a != b]
    n_ok = len(ur.get("results", [])) == len(uc.get("results", []))
    res["unit"] = {"applicable": True, "n": len(uc.get("results", [])), "differ": len(diff), "first_differ": diff[:8],
                   "s_ref": ur.get("s"), "s_cand": uc.get("s")}
    res["pass"] &= (not diff and n_ok)
print(json.dumps(res)); sys.exit(0 if res["pass"] else 1)
