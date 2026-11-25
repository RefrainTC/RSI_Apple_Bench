"""outcheck.py : #16436 scored-output check (review F2, applied to the shared harness). check(cand, base, taskdir).
The scored runs are the bench client's (pristine venv, python -I) requests; their outputs are in bench_r<k>.json (one per
repetition of the load on the same server; repetition k uses seed k-1 on both sides). The candidate's run is compared
with the parent run of the same pair (rate-1 arrivals change batch composition from run to run, so parent runs differ
from each other in a few texts at numerical near-ties; there are no logprobs).
Pass iff, in every repetition: every request completed, the per-request output lengths equal the parent's (ignore_eos),
no errors, no empty text where the parent has one, the fraction of identical texts >= OUT_MIN_EXACT (env, else
calib.json out_min_exact, else 0.9), and (2026-11-23) the median TTFT <= the parent's x (1 + TTFT_TOL).
Why the TTFT guard: the metric is median TPOT = (end-to-end - TTFT) / (tokens - 1), measured by the client. A server
that holds tokens back and streams them in a burst moves time from TPOT into TTFT with identical outputs (canary C2:
"speedup" of orders of magnitude). With the guard, such shifting can gain at most TTFT_TOL x TTFT / 99 ~ 0.018 ms of
TPOT (~0.35 %, reward ~+0.04 at REF 1.093). TTFT_TOL: env OUT_TTFT_TOL, else calib.json ttft_tol, else 0.05
(coordinator 2026-11-23; calibration: paired fix/parent and parent/parent median-TTFT ratios 0.93-1.02). Ratios above
1.03 do not fail but are listed in "ttft_watch" for the audit, so legitimate TTFT/TPOT trade-offs stay visible."""
import glob, json, os
def _minex(td):
    if os.environ.get("OUT_MIN_EXACT"):
        return float(os.environ["OUT_MIN_EXACT"])
    try:
        return float(json.load(open(os.path.join(td, "calib", "calib.json")))["out_min_exact"])
    except Exception:
        return 0.9
def _ttol(td):
    if os.environ.get("OUT_TTFT_TOL"):
        return float(os.environ["OUT_TTFT_TOL"])
    try:
        return float(json.load(open(os.path.join(td, "calib", "calib.json")))["ttft_tol"])
    except Exception:
        return 0.05
def _reps(d):
    return sorted(glob.glob(os.path.join(d, "bench_r*.json"))) or [os.path.join(d, "bench.json")]
def check(cand, base, td):
    MINEX = _minex(td); TT = _ttol(td); fc, fb = _reps(cand), _reps(base)
    res = {"pass": True, "min_exact": MINEX, "ttft_tol": TT, "reps": len(fc), "identical_frac": None, "per_rep": []}
    if [os.path.basename(x) for x in fc] != [os.path.basename(x) for x in fb]:
        res.update({"pass": False, "reason": "repetitions differ"}); return res
    fr = []
    for x, y in zip(fc, fb):
        c, b = json.load(open(x)), json.load(open(y))
        tc, tb = c.get("generated_texts") or [], b.get("generated_texts") or []
        n = len(tb); f = sum(u == v for u, v in zip(tc, tb)) / max(n, 1); fr.append(f)
        why = None
        if c.get("completed") != b.get("completed") or len(tc) != n: why = "completed"
        elif c.get("output_lens") != b.get("output_lens"): why = "output_lens"
        elif any(e for e in (c.get("errors") or [])): why = "errors"
        elif any((not u) and v for u, v in zip(tc, tb)): why = "empty_text"
        elif f < MINEX: why = "texts"
        rt = (c.get("median_ttft_ms") or 0) / b["median_ttft_ms"] if b.get("median_ttft_ms") else None
        if why is None and (rt is None or rt > 1 + TT): why = "ttft_regression"
        res["per_rep"].append({"rep": os.path.basename(x), "identical_frac": f, "ttft_ratio": rt, "completed": [c.get("completed"), b.get("completed")], "fail": why})
        if why:
            res["pass"] = False; res.setdefault("reason", why)
    res["identical_frac"] = min(fr) if fr else None
    res["ttft_ratios"] = [r["ttft_ratio"] for r in res["per_rep"]]
    res["ttft_watch"] = [r["rep"] for r in res["per_rep"] if r["ttft_ratio"] is not None and r["ttft_ratio"] > 1.03]
    return res
