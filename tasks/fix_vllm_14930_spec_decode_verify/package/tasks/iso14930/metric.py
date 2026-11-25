"""metric.py : #14930 primary metric = median over the timed reps of the rep's wall time (decode phase of N requests
with a primed prefix cache). The median, not the sum, so a single slow rep (seen in the screen: 2.03 s vs 1.52 s) does
not decide a pair. Also returned: every rep, mean, the steps per rep, and the loop time for the out-of-process witness."""
import json, os, statistics as st
def metric(out):
    p = os.path.join(out, "passes.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p)); R = d["reps"]
    if not R:
        return None
    w = [r["wall_s"] for r in R]
    return {"primary": st.median(w), "reps_s": [round(x, 4) for x in w], "mean_s": st.mean(w), "sum_s": sum(w),
            "max_over_median": max(w) / st.median(w), "prime_s": [r.get("prime_s") for r in R],
            "steps": [r.get("steps") for r in R], "out_tokens": [r.get("out_tokens") for r in R],
            "witness_reported_s": d.get("loop_wall_s"), "process_wall_s": d.get("process_wall_s"), "workload": d.get("workload")}
