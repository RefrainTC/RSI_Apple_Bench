"""metric.py : #13365 metric = sum of the timed generate() walls (s, lower is better); the witness compares the whole
measured region (primes + timed reps, in-process) with the out-of-process GPU-busy window."""
import json, os
def metric(out):
    p = os.path.join(out, "passes.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    if not d.get("timed_s"):
        return None
    return {"primary": sum(d["timed_s"]), "timed_s": d["timed_s"], "prime_s": d.get("prime_s"), "warm_s": d.get("warm_s"),
            "witness_reported_s": d.get("region_s"), "fe_steps": [x.get("steps") for x in d.get("fe_timed", [])],
            "core_lines": d.get("core_lines"), "core_pids": d.get("core_pids"), "runner_pid": d.get("runner_pid"),
            "process_wall_s": d.get("process_wall_s"), "key": d.get("key")}
