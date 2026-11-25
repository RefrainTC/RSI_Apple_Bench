"""microbench.py : CPU-only cost of one NgramProposer.propose call, parent (Python KMP) vs fix (Numba), on contexts like
the official and transfer loads. Loads the two ngram_proposer.py files directly (numpy / numba only, no vllm import)."""
import importlib.util, json, sys, time
import numpy as np
def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
P = load(sys.argv[1], "ngp_parent").NgramProposer(); F = load(sys.argv[2], "ngp_fix").NgramProposer()
F.propose(np.zeros(1024, dtype=np.int32), 3, 5)   # the fix's own JIT warm-up
rs = np.random.RandomState(0); seg = rs.randint(100, 30000, size=40).astype(np.int32)
def ctx(kind, L):
    if kind == "repeat_inperiod": return np.resize(seg, L).astype(np.int32)           # last 3 tokens continue the period
    if kind == "repeat_offperiod":                                                      # prompt periodic, last 3 tokens novel
        c = np.resize(seg, L).astype(np.int32); c[-3:] = [31001, 31002, 31003]; return c
    if kind == "random": return rs.randint(100, 30000, size=L).astype(np.int32)
    if kind == "late":
        c = np.concatenate([rs.randint(100, 30000, size=L // 2), np.resize(seg, L - L // 2)]).astype(np.int32); return c
res = {}
for kind, L in [("repeat_inperiod", 2100), ("repeat_offperiod", 2100), ("random", 2100), ("late", 2100), ("repeat_offperiod", 4000), ("random", 4000)]:
    c = ctx(kind, L); out = {}
    for nm, pr in (("parent", P), ("fix", F)):
        n = 200 if nm == "parent" else 5000
        t = time.perf_counter()
        for _ in range(n): r = pr.propose(c, 3, 5)
        out[nm + "_us"] = round((time.perf_counter() - t) / n * 1e6, 2)
    out["same"] = (lambda a, b: (a is None and b is None) or (a is not None and b is not None and list(a) == list(b)))(P.propose(c, 3, 5), F.propose(c, 3, 5))
    res[f"{kind}_{L}"] = out; print(kind, L, out, flush=True)
json.dump(res, open(sys.argv[3], "w"), indent=1)
