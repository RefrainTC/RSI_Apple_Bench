"""cal_analyze.py <cal root b3/cal/s<pr>> <ref correctness dir> <label> ... : per label: speedup per AB pair (parent / fix,
median rep), REF = median of the pairs, A/A deviations, output check (fix rep shas == paired parent rep shas), correctness
(fix vs the reference parent pass: exact fraction, divergences with their top-1 logprob gap), transfer speedups, job cost.
Across labels: median REF, range, SD relative to the median (between-job), as jobvar.py does for cards 1-3."""
import glob, json, os, statistics, sys
root, refdir, labels = sys.argv[1], sys.argv[2], sys.argv[3:]
def lat(d):
    try: return json.load(open(os.path.join(d, "latency.json")))
    except Exception: return None
def meta(d):
    try: return json.load(open(os.path.join(d, "job_meta.json")))
    except Exception: return {}
def cor(a, b):
    A = json.load(open(a)); Bv = json.load(open(b)); ex = 0; div = []
    for i, (x, y) in enumerate(zip(A, Bv)):
        if x["ids"] == y["ids"] and x["text"] == y["text"]: ex += 1; continue
        k = next((j for j in range(min(len(x["ids"]), len(y["ids"]))) if x["ids"][j] != y["ids"][j]), None)
        gap = None
        if k is not None and k < len(y["lp"]):
            lp = y["lp"][k]; ta, tb = str(x["ids"][k]), str(y["ids"][k])
            if ta in lp and tb in lp: gap = abs(lp[ta] - lp[tb])
        div.append((i, k, gap))
    return ex, len(A), div
refs = {}
for L in labels:
    R = os.path.join(root, L); sp = []; aa = []; outok = []; aash = []; cost = 0.0
    for d in sorted(glob.glob(R + "/*/")):
        cost += float(meta(d).get("wall_s", 0) or 0)
    for i in range(1, 6):
        ii = f"{i:02d}"; f = glob.glob(f"{R}/ab{ii}_*_fix"); p = glob.glob(f"{R}/ab{ii}_*_parent")
        if f and p and lat(f[0]) and lat(p[0]):
            lf, lp_ = lat(f[0]), lat(p[0]); sp.append(lp_["median_s"] / lf["median_s"])
            outok.append((sum(a == b for a, b in zip(lf["out_sha_per_rep"], lp_["out_sha_per_rep"])), len(lf["out_sha_per_rep"]), lf["out_tokens_per_rep"] == lp_["out_tokens_per_rep"]))
        a1, a2 = lat(f"{R}/aa{ii}_1_parent"), lat(f"{R}/aa{ii}_2_parent")
        if a1 and a2: aa.append(a2["median_s"] / a1["median_s"]); aash.append((sum(a == b for a, b in zip(a1["out_sha_per_rep"], a2["out_sha_per_rep"])), len(a1["out_sha_per_rep"])))
    trs = {}
    for d in sorted(glob.glob(R + "/tr_*_1_parent")):
        t = os.path.basename(d)[3:-9]; f = lat(f"{R}/tr_{t}_2_fix"); p = lat(d)
        if f and p: trs[t] = (round(p["median_s"] / f["median_s"], 4), f["out_sha_per_rep"] == p["out_sha_per_rep"], round(p["median_s"], 3), round(f["median_s"], 3))
    cf = glob.glob(R + "/cor_3_fix/correct.json"); cr = os.path.join(refdir, "correct.json")
    cres = cor(cf[0], cr) if cf and os.path.exists(cr) else None
    fixm = [lat(d)["median_s"] for d in glob.glob(R + "/ab*_fix") if lat(d)]; parm = [lat(d)["median_s"] for d in glob.glob(R + "/ab*_parent") if lat(d)]
    ref = statistics.median(sp) if sp else None; refs[L] = ref
    print(f"== {L}: pairs {['%.4f' % x for x in sp]}  REF {ref and round(ref, 4)}  fix median reps {['%.4f' % x for x in sorted(fixm)]}  parent {['%.4f' % x for x in sorted(parm)]}")
    if aa: print(f"   A/A {['%.4f' % x for x in aa]}  max dev {max(abs(1 - x) for x in aa):.4f}")
    print(f"   outputs, reps with identical text sha / reps, token counts equal: fix vs paired parent {outok}; parent A/A {aash}")
    if cres: print(f"   correctness fix vs ref parent: {cres[0]}/{cres[1]} exact; divergences (prompt, step, top-1 gap nats): {cres[2][:6]}")
    if trs: print(f"   transfer (speedup, outputs equal, parent s, fix s): {trs}")
    print(f"   GPU job time {cost:.0f} s")
v = [x for x in refs.values() if x]
if len(v) > 1:
    m = statistics.median(v); sd = statistics.stdev(v)
    print(f"ACROSS {len(v)} jobs: median REF {m:.4f}, range {min(v):.4f}-{max(v):.4f}, SD {sd:.4f} ({100 * sd / m:.2f} % of REF), band max(0.5 %, 2 SD) = {max(0.5, 200 * sd / m):.2f} %")
