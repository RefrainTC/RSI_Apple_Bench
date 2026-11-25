"""check.py <ISO_ROOT> <label> : timing, output checks and verdict for one results/<label> (written by harness/run.sh).
Stdlib only. Thresholds come from harness/gate.json (set from the calibration; see the card).

Timing: speedup of a pair = parent metric / candidate metric (metric = median rep, turn 2 for the two-turn task);
job speedup = median of the pairs; cal mode also reports the 5 parent/parent A/A pairs.
Gate (a candidate fails on any of these; every one is printed):
  counts    output token counts per rep (both turns) equal the paired parent's
  settings  the live engine's ServerArgs equal the paired parent's (keys in gate.json "settings_ignore" excepted)
  exact     (gate.json "exact_outputs": true; the logprob task) every timed rep's digest (ids + top-k logprobs rounded to
            1e-3) equals the paired parent's
  correct   correctness pass vs the parent's pass of the same job:
              logprob task: ids identical; every returned logprob (sampled token, top-k, token-id) within lp_tol nats; a
              top-k id may differ only where its value ties the k-th value within lp_tol
              two-turn task: each turn-2 output has 32 tokens; plus the teacher-forced rule on its sequences
  tf        (two-turn task) teacher-forced re-read by the trusted parent (results/<label>/tf/tf.json): for every output
            token of every candidate run (timed and correctness), gap = top-1 logprob - logprob of the produced token
            must be <= tau nats, and the share of tokens with gap > 0 (exact ties excluded) in each turn <= max_nonargmax_frac
Verdict: fail:<checks> (reward 0) | no_gain | below_ref | ties_ref | above_ref, reward = min(1, (S - 1) / (REF - 1))."""
import glob, json, os, statistics, sys

root, label = sys.argv[1], sys.argv[2]
R = os.path.join(root, "results", label)
G = json.load(open(os.path.join(root, "harness", "gate.json")))
meta = json.load(open(os.path.join(R, "run_meta.json")))
mode = meta["mode"]


def lat(d):
    try: return json.load(open(os.path.join(d, "latency.json")))
    except Exception: return None


def jl(d, f):
    try: return json.load(open(os.path.join(d, f)))
    except Exception: return None


def wall(d):
    m = jl(d, "job_meta.json") or {}
    return float(m.get("wall_s", 0) or 0), int(m.get("peak_mem_mib", 0) or 0)


def pairs(prefix_role):
    """[(pair id, cand dir, parent dir)] for ab01..ab05 (cal: cand = fix; score: cand)."""
    out = []
    for i in range(1, 6):
        ii = f"{i:02d}"; c = [d for d in glob.glob(f"{R}/ab{ii}_*_{prefix_role}") if "failed" not in d]
        p = [d for d in glob.glob(f"{R}/ab{ii}_*_parent") if "failed" not in d]
        if c and p: out.append((ii, c[0], p[0]))
    return out


cand_role = "fix" if mode == "cal" else "cand"
fails = []; info = []
P = pairs(cand_role)
sp = []
for ii, c, p in P:
    lc, lp = lat(c), lat(p)
    if lc and lp: sp.append(lp["metric_s"] / lc["metric_s"])
aa = []
for i in range(1, 6):
    a1, a2 = lat(f"{R}/aa{i:02d}_1_parent"), lat(f"{R}/aa{i:02d}_2_parent")
    if a1 and a2: aa.append(a2["metric_s"] / a1["metric_s"])
trs = {}
for d in sorted(glob.glob(f"{R}/tr_*_1_parent")):
    t = os.path.basename(d)[3:-9]; lc = lat(f"{R}/tr_{t}_2_cand"); lp = lat(d)
    if lc and lp: trs[t] = (round(lp["metric_s"] / lc["metric_s"], 4), round(lp["metric_s"], 4), round(lc["metric_s"], 4),
                            sum(a == b for a, b in zip(lc["out_sha_per_rep"], lp["out_sha_per_rep"])), len(lp["out_sha_per_rep"]))

# --- counts, exact digests, settings (timed pairs and transfer pairs) ---
ign = set(G.get("settings_ignore", []))
def settings_diff(c, p):
    a, b = jl(c, "settings.json"), jl(p, "settings.json")
    if a is None or b is None or "error" in a or "error" in b: return ["<missing settings.json>"]
    return sorted(k for k in set(a) | set(b) if k not in ign and a.get(k) != b.get(k))
cnt_bad = []; ex = []; sdiff = set()
allp = list(P) + [(t, f"{R}/tr_{t}_2_cand", f"{R}/tr_{t}_1_parent") for t in trs]
for ii, c, p in allp:
    lc, lp = lat(c), lat(p)
    if not (lc and lp): continue
    if lc["out_tokens_per_rep"] != lp["out_tokens_per_rep"] or lc.get("out_tokens_turn1_per_rep") != lp.get("out_tokens_turn1_per_rep"):
        cnt_bad.append(ii)
    ex.append((ii, sum(a == b for a, b in zip(lc["out_sha_per_rep"], lp["out_sha_per_rep"])), len(lp["out_sha_per_rep"])))
    sdiff |= set(settings_diff(c, p))
if cnt_bad: fails.append("counts")
info.append(f"timed reps with identical outputs (cand vs paired parent): {ex}")
if G.get("exact_outputs") and any(n != m for _, n, m in ex): fails.append("exact")
# --- correctness pass ---
corc = f"{R}/cor_3_fix" if mode == "cal" else f"{R}/cor_2_cand"
corp = f"{R}/cor_1_parent"
cc, cp = jl(corc, "correct.json"), jl(corp, "correct.json")
if cc is None or cp is None:
    if mode != "transfer": fails.append("correct(missing)")
else:
    sdiff |= set(settings_diff(corc, corp))
    tol = G.get("lp_tol", 1e-3); bad = []; exact = 0
    for i, (x, y) in enumerate(zip(cc, cp)):
        same_ids = x["ids"] == y["ids"]; exact += same_ids
        if G["task"] == "logprob":
            ok = same_ids and len(x["lp"]) == len(y["lp"]) and all(abs(a - b) <= tol for a, b in zip(x["lp"], y["lp"]))
            if ok and len(x["top"]) == len(y["top"]):
                for tx, ty in zip(x["top"], y["top"]):
                    vx, vy = [v for v, _ in tx], [v for v, _ in ty]
                    if len(vx) != len(vy) or any(abs(a - b) > tol for a, b in zip(vx, vy)): ok = False; break
                    kth = vy[-1] if vy else 0
                    if {t for v, t in tx if abs(v - kth) > tol} != {t for v, t in ty if abs(v - kth) > tol}: ok = False; break
            else: ok = False
            if ok and len(x.get("tid") or []) == len(y.get("tid") or []):
                for tx, ty in zip(x.get("tid") or [], y.get("tid") or []):
                    if [t for _, t in tx] != [t for _, t in ty] or any(abs(a[0] - b[0]) > tol for a, b in zip(tx, ty)): ok = False; break
            elif ok: ok = False
        else:
            ok = len(x["ids"]) == len(y["ids"])
        if not ok: bad.append(i)
    if len(cc) != len(cp): bad.append("n")
    info.append(f"correctness pass: {exact}/{len(cp)} outputs with the same token ids as the parent's pass; failing prompts {bad[:10]}")
    if bad: fails.append("correct")
if sdiff: fails.append("settings"); info.append(f"settings differing from the paired parent: {sorted(sdiff)[:12]}")
# --- teacher-forced check (two-turn task) ---
tfr = {}
if G["task"] == "mt" and mode != "smoke":
    T = jl(f"{R}/tf", "tf.json")
    if T is None: fails.append("tf(missing)")
    else:
        worst = {"cand": [0, 0.0, 0], "parent": [0, 0.0, 0]}
        for d, v in T["dirs"].items():
            role = "parent" if d.endswith("_parent") else "cand"
            npos = sum(x["n_pos"] for x in v.values()); nn = sum(x["n_nonargmax"] for x in v.values())
            mg = max(x["max_gap"] for x in v.values()); lm = sum(x["len_mismatch"] for x in v.values())
            # share of tokens strictly below the top-1 (gap > 0; exact ties are not counted), worst turn
            frac = max(x.get("n_worse", sum(1 for z in x["nonargmax"] if z[3] > 0)) / max(x["n_pos"], 1) for x in v.values())
            tfr[os.path.basename(d)] = (round(mg, 4), nn, npos, lm)
            w = worst[role]; w[0] = max(w[0], nn); w[1] = max(w[1], mg); w[2] = max(w[2], lm)
            if role == "cand" and (mg > G["tau"] or frac > G["max_nonargmax_frac"] or lm):
                if "tf" not in fails: fails.append("tf")
        info.append(f"teacher-forced (max gap nats, non-argmax tokens, tokens, length mismatches) per run: {tfr}")
        info.append(f"teacher-forced worst: cand max gap {worst['cand'][1]:.4f} / max non-argmax {worst['cand'][0]}; "
                    f"parent max gap {worst['parent'][1]:.4f} / max non-argmax {worst['parent'][0]} (tau {G['tau']}, max share {G['max_nonargmax_frac']})")
# --- verdict ---
S = statistics.median(sp) if sp else None
REF = G.get("ref"); band = G.get("band", 0.01); aab = G.get("aa_band", 0.01)
if S is None: verdict, reward = ("incomplete" if mode != "transfer" else "transfer"), None
elif fails: verdict, reward = "fail:" + ",".join(fails), 0.0
elif REF is None: verdict, reward = "uncalibrated", None
else:
    if S <= 1 + aab: verdict = "no_gain"
    elif S < REF * (1 - band): verdict = "below_ref"
    elif S <= REF * (1 + band): verdict = "ties_ref"
    else: verdict = "above_ref"
    reward = max(0.0, min(1.0, (S - 1) / (REF - 1)))
runs = [d for d in glob.glob(f"{R}/*/") if "." not in os.path.basename(d.rstrip("/"))]   # skip *.failed.* / set-aside runs
gpu_s = sum(wall(d)[0] for d in runs)
peak = max([wall(d)[1] for d in runs] or [0])
fixm = sorted(round(lat(c)["metric_s"], 4) for _, c, _ in P if lat(c)); parm = sorted(round(lat(p)["metric_s"], 4) for _, _, p in P if lat(p))
res = {"label": label, "mode": mode, "cand": meta.get("cand"), "gpu": meta.get("gpu"), "host": meta.get("host"),
       "pairs": [round(x, 4) for x in sp], "speedup": S and round(S, 4), "aa": [round(x, 4) for x in aa],
       "aa_max_dev": round(max(abs(1 - x) for x in aa), 4) if aa else None, "transfer": trs,
       "cand_metric_s": fixm, "parent_metric_s": parm, "fails": fails, "verdict": verdict, "reward": reward,
       "gpu_job_s": round(gpu_s), "peak_mem_mib": peak, "info": info}
json.dump(res, open(os.path.join(R, "verdict.json"), "w"), indent=1)
print(f"== {label} ({mode}, cand {meta.get('cand')}, gpu {meta.get('gpu')}): pairs {res['pairs']} -> speedup {res['speedup']}  "
      f"verdict {verdict} reward {reward}")
print(f"   cand metric {fixm}  parent {parm}")
if aa: print(f"   A/A {res['aa']} max dev {res['aa_max_dev']}")
if trs: print(f"   transfer (speedup, parent s, cand s, identical reps, reps): {trs}")
for x in info: print("   " + x)
print(f"   GPU job time {gpu_s:.0f} s, peak VRAM {peak} MiB")
