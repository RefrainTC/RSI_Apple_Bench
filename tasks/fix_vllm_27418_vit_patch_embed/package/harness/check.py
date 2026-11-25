"""check.py <ISO_ROOT> <label> : timing, output checks and verdict for one results/<label> (written by harness/run.sh).
Stdlib only. Thresholds come from harness/gate.json (set from the calibration; see the card).

Timing: speedup of a pair = parent metric / candidate metric (metric = median rep); job speedup = median of the pairs;
cal mode also reports the 5 parent/parent A/A pairs.
Gate (a candidate fails on any of these; every one is printed). The fix replaces the vision tower's Conv3d patch
embedding by a Linear on the flattened patch: the same products, summed in another order, so image outputs are close to
the parent's but not bit-identical, and text-only outputs are identical:
  counts    per rep, output token counts and the prompt length of every request (after the image placeholder is
            expanded, i.e. the number of image tokens) equal the paired parent's; the image pass's prompt lengths equal
            the calibration's (gate.json "img_n_prompt")
  settings  the engine's resolved configuration (vllm_config) equals the paired parent's (gate.json "settings_ignore" aside)
  tf        teacher-forced re-read by the trusted parent of every candidate output (timed reps and image pass): gap =
            top-1 logprob - logprob of the produced token <= tau nats for every token, and the share of tokens with gap >
            sig_gap <= max_sig_frac
  text      text pass (24 text prompts, no image, one per call, 64 tokens with logprobs): ids identical to the parent's
            pass of the same job (or to the reference pass harness/ref_correct.json), every logprob within lp_tol nats
The timed reps' output digests are reported (candidate vs paired parent) but do not gate.
Verdict: fail:<checks> (reward 0) | no_gain | below_ref | ties_ref | above_ref, reward = min(1, (S - 1) / (REF - 1))."""
import glob, json, os, statistics, sys

root, label = sys.argv[1], sys.argv[2]
R = os.path.join(root, "results", label)
G = json.load(open(os.path.join(root, "harness", "gate.json")))
meta = json.load(open(os.path.join(R, "run_meta.json")))
mode = meta["mode"]


def jl(d, f):
    try: return json.load(open(os.path.join(d, f)))
    except Exception: return None


lat = lambda d: jl(d, "latency.json")


def wall(d):
    m = jl(d, "job_meta.json") or {}
    return float(m.get("wall_s", 0) or 0), int(m.get("peak_mem_mib", 0) or 0)


def pairs(role):
    out = []
    for i in range(1, 6):
        ii = f"{i:02d}"; c = [d for d in glob.glob(f"{R}/ab{ii}_*_{role}") if "failed" not in d]
        p = [d for d in glob.glob(f"{R}/ab{ii}_*_parent") if "failed" not in d]
        if c and p: out.append((ii, c[0], p[0]))
    return out


def agree(c, p):
    """per rep: 1 if the rep's output digest equals the other run's"""
    a, b = lat(c), lat(p)
    if not (a and b): return None
    return [int(x == y) for x, y in zip(a["out_sha_per_rep"], b["out_sha_per_rep"])]


cand_role = "fix" if mode == "cal" else "cand"
fails = []; info = []
P = pairs(cand_role)
sp = []
for ii, c, p in P:
    lc, lp = lat(c), lat(p)
    if lc and lp: sp.append(lp["metric_s"] / lc["metric_s"])
aa = []; aa_agree = []
for i in range(1, 6):
    d1, d2 = f"{R}/aa{i:02d}_1_parent", f"{R}/aa{i:02d}_2_parent"
    a1, a2 = lat(d1), lat(d2)
    if a1 and a2: aa.append(a2["metric_s"] / a1["metric_s"]); aa_agree.append(agree(d2, d1))
trs = {}
for d in sorted(glob.glob(f"{R}/tr_*_1_parent")):
    t = os.path.basename(d)[3:-9]; lc = lat(f"{R}/tr_{t}_2_cand"); lp = lat(d)
    if lc and lp: trs[t] = (round(lp["metric_s"] / lc["metric_s"], 4), round(lp["metric_s"], 4), round(lc["metric_s"], 4))

# --- counts and settings (timed and transfer pairs) ---
ign = set(G.get("settings_ignore", []))
def settings_diff(c, p):
    a, b = jl(c, "settings.json"), jl(p, "settings.json")
    if a is None or b is None or "error" in a or "error" in b: return ["<missing settings.json>"]
    return sorted(k for k in set(a) | set(b) if k not in ign and a.get(k) != b.get(k))
cnt_bad = []; agr = []; sdiff = set()
allp = list(P) + [(t, f"{R}/tr_{t}_2_cand", f"{R}/tr_{t}_1_parent") for t in trs]
for ii, c, p in allp:
    lc, lp = lat(c), lat(p)
    if not (lc and lp): continue
    if lc["out_tokens_per_rep"] != lp["out_tokens_per_rep"] or lc.get("prompt_tokens_per_rep") != lp.get("prompt_tokens_per_rep"): cnt_bad.append(ii)
    agr.append((ii, agree(c, p)))
    sdiff |= set(settings_diff(c, p))
if cnt_bad: fails.append("counts"); info.append(f"output or prompt (image) token counts differ from the paired parent in pairs {cnt_bad}")
info.append(f"timed reps with identical digests (cand vs paired parent; reported, not gating): {agr}")
imgc = f"{R}/img_3_fix" if mode == "cal" else f"{R}/img_2_cand"
IP = jl(imgc, "imgpass.json")
if mode in ("cal", "score", "quick"):
    if IP is None: fails.append("counts(imgpass missing)")
    elif G.get("img_n_prompt") and [x["n_prompt"] for x in IP] != G["img_n_prompt"]:
        fails.append("counts") if "counts" not in fails else None
        info.append(f"image pass prompt lengths {[x['n_prompt'] for x in IP]} != calibration {G['img_n_prompt']}")
    if IP is not None: info.append(f"image pass answers (first 50 chars): {[x['text'][:50] for x in IP][:4]}")
if aa_agree: info.append(f"identical digests parent vs parent (A/A pairs): {aa_agree}")
# --- correctness pass ---
corc = f"{R}/cor_3_fix" if mode == "cal" else f"{R}/cor_2_cand"
corp = f"{R}/cor_1_parent"
cc, cp = jl(corc, "correct.json"), jl(corp, "correct.json")
if cc is None or cp is None:
    if mode not in ("transfer",): fails.append("text(missing)")
else:
    sdiff |= set(settings_diff(corc, corp))
    tol = G.get("lp_tol", 1e-3)
    def cmp(cc, cp):
        bad = []; mx = 0.0
        for i, (x, y) in enumerate(zip(cc, cp)):
            ok = x["ids"] == y["ids"] and len(x["lp"]) == len(y["lp"])
            if ok:
                d = max([abs(a - b) if a is not None and b is not None else (0.0 if a is None and b is None else 99.0)
                         for a, b in zip(x["lp"], y["lp"])] or [0.0]); mx = max(mx, d); ok = d <= tol
            if not ok: bad.append(i)
        if len(cc) != len(cp): bad.append("n")
        return bad, mx
    bad, mx = cmp(cc, cp)
    info.append(f"text pass: {len(cp) - len(bad)}/{len(cp)} prompts identical to the parent's pass (max logprob diff {mx:.2e}); failing prompts {bad[:16]}")
    # A candidate also passes if it reproduces the authors' reference pass (harness/ref_correct.json, recorded in
    # calibration), in case the parent's own pass of this job drifted (seen once in C47, see CARD_sgl9460.md).
    rp = os.path.join(root, "harness", "ref_correct.json")
    if bad and os.path.exists(rp):
        rbad, rmx = cmp(cc, json.load(open(rp)))
        pbad, _ = cmp(cp, json.load(open(rp)))
        info.append(f"text pass vs the reference pass: candidate {len(cp) - len(rbad)}/{len(cp)} identical (max logprob diff {rmx:.2e}); this job's parent pass {len(cp) - len(pbad)}/{len(cp)}")
        if not rbad: bad = []
    if bad: fails.append("text")
    if mode == "cal":
        c2 = jl(f"{R}/cor_2_parent", "correct.json")
        if c2: info.append(f"text pass parent vs parent: {sum(x['ids'] == y['ids'] for x, y in zip(c2, cp))}/{len(cp)} identical")
if sdiff: fails.append("settings"); info.append(f"settings differing from the paired parent: {sorted(sdiff)[:12]}")
# --- teacher-forced re-read ---
if mode != "smoke" or True:
    TF = jl(f"{R}/tf", "tf.json")
    if TF is None: fails.append("tf(missing)")
    else:
        worst = {"cand": [0.0, 0.0], "parent": [0.0, 0.0]}; tfr = {}; sg = str(G.get("sig_gap", 0.5))
        for d, v in TF["dirs"].items():
            b = os.path.basename(d)
            if "skipped" in v: tfr[b] = v["skipped"]; continue
            role = "parent" if b.endswith("_parent") else "cand"
            x = v["all"]; nsig = x.get("n_gap_above", {}).get(sg, 10 ** 9); frac = nsig / max(x["n_pos"], 1)
            tfr[b] = (x["max_gap"], nsig, x["n_nonargmax"], x["n_pos"], x["len_mismatch"])
            w = worst[role]; w[0] = max(w[0], x["max_gap"]); w[1] = max(w[1], frac)
            if role == "cand" and G.get("tau") is not None and (x["max_gap"] > G["tau"] or frac > G["max_sig_frac"] or x["len_mismatch"] or x["n_pos"] == 0):
                if "tf" not in fails: fails.append("tf")
        info.append(f"teacher-forced (max gap, tokens with gap > {sg}, non-argmax, tokens, mismatches) per run: {tfr}")
        info.append(f"teacher-forced worst: cand {worst['cand'][0]:.4f} / share {worst['cand'][1]:.4f}; parent {worst['parent'][0]:.4f} / {worst['parent'][1]:.4f} (tau {G.get('tau')}, max share {G.get('max_sig_frac')})")
# --- verdict ---
S = statistics.median(sp) if sp else None
REF = G.get("ref"); band = G.get("band", 0.01); aab = G.get("aa_band", 0.01)
if S is None: verdict, reward = ("incomplete" if mode != "transfer" else ("fail:" + ",".join(fails) if fails else "transfer")), None
elif fails: verdict, reward = "fail:" + ",".join(fails), 0.0
elif REF is None: verdict, reward = "uncalibrated", None
else:
    if S <= 1 + aab: verdict = "no_gain"
    elif S < REF * (1 - band): verdict = "below_ref"
    elif S <= REF * (1 + band): verdict = "ties_ref"
    else: verdict = "above_ref"
    reward = max(0.0, min(1.0, (S - 1) / (REF - 1)))
gpu_s = sum(wall(d)[0] for d in glob.glob(f"{R}/*/") if "failed" not in d)
peak = max([wall(d)[1] for d in glob.glob(f"{R}/*/")] or [0])
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
if trs: print(f"   transfer (speedup, parent s, cand s): {trs}")
for x in info: print("   " + x)
print(f"   GPU job time {gpu_s:.0f} s, peak VRAM {peak} MiB")
