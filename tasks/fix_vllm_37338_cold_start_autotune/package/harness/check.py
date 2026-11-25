"""check.py <ISO_ROOT> <label> : timing, checks and verdict for one results/<label> (written by harness/run.sh). Stdlib only.
Thresholds come from harness/gate.json (set from the calibration; see the card).

Score: cold time to first result of a run = time.time() when the first generate() returned - job.sh's launch stamp
(job_meta.json "start", taken just before the Python process is started). Every run starts with empty compile caches.
Speedup of a pair = parent cold time / candidate cold time; job speedup = median of the pairs. Secondary numbers per run:
LLM() time, first-batch time, steady reps, whole-process wall time.
Gate (a candidate fails on any of these; every one is printed):
  counts    output token counts of the first batch and of every steady rep equal the paired parent's
  settings  the engine's resolved configuration equals the paired parent's (keys in gate.json "settings_ignore" aside)
  tf        teacher-forced re-read of every candidate output by the trusted parent (results/<label>/tf/tf.json): gap =
            top-1 logprob - logprob of the produced token <= tau nats for every token, and the share of tokens with
            gap > sig_gap <= max_sig_frac
  audit     harness/audit.py found something (non-Python files, blobs, cache-location names) in the candidate venv
  venv      a candidate run changed its own venv (files added / changed while running)
  cold      cold witness: the candidate's run compiled fewer than min_cubins Triton kernels into its own empty cache (a
            run that loads kernels compiled elsewhere compiles none or few)
  steady    a steady rep of the candidate took longer than steady_factor x the slowest steady rep of the paired parent
            + steady_slack_s (cost moved from the first batch to the next ones)
  after     time from first result to process end (steady reps + shutdown) of the candidate > the paired parent's
            x after_factor + after_slack_s (cost moved after the measured point)
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


def run(d):
    """timing summary of one run dir, or None"""
    l, m = jl(d, "latency.json"), jl(d, "job_meta.json")
    if not (l and m) or m.get("rc") != 0: return None
    st = float(m["start"]); en = float(m["end"])
    return {"cold": l["t_first_done"] - st, "import": l["t_import_done"] - st, "llm": l["startup_s"], "first": l["first_s"],
            "steady": l["per_rep_s"], "after": en - l["t_first_done"], "wall": en - st, "tokens": l["out_tokens_per_rep"],
            "sha": l["out_sha_per_rep"], "w": jl(d, "cold_witness.json") or {}, "v": jl(d, "venv_check.json")}


cand_role = "fix" if mode == "cal" else "cand"
fails = []; info = []
P = []
for i in range(1, 6):
    ii = f"{i:02d}"
    c = [d for d in glob.glob(f"{R}/ab{ii}_*_{cand_role}") if "failed" not in d]
    p = [d for d in glob.glob(f"{R}/ab{ii}_*_parent") if "failed" not in d]
    if c and p: P.append((ii, c[0], p[0]))
T = {}
for d in glob.glob(f"{R}/*/"):
    d = d.rstrip("/"); r = run(d)
    if r: T[os.path.basename(d)] = r
sp = []; fsp = []
for ii, c, p in P:
    rc, rp = T.get(os.path.basename(c)), T.get(os.path.basename(p))
    if rc and rp: sp.append(rp["cold"] / rc["cold"]); fsp.append(rp["first"] / rc["first"])
aa = []
for i in range(1, 6):
    a1, a2 = T.get(f"aa{i:02d}_1_parent"), T.get(f"aa{i:02d}_2_parent")
    if a1 and a2: aa.append(a2["cold"] / a1["cold"])
pairs_all = [(ii, os.path.basename(c), os.path.basename(p)) for ii, c, p in P]
trs = {}
for d in sorted(glob.glob(f"{R}/tr_*_1_parent")):
    t = os.path.basename(d)[3:-9]; c, p = f"tr_{t}_2_cand", os.path.basename(d)
    if T.get(c) and T.get(p):
        trs[t] = {"cold": round(T[p]["cold"] / T[c]["cold"], 4), "first": round(T[p]["first"] / T[c]["first"], 3),
                  "parent_cold_s": round(T[p]["cold"], 1), "cand_cold_s": round(T[c]["cold"], 1),
                  "parent_first_s": round(T[p]["first"], 3), "cand_first_s": round(T[c]["first"], 3)}
        pairs_all.append((t, c, p))

ign = set(G.get("settings_ignore", []))
def settings_diff(c, p):
    a, b = jl(f"{R}/{c}", "settings.json"), jl(f"{R}/{p}", "settings.json")
    if a is None or b is None or "error" in a or "error" in b: return ["<missing settings.json>"]
    return sorted(k for k in set(a) | set(b) if k not in ign and a.get(k) != b.get(k))
cnt_bad, sdiff, ident, steady_bad, after_bad, cold_bad, venv_bad = [], set(), [], [], [], [], []
for ii, c, p in pairs_all:
    rc, rp = T.get(c), T.get(p)
    if not (rc and rp): continue
    if rc["tokens"] != rp["tokens"]: cnt_bad.append(ii)
    ident.append((ii, sum(x == y for x, y in zip(rc["sha"], rp["sha"])), len(rp["sha"])))
    sdiff |= set(settings_diff(c, p))
    lim = max(rp["steady"] or [0]) * G["steady_factor"] + G["steady_slack_s"]
    if any(x > lim for x in rc["steady"]): steady_bad.append((ii, [round(x, 3) for x in rc["steady"]], round(lim, 3)))
    alim = rp["after"] * G["after_factor"] + G["after_slack_s"]
    if rc["after"] > alim: after_bad.append((ii, round(rc["after"], 1), round(alim, 1)))
    if ii != "warm" and rc["w"].get("triton_cubins", 0) < G["min_cubins"]: cold_bad.append((ii, rc["w"].get("triton_cubins")))
    if rc["v"] is None or rc["v"].get("venv_changed_lines", 1) != 0: venv_bad.append((ii, rc["v"]))
if cnt_bad: fails.append("counts"); info.append(f"output token counts differ from the paired parent in {cnt_bad}")
if sdiff: fails.append("settings"); info.append(f"settings differing from the paired parent: {sorted(sdiff)[:12]}")
info.append(f"digests identical to the paired parent (first batch + steady reps): {ident}")
A = jl(R, "audit.json")
if mode in ("cal", "score", "quick", "transfer"):
    if A is None: fails.append("audit(missing)")
    elif not A["ok"]: fails.append("audit"); info.append(f"audit findings: {A['findings'][:8]}")
    else: info.append(f"audit: {A['n_files_differing']} files differ from the parent, all Python source ({len(A['changed_py'])} .py, {A['added_py_bytes']} bytes added)")
if venv_bad: fails.append("venv"); info.append(f"candidate venv changed during runs: {venv_bad[:6]}")
if cold_bad: fails.append("cold"); info.append(f"cold witness below {G['min_cubins']} Triton kernels compiled: {cold_bad}")
if steady_bad: fails.append("steady"); info.append(f"steady reps slower than the bound: {steady_bad}")
if after_bad: fails.append("after"); info.append(f"time after the first result above the bound: {after_bad}")
wit = {k: (v["w"].get("triton_cubins"), v["w"].get("inductor_files"), v["w"].get("vllm_files")) for k, v in sorted(T.items())}
info.append(f"cold witness (Triton kernels, Inductor files, vLLM cache files) per run: {wit}")
# teacher-forced check
if mode != "donor":
    TF = jl(f"{R}/tf", "tf.json")
    if TF is None: fails.append("tf(missing)")
    else:
        worst = {"cand": [0.0, 0.0], "parent": [0.0, 0.0]}; tfr = {}; sg = str(G["sig_gap"])
        for d, v in TF["dirs"].items():
            b = os.path.basename(d); role = "parent" if b.endswith("_parent") else "cand"
            x = v["all"]; nsig = x.get("n_gap_above", {}).get(sg, 10 ** 9); frac = nsig / max(x["n_pos"], 1)
            tfr[b] = (x["max_gap"], nsig, x["n_nonargmax"], x["n_pos"], x["len_mismatch"])
            w = worst[role]; w[0] = max(w[0], x["max_gap"]); w[1] = max(w[1], frac)
            if role == "cand" and (x["max_gap"] > G["tau"] or frac > G["max_sig_frac"] or x["len_mismatch"]):
                if "tf" not in fails: fails.append("tf")
        info.append(f"teacher-forced (max gap, tokens with gap > {sg}, non-argmax, tokens, mismatches) per run: {tfr}")
        info.append(f"teacher-forced worst: cand {worst['cand'][0]:.4f} / share {worst['cand'][1]:.4f}; parent {worst['parent'][0]:.4f} / {worst['parent'][1]:.4f} (tau {G['tau']}, max share {G['max_sig_frac']})")
S = statistics.median(sp) if sp else None
REF = G.get("ref"); band = G.get("band", 0.02); aab = G.get("aa_band", 0.02)
if S is None: verdict, reward = ("incomplete" if mode not in ("transfer", "donor") else ("fail:" + ",".join(fails) if fails else mode)), None
elif fails: verdict, reward = "fail:" + ",".join(fails), 0.0
elif REF is None: verdict, reward = "uncalibrated", None
else:
    if S <= 1 + aab: verdict = "no_gain"
    elif S < REF * (1 - band): verdict = "below_ref"
    elif S <= REF * (1 + band): verdict = "ties_ref"
    else: verdict = "above_ref"
    reward = max(0.0, min(1.0, (S - 1) / (REF - 1)))
gpu_s = sum(float((jl(d, "job_meta.json") or {}).get("wall_s", 0) or 0) for d in glob.glob(f"{R}/*/") if "failed" not in d)
peak = max([int((jl(d, "job_meta.json") or {}).get("peak_mem_mib", 0) or 0) for d in glob.glob(f"{R}/*/")] or [0])
summ = lambda role: {k: [round(T[os.path.basename(x)][k], 2) for _, c, p in P for x in ([c] if role == "c" else [p]) if T.get(os.path.basename(x))]
                     for k in ("cold", "llm", "first", "after", "wall")}
res = {"label": label, "mode": mode, "cand": meta.get("cand"), "gpu": meta.get("gpu"), "host": meta.get("host"),
       "pairs": [round(x, 4) for x in sp], "speedup": S and round(S, 4), "first_batch_pairs": [round(x, 2) for x in fsp],
       "aa": [round(x, 4) for x in aa], "aa_max_dev": round(max(abs(1 - x) for x in aa), 4) if aa else None,
       "transfer": trs, "cand_runs": summ("c"), "parent_runs": summ("p"), "fails": fails, "verdict": verdict, "reward": reward,
       "gpu_job_s": round(gpu_s), "peak_mem_mib": peak, "info": info}
json.dump(res, open(os.path.join(R, "verdict.json"), "w"), indent=1)
print(f"== {label} ({mode}, cand {meta.get('cand')}, gpu {meta.get('gpu')}): pairs {res['pairs']} -> speedup {res['speedup']}  "
      f"verdict {verdict} reward {reward}")
print(f"   first-batch ratios {res['first_batch_pairs']}")
print(f"   cand   {res['cand_runs']}\n   parent {res['parent_runs']}")
if aa: print(f"   A/A {res['aa']} max dev {res['aa_max_dev']}")
if trs: print(f"   transfer: {trs}")
for x in info: print("   " + x)
print(f"   GPU job time {gpu_s:.0f} s, peak VRAM {peak} MiB")
