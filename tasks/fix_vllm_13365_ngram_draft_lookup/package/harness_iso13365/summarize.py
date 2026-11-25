#!/usr/bin/env python3
"""summarize.py <taskdir> <runs_dir> [--label L] [--ref R] [--calib calib.json] -> JSON on stdout.
Uses <taskdir>/metric.py: metric(outdir) -> {"primary": float (lower is better), "witness_reported_s": optional, ...}.
Pairs are grouped by tag <label>_<kind><ii>_<k>_<role> (kind "warm" = untimed warm-up, never scored). For each pair:
ratio = B/A in run order, and speedup = metric(baseline)/metric(other) for the two roles.
Integrity checks (review of #7822, F1-F5), per run:
  witness  : out-of-process GPU-busy window vs the in-process reported time (harness/witness.py), when the metric
             reports witness_reported_s; flagged when residual - calib median > calib tolerance
             (one-sided: under-reported time raises the residual)
  audit    : audit-hook events (harness/audit_check.py); env P2_AUDIT_NEED_END, P2_AUDIT_IGNORE
  settings : effective engine settings (task settings.py: settings(outdir) -> dict); pairs must match except keys in
             env SETTINGS_IGNORE (space separated)
  venv     : venv_guard.json post state (changed_by_run = the run modified the venv)
per pair: outcheck = task outcheck.py check(cand_dir, base_dir, taskdir) -> {"pass": bool, ...} (scored outputs).
Every pair gets "flags"; a summary key lists flagged pairs. Also: storage / cpus / cache state of the runs."""
import argparse, importlib.util, json, os, re, statistics as st, sys
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import witness as W, audit_check as AC
ap = argparse.ArgumentParser(); ap.add_argument("taskdir"); ap.add_argument("runs"); ap.add_argument("--label"); ap.add_argument("--ref", type=float)
ap.add_argument("--calib", default=None)
a = ap.parse_args()
def load(name):
    p = os.path.join(a.taskdir, name + ".py")
    if not os.path.exists(p):
        return None
    spec = importlib.util.spec_from_file_location("task_" + name, p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
M = load("metric"); OC = load("outcheck"); SE = load("settings")
calib = {}
cp = a.calib or os.path.join(a.taskdir, "calib", "calib.json")
if os.path.exists(cp):
    try: calib = json.load(open(cp))
    except Exception: calib = {}
WTOL = (calib.get("witness") or {}).get("tolerance_s"); WMED = (calib.get("witness") or {}).get("median_residual_s")
NEED_END = os.environ.get("P2_AUDIT_NEED_END") == "1"; AIGN = os.environ.get("P2_AUDIT_IGNORE", "")
SIGN = set(os.environ.get("SETTINGS_IGNORE", "").split())
rx = re.compile(r"^(?P<label>.+)_(?P<kind>[a-z]+)(?P<ii>\d\d)_(?P<k>[12])_(?P<role>[a-z]+)$")
def jl(p):
    try: return json.load(open(p))
    except Exception: return {}
runs = {}
for d in sorted(os.listdir(a.runs)):
    full = os.path.join(a.runs, d)
    if not os.path.isdir(full) or ".aborted." in d: continue
    m = rx.match(d)
    if not m or (a.label and m["label"] != a.label) or m["kind"] == "warm": continue
    meta = jl(os.path.join(full, "job_meta.json"))
    try: met = M.metric(full)
    except Exception as e: met = {"error": repr(e)}
    r = {"meta": {k: meta.get(k) for k in ("rc", "aborted", "wall_s", "peak_mem_mib", "peak_frac", "storage", "cpus", "cache")},
         "metric": met, "dir": full, **m.groupdict()}
    flags = []
    if isinstance(met, dict) and met.get("witness_reported_s") is not None:
        w = W.witness(full, met["witness_reported_s"]); r["witness"] = w
        if WTOL is not None and w.get("residual_s") is not None and w["residual_s"] - (WMED or 0) > WTOL:   # one-sided
            flags.append("timing_witness")
    au = AC.events(full, NEED_END, AIGN); r["audit"] = au
    if au.get("events") or au.get("missing_end"):
        flags.append("audit")
    if SE:
        try: r["settings"] = SE.settings(full)
        except Exception as e: r["settings"] = {"error": repr(e)}
    vg = jl(os.path.join(full, "venv_guard.json"))
    if vg:
        r["venv_guard"] = vg
        if (vg.get("post") or {}).get("state") == "changed_by_run":
            flags.append("venv_changed")
    r["flags"] = flags
    runs[d] = r
groups = {}
for d, r in runs.items():
    groups.setdefault((r["label"], r["kind"], r["ii"]), {})[int(r["k"])] = r
out = {"runs": runs, "pairs": [], "by_kind": {}}
for (label, kind, ii), g in sorted(groups.items()):
    if 1 not in g or 2 not in g: continue
    x, y = g[1], g[2]
    px, py = (x["metric"] or {}).get("primary"), (y["metric"] or {}).get("primary")
    if not px or not py or x["meta"].get("rc") != 0 or y["meta"].get("rc") != 0: continue
    rec = {"label": label, "kind": kind, "pair": ii, "first": x["role"], "second": y["role"], "m_first": px, "m_second": py,
           "ratio_second_over_first": py / px}
    roles = sorted({x["role"], y["role"]})
    if len(roles) == 2:
        base = "parent" if "parent" in roles else ("parentvenv" if "parentvenv" in roles else roles[0])
        other = [r for r in roles if r != base][0]
        mb = px if x["role"] == base else py; mo = py if y["role"] == other else px
        rec["pair_roles"] = f"{other}_vs_{base}"; rec["speedup"] = mb / mo
        B, O = (x, y) if x["role"] == base else (y, x)
    else:
        rec["pair_roles"] = f"{roles[0]}_vs_{roles[0]}"; rec["speedup"] = px / py
        B, O = x, y
    flags = sorted(set(x["flags"]) | set(y["flags"]))
    if OC:
        try: oc = OC.check(O["dir"], B["dir"], a.taskdir)
        except Exception as e: oc = {"pass": False, "error": repr(e)}
        rec["outcheck"] = {k: v for k, v in oc.items() if k != "detail"}
        if not oc.get("pass"):
            flags.append("output_mismatch")
    if SE and isinstance(B.get("settings"), dict) and isinstance(O.get("settings"), dict):
        diff = {k: (B["settings"].get(k), O["settings"].get(k)) for k in set(B["settings"]) | set(O["settings"])
                if k not in SIGN and B["settings"].get(k) != O["settings"].get(k)}
        rec["settings_diff"] = diff
        if diff:
            flags.append("settings_mismatch")
    rec["flags"] = flags
    out["pairs"].append(rec)
for rec in out["pairs"]:
    key = f"{rec['label']}:{rec['pair_roles']}"
    out["by_kind"].setdefault(key, []).append(rec)
summ = {}
for k, recs in out["by_kind"].items():
    v = [r["speedup"] for r in recs]
    s = {"n": len(v), "median_speedup": st.median(v), "min": min(v), "max": max(v), "values": [round(x, 4) for x in v],
         "flagged_pairs": [{"pair": r["pair"], "flags": r["flags"]} for r in recs if r["flags"]]}
    if k.endswith(("parent_vs_parent", "parentvenv_vs_parentvenv")):
        s["max_abs_dev"] = max(abs(x - 1) for x in v)
    summ[k] = s
out["summary"] = summ
out["by_kind"] = {k: [r["speedup"] for r in recs] for k, recs in out["by_kind"].items()}
ok_runs = [r for r in runs.values() if r["meta"].get("rc") == 0]
stor = sorted({(r["meta"].get("storage") or "nas") for r in ok_runs})
out["storage"] = stor; out["mixed_storage"] = len(stor) > 1
out["cpus"] = sorted({str(r["meta"].get("cpus")) for r in ok_runs})
out["cache_states"] = sorted({str(r["meta"].get("cache")) for r in ok_runs})
if a.ref:
    for k, s in summ.items():
        if k.split(":")[1].startswith("cand_vs"):
            s["reward_vs_ref"] = max(0.0, min(1.0, (s["median_speedup"] - 1) / (a.ref - 1))) if a.ref > 1 else None
            s["speedup_over_ref"] = s["median_speedup"] / a.ref
print(json.dumps(out, indent=1, default=str))
