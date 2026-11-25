"""report.py : compact iso12380 numbers for the card (calibration, every scoring job, transfers, canaries, cost).
Usage: python3 -I report.py > report.json"""
import glob, json, os, statistics as st
TD = os.path.join(os.environ["ISO_ROOT"], "tasks", "iso12380")  # package: from env.sh
def jl(p):
    try: return json.load(open(p))
    except Exception: return None
out = {}
c = jl(f"{TD}/calib/calib.json") or {}
cs = jl(f"{TD}/calib/summary.json") or {}
cal_runs = [r for r in (cs.get("runs") or {}).values()]
out["calib"] = {"ref": c.get("ref_speedup"), "fix_pairs": (c.get("fix_pairs") or {}).get("values"),
                "aa": (c.get("aa") or {}).get("values"), "noise_band": c.get("noise_band"),
                "xv": (c.get("overlay_equivalence_fixvenv_vs_parentvenv") or {}).get("values"),
                "witness": {k: (c.get("witness") or {}).get(k) for k in ("n", "median_residual_s", "max_up_dev_s", "tolerance_s")},
                "outcheck_thresholds": {k: v for k, v in (c.get("outcheck_thresholds") or {}).items() if k != "basis"},
                "outcheck_basis": {k: v for k, v in ((c.get("outcheck_thresholds") or {}).get("basis") or {}).items() if k != "fix_vs_first_fix"},
                "parent_primary": c.get("parent_primary"), "flags": c.get("calibration_flags"),
                "audit_trusted": c.get("audit_events_trusted_runs"),
                "gpu_s": round(sum((r["meta"].get("wall_s") or 0) for r in cal_runs), 1), "n_runs_timed": len(cal_runs),
                "peak_mem_mib": max([r["meta"].get("peak_mem_mib") or 0 for r in cal_runs] or [0])}
for f in ("correct_ref2", "correct_fixc"):
    pass
log = open(f"{TD}/calib/calibrate.log").read() if os.path.exists(f"{TD}/calib/calibrate.log") else ""
out["calib"]["correct_lines"] = [l[:400] for l in log.splitlines() if l.startswith("correct ")]
out["calib"]["out_ref_line"] = [l[:300] for l in log.splitlines() if "out_ref:" in l]
jobs = []
for R in sorted(glob.glob(f"{TD}/results/*")):
    L = os.path.basename(R); s = jl(f"{R}/summary.json")
    if not s: continue
    rs = jl(f"{R}/runs_summary.json") or {}
    runs = rs.get("runs") or {}
    metas = []
    for d in glob.glob(f"{R}/runs/*/job_meta.json") + glob.glob(f"{R}/correct/job_meta.json"):
        m = jl(d)
        if m: metas.append(m)
    gpu_s = sum(m.get("wall_s") or 0 for m in metas)
    t0 = min((m.get("start") or 9e18) for m in metas) if metas else None; t1 = max((m.get("end") or 0) for m in metas) if metas else None
    sc = [v for v in runs.values() if v.get("label") == "sc" and v.get("kind") == "ab"]
    cm = [v["metric"]["primary"] for v in sc if v.get("role") == "cand" and v.get("metric")]
    pm = [v["metric"]["primary"] for v in sc if v.get("role") == "parent" and v.get("metric")]
    j = {"label": L, "verdict": s.get("verdict"), "speedup": s.get("speedup"), "reward": s.get("reward"), "flags": s.get("flags"),
         "pairs": (s.get("pairs") or {}).get("values"), "hosts": s.get("hosts"), "cand_mean_s": st.median(cm) if cm else None,
         "parent_mean_s": st.median(pm) if pm else None, "correct_pass": (s.get("correctness") or {}).get("pass"),
         "correct_exact": (s.get("correctness") or {}).get("exact"), "env_suspect": s.get("environment_suspect_pairs"),
         "integrity": [((s.get("integrity") or {}).get(k) or {}).get("ok") for k in ("pre", "post")],
         "gpu_s_runs": round(gpu_s, 1), "n_runs": len(metas), "span_s": round(t1 - t0, 1) if metas else None,
         "peak_mem_mib": max([m.get("peak_mem_mib") or 0 for m in metas] or [0]),
         "transfer": {k: {"speedup": v.get("median_speedup"), "flags": v.get("flagged_pairs")} for k, v in (s.get("transfer") or {}).items()},
         "pair_checks": [{"pair": p.get("pair"), "speedup": round(p.get("speedup") or 0, 4), "flags": p.get("flags"),
                          "out": {k: ((p.get("outcheck") or {}).get("outputs") or {}).get(k) for k in ("exact_frac", "bad", "bad_sample")},
                          "steps": {k: ((p.get("outcheck") or {}).get("steps") or {}).get(k) for k in ("pass", "graph_decode_frac_cand", "graph_decode_frac_base", "first_mismatch")},
                          "settings_diff": p.get("settings_diff")} for p in (s.get("pair_checks") or [])]}
    if s.get("verdict") in ("audit_flag", "timing_witness_fail"):
        j["audit_detail"] = {k: (v.get("audit") or {}).get("kinds") for k, v in runs.items() if (v.get("audit") or {}).get("events")}
        j["witness_detail"] = {k: (v.get("witness") or {}).get("residual_s") for k, v in runs.items()}
    if s.get("verdict") == "gate_reject":
        j["gate"] = open(f"{R}/gate.txt").read()[-800:] if os.path.exists(f"{R}/gate.txt") else None
    jobs.append(j)
out["jobs"] = jobs
fix = [j for j in jobs if j["label"].startswith("j") and j.get("speedup")]
if len(fix) >= 2:
    sp = [j["speedup"] for j in fix]; ref = st.median(sp); sd = st.stdev(sp)
    out["between_jobs"] = {"n": len(sp), "median": ref, "min": min(sp), "max": max(sp), "sd": sd, "rel_sd": sd / ref,
                           "band": max(0.005, 2 * sd / ref), "by_host": {h: [j["speedup"] for j in fix if h in str(j["hosts"])] for h in ("main", "n1")},
                           "parent_mean_by_job": {j["label"]: j["parent_mean_s"] for j in fix}, "cand_mean_by_job": {j["label"]: j["cand_mean_s"] for j in fix}}
print(json.dumps(out, indent=1, default=str))
