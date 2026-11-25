#!/usr/bin/env python3
"""report.py : one-shot summary of #14930 results (calibration, scoring jobs, transfers, canaries, cost, VRAM)."""
import glob, json, os, re, statistics as st, sys, datetime as dt
TD = os.path.join(os.environ["ISO_ROOT"], "tasks", "iso14930")  # package: from env.sh; R = os.path.join(TD, "results")
c = json.load(open(os.path.join(TD, "calib", "calib.json")))
print("CALIB", json.dumps({k: c.get(k) for k in ("ref_speedup", "noise_band", "parent_primary", "outcheck_thresholds", "calibration_flags", "audit_events_trusted_runs", "storage", "cpus")}))
print("CALIB fix pairs", c["fix_pairs"]["values"], "aa", c["aa"]["values"], "xv", (c.get("overlay_equivalence_fixvenv_vs_parentvenv") or {}).get("values"))
print("CALIB witness", json.dumps(c.get("witness")))
CCD = {"0-7,32-39": "0", "8-15,40-47": "1", "16-23,48-55": "2", "24-31,56-63": "3"}
def tparse(l):
    try: return dt.datetime.strptime(l[:19], "%Y-%m-%d %H:%M:%S")
    except Exception: return None
for d in sorted(glob.glob(os.path.join(R, "*"))):
    L = os.path.basename(d)
    try: s = json.load(open(os.path.join(d, "summary.json")))
    except Exception: print("JOB", L, "no summary"); continue
    lines = open(os.path.join(d, "score.log")).read().splitlines() if os.path.exists(os.path.join(d, "score.log")) else []
    ts = [t for t in (tparse(x) for x in lines) if t]
    dur = (ts[-1] - ts[0]).total_seconds() / 60 if len(ts) > 1 else None
    runs = glob.glob(os.path.join(d, "runs", "*", "job_meta.json")) + glob.glob(os.path.join(d, "correct", "job_meta.json"))
    metas = [json.load(open(x)) for x in runs]
    gpu_s = sum(m.get("wall_s", 0) for m in metas); peak = max([m.get("peak_mem_mib", 0) for m in metas] or [0])
    tr = {k: (v.get("median_speedup"), v.get("values")) for k, v in (s.get("transfer") or {}).items()}
    trf = {}
    try:
        rs = json.load(open(os.path.join(d, "runs_summary.json")))
        for p in rs["pairs"]:
            if p["label"].startswith("tr"): trf[p["label"]] = (p.get("flags"), (p.get("outcheck") or {}).get("why"))
    except Exception: pass
    print("JOB", json.dumps({"label": L, "gpu": CCD.get((s.get("cpus") or ["?"])[0], "?"), "verdict": s.get("verdict"), "speedup": s.get("speedup"),
          "pairs": (s.get("pairs") or {}).get("values"), "flags": s.get("flags"), "corr": (s.get("correctness") or {}).get("pass"),
          "corr_why": (s.get("correctness") or {}).get("why"), "greedy_exact": ((s.get("correctness") or {}).get("greedy") or {}).get("frac"),
          "sampled_all_rel": (((s.get("correctness") or {}).get("sampled") or {}).get("mean_tokens_per_step") or {}).get("all", {}).get("rel"),
          "sampled_lure_rel": (((s.get("correctness") or {}).get("sampled") or {}).get("mean_tokens_per_step") or {}).get("lure", {}).get("rel"),
          "pair_checks": [(p.get("pair"), p.get("flags"), ((p.get("outcheck") or {}).get("why") or [])[:3], (p.get("outcheck") or {}).get("identical_frac"), (p.get("outcheck") or {}).get("min_same_inc_frac")) for p in s.get("pair_checks") or []],
          "transfer": tr, "transfer_flags": trf, "env_suspect": s.get("environment_suspect_pairs"), "integrity": [(s.get("integrity") or {}).get("pre", {}).get("ok"), (s.get("integrity") or {}).get("post", {}).get("ok")],
          "job_min": round(dur, 1) if dur else None, "gpu_run_min": round(gpu_s / 60, 1), "n_runs": len(metas), "peak_mib": peak}))
