#!/usr/bin/env python3
"""calib_summary.py <summary.json> <calib.json> : calibration record from summarize.py's output (REF, noise band, witness
tolerance, parent settings / median, scored-output checks, flags, audit events on trusted runs). Keeps an existing
calib.json's outcheck_thresholds (written by calib_outputs.py), so it can be re-run after the thresholds exist."""
import json, statistics as st, sys
S = json.load(open(sys.argv[1])); s = S["summary"]
ref = s.get("cal:fix_vs_parent", {}).get("median_speedup"); aa = s.get("cal:parent_vs_parent", {})
c = {"ref_speedup": ref, "fix_pairs": s.get("cal:fix_vs_parent"), "aa": aa, "noise_band": aa.get("max_abs_dev"),
     "overlay_equivalence_fixvenv_vs_parentvenv": s.get("cal:fixvenv_vs_parentvenv"),
     "storage": S.get("storage"), "mixed_storage": S.get("mixed_storage"), "cpus": S.get("cpus"), "cache_states": S.get("cache_states")}
runs = [r for r in S["runs"].values() if r["meta"].get("rc") == 0 and r["role"] in ("parent", "fix")]
res = [r["witness"]["residual_s"] for r in runs if (r.get("witness") or {}).get("residual_s") is not None]
if res:   # witness tolerance from trusted code, one-sided: forging a faster pass makes the residual LARGER, while a
    # smaller residual only means fewer busy samples (nvidia-smi hiccup) or over-reported time. 3x the largest upward
    # deviation from the median, at least 0.5 s.
    med = st.median(res); up = max([x - med for x in res if x > med] or [0.0]); lo = max([med - x for x in res if x < med] or [0.0])
    c["witness"] = {"n": len(res), "median_residual_s": med, "max_up_dev_s": up, "max_down_dev_s": lo,
                    "tolerance_s": max(0.5, 3 * up), "one_sided": True, "residuals": [round(x, 3) for x in res]}
pp = [r for r in runs if r["role"] == "parent"]
if pp and isinstance(pp[0].get("settings"), dict):
    c["parent_settings"] = pp[0]["settings"]
c["parent_primary"] = {"median": st.median([r["metric"]["primary"] for r in pp]), "n": len(pp)} if pp else None
c["outcheck"] = {k: [p.get("outcheck", {}).get("pass") for p in S["pairs"] if f"{p['label']}:{p['pair_roles']}" == k]
                 for k in s}
c["outcheck_detail"] = [{"pair": p["pair"], "roles": p["pair_roles"], **{x: y for x, y in (p.get("outcheck") or {}).items() if x in ("pass", "identical_frac", "pass0", "pass1", "parent_matches_ref", "ref")}} for p in S["pairs"]]
# scored-output threshold from the human fix (offline tasks: per pass exact fraction; serving: identical texts)
fx = [p.get("outcheck") or {} for p in S["pairs"] if p["pair_roles"] == "fix_vs_parent"]
vals = [o[k]["exact_frac"] for o in fx for k in ("pass0", "pass1") if isinstance(o.get(k), dict)] or [o["identical_frac"] for o in fx if "identical_frac" in o]
if vals:
    c["fix_output_exact_min"] = min(vals); c["out_min_exact_suggested"] = max(0.05, min(vals) - 0.1)   # task.env OUT_MIN_EXACT wins
c["calibration_flags"] = [{"pair": p["pair"], "roles": p["pair_roles"], "flags": p["flags"]} for p in S["pairs"] if p["flags"]]
c["audit_events_trusted_runs"] = {r["role"] + ":" + r["ii"] + "_" + r["k"]: r["audit"].get("kinds") for r in runs if (r.get("audit") or {}).get("events")}
import os
if os.path.exists(sys.argv[2]):
    try:
        old = json.load(open(sys.argv[2]))
        if "outcheck_thresholds" in old: c["outcheck_thresholds"] = old["outcheck_thresholds"]
    except Exception:
        pass
json.dump(c, open(sys.argv[2], "w"), indent=1); print(json.dumps({k: c[k] for k in ("ref_speedup", "noise_band", "storage")}))
