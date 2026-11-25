#!/usr/bin/env python3
"""calib_report.py <taskdir> [runs_dir=calib/runs] [label=cal] : compact text report of a calibration (or any pairs dir):
pairs with per-rep medians, A/A, outcheck / counter equality, witness residuals, accepted tokens, cost and peak VRAM."""
import glob, json, os, statistics as st, sys
td = sys.argv[1]; rd = sys.argv[2] if len(sys.argv) > 2 else os.path.join(td, "calib", "runs"); lab = sys.argv[3] if len(sys.argv) > 3 else "cal"
S = json.load(open(os.path.join(td, "calib", "summary.json"))) if rd.endswith("calib/runs") and os.path.exists(os.path.join(td, "calib", "summary.json")) else None
if S is None:
    import subprocess
    S = json.loads(subprocess.check_output(["python3", "-I", os.path.join(os.environ["ISO_ROOT"], "harness_iso13365", "summarize.py"), td, rd]))
for k, v in S["summary"].items():
    print(f"{k}: n={v['n']} median={v['median_speedup']:.4f} min={v['min']:.4f} max={v['max']:.4f} values={v['values']} flagged={v['flagged_pairs']}" + (f" max_abs_dev={v['max_abs_dev']:.4f}" if 'max_abs_dev' in v else ""))
for p in S["pairs"]:
    oc = p.get("outcheck") or {}
    print(f"  {p['label']} {p['kind']}{p['pair']} {p['first']}->{p['second']} {p['m_first']:.4f} {p['m_second']:.4f} speedup={p['speedup']:.4f} oc={oc.get('pass')} ref={oc.get('ref')} exact={oc.get('exact')}/{oc.get('total')} core_eq={oc.get('core_equal')} parent_matches_ref={oc.get('parent_matches_ref')} flags={p['flags']}")
runs = S["runs"]
res = [r["witness"]["residual_s"] for r in runs.values() if (r.get("witness") or {}).get("residual_s") is not None]
if res: print("witness residuals", [round(x, 3) for x in res])
by = {}
for n, r in runs.items():
    m = r.get("metric") or {}
    if m.get("timed_s"): by.setdefault(r["role"], []).append(m["timed_s"])
for role, L in by.items():
    per = list(zip(*L)); print(role, "n", len(L), "per-rep medians", [round(st.median(x), 4) for x in per], "sum median", round(st.median([sum(x) for x in L]), 4), "sum range", round(min(sum(x) for x in L), 4), round(max(sum(x) for x in L), 4))
metas = [json.load(open(f)) for f in glob.glob(os.path.join(rd, "*", "job_meta.json"))]
if metas:
    w = [float(m["wall_s"]) for m in metas]
    print("runs", len(metas), "job wall median", round(st.median(w), 1), "sum GPU-min", round(sum(w) / 60, 1), "peak MiB", max(int(m["peak_mem_mib"]) for m in metas), "cpus", sorted({m["cpus"] for m in metas}))
