"""Config screening: base, s1, ..., sN, base — one session each, identical seeds for every session.

  python -I sweep.py --task TASKDIR --stages a,b,c --label L [--wlset workloads] [--seed N]

Stage names refer to tasks/<t>/sweep/<name> (or 'pristine'). Ratio per workload = stage / mean(base
sessions); the two base sessions also give a drift estimate. Screening only: the reference is then
confirmed with score.py (alternating rounds).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3lib  # noqa: E402
from score import session, trimmed_mean, geomean  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--stages", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--wlset", default="workloads")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-base-end", action="store_true")
    ap.add_argument("--counters", action="store_true")
    a = ap.parse_args()
    p3lib.install_signal_handlers()  # v5d: SIGINT / SIGTERM stop every live server, exit 130 / 143
    tdir = Path(a.task)
    task = p3lib.load_task(tdir)
    hidden = json.loads((tdir / "scoring" / "hidden.json").read_text())
    wls = hidden[a.wlset]
    seed = a.seed if a.seed is not None else (int(time.time() * 1000) & 0x7FFFFFFF)
    out = tdir / "results" / a.label
    out.mkdir(parents=True, exist_ok=True)
    p3lib.ensure_staged(task)
    scratch = p3lib.scratch_of(task, a.label, out)
    ref = tdir / "scoring" / "corr_ref.json"
    if ref.exists():
        (out / "corr_ref.json").write_text(ref.read_text())
    logf = open(out / "sweep.log", "a")

    def log(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    names = ["pristine"] + [s for s in a.stages.split(",") if s] + ([] if a.no_base_end else ["pristine"])
    overlays = {}
    recs = []
    integ0 = p3lib.tree_digest(p3lib.integrity_roots(task))
    log(f"sweep {a.label}: {names} wlset={a.wlset} seed={seed}")
    for k, nm in enumerate(names):
        st = tdir / "pristine" if nm == "pristine" else (tdir / nm if "/" in nm else tdir / "sweep" / nm)
        key = os.path.realpath(st / "repo")
        if key not in overlays:
            overlays[key] = p3lib.build_overlay(st, scratch / "overlay" / f"o{len(overlays)}", p3lib.site_vllm_of(task))
        arm = f"{k:02d}_{nm.replace('/', '_')}"
        try:
            rec = session(task, st, overlays[key], arm, 1, seed, out, wls, nm != "pristine", 1.0, log,
                          counters=a.counters, scratch=scratch, integ0=integ0)
            for c in rec.get("sched_counters", []):
                log(f"  [{arm}] sched counters: " + json.dumps(c))
        except Exception as e:  # e.g. rejected submission
            rec = {"arm": arm, "error": repr(e)}
        rec["name"] = nm
        recs.append(rec)
        (out / "sessions.json").write_text(json.dumps(recs, indent=1))
        if rec.get("error"):
            log(f"  {nm}: ERROR {rec['error']}")
    base = [r for r in recs if r["name"] == "pristine" and not r.get("error")]
    table = {}
    for r in recs:
        if r.get("error"):
            table[r["arm"]] = {"error": r["error"]}
            continue
        row = {}
        for wl in wls:
            b = [x for br in base for x in br.get("wl_scores", {}).get(wl["name"], []) if x]
            c = [x for x in r.get("wl_scores", {}).get(wl["name"], []) if x]
            row[wl["name"]] = (trimmed_mean(c) / (sum(b) / len(b))) if (b and c) else None
        row["geomean"] = geomean([v for v in row.values() if v]) if all(row.values()) else None
        row["correctness_ok"] = (r.get("correctness") or {}).get("ok")
        row["peak_mib"] = max((r.get("peak_mib") or {"x": 0}).values())
        row["flags"] = sorted({f["kind"] for f in r.get("flags", [])})
        c = r.get("correctness") or {}
        row.update({k: c.get(k) for k in ("pass_rate", "decode_pass_rate", "tf_mean_abs", "dec_mean_abs") if k in c})
        table[r["arm"]] = row
        log(f"  {r['arm']}: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
    (out / "summary.json").write_text(json.dumps({"label": a.label, "seed": seed, "wlset": a.wlset, "table": table,
                                                  **p3lib.storage_meta(task)}, indent=1))
    if scratch != out:
        p3lib.rmtree_rw(scratch)
    log("SWEEP_DONE")


if __name__ == "__main__":
    main()
