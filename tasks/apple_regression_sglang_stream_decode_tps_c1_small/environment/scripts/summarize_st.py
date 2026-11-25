"""summarize_st.py <out.json> <run dir> [...] : ST verifier runs -> one JSON + a text table .
Per run: reward, reason, geomean, per-point ratios, guards, floor-probe classes per arm/round, output equivalence,
harness used, wall time, peak GPU memory (from $RSI_LOGS/mem_st_<run>.log or the copy in the run dir)."""
import json
import sys
from pathlib import Path

import os
L = Path(os.environ.get("RSI_ROOT") or ".") / "st" / "logs"
out = {}
for d in map(Path, sys.argv[2:]):
    try:
        m = json.loads((d / "metrics.json").read_text())
        rew = json.loads((d / "verifier" / "reward.json").read_text())
    except Exception as e:  # noqa: BLE001
        print(f"== {d.name}: unreadable ({e})"); continue
    p, c = m["performance_checks"], m["candidate_checks"]
    fl = c.get("min_new_tokens_floor") or {}
    meta = (d / "run_meta.txt").read_text() if (d / "run_meta.txt").exists() else ""
    wall = [l.split("wall_s=")[1].strip() for l in meta.splitlines() if "wall_s=" in l]
    harness = [l.split("harness=")[1].strip() for l in meta.splitlines() if "harness=" in l]
    peak = None
    for mem in (d / f"mem_st_{d.name}.log", L / f"mem_st_{d.name}.log"):
        if mem.exists():
            v = [int(x.split(",")[1]) for x in mem.read_text().splitlines()
                 if x.count(",") >= 2 and x.split(",")[1].strip().isdigit()]
            peak = max(v) if v else None
            break
    row = {"reward": rew.get("reward"), "reason": m.get("reason"), "speedup": p.get("speedup_vs_baseline"),
           "per_point": p.get("speedup_per_workload"),
           "guards": {k: round(v["ratio"], 4) for k, v in (p.get("guard_ratios") or {}).items()},
           "output_equivalent": c.get("output_equivalent"), "floor_broken": fl.get("floor_broken"),
           "floor": [(a["arm"], a["round"], a["classification"], a.get("generated")) for a in fl.get("arms", [])],
           "harness": harness[0] if harness else None, "wall_s": wall[0] if wall else None, "peak_mib": peak}
    out[d.name] = row
    print(f"== {d.name}: reward={row['reward']} reason={row['reason']} speedup={row['speedup']:.4f} "
          f"guards={row['guards']} out_eq={row['output_equivalent']} peak={peak} wall={row['wall_s']} "
          f"harness={Path(row['harness']).name if row['harness'] else None}")
    print(f"   points={ {k: round(v, 4) for k, v in (row['per_point'] or {}).items()} }")
    print(f"   floor={sorted(set((a, cl, g) for a, _, cl, g in row['floor']))}")
Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + "\n")
