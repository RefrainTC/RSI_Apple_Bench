"""Summarise MM phase result dirs (verifier metrics + bench_rounds) -> one JSON + a text table.
usage: python -I summarize.py <out.json> <phase_dir> [...]   (summarize.py with our paths / mem-log format)"""
import json, sys
from pathlib import Path
import os
L = Path(os.environ.get("RSI_ROOT") or ".") / "mm" / "logs"
out = {}
for d in map(Path, sys.argv[2:]):
    try:
        m = json.loads((d / "metrics.json").read_text())
        r = json.loads((d / "bench_rounds.json").read_text())
        rew = json.loads((d / "verifier" / "reward.json").read_text())
    except Exception as e:  # noqa: BLE001
        print(f"== {d.name}: unreadable ({e})"); continue
    pc = m["performance_checks"]
    mem = d / f"mem_{d.name}.log"
    if not mem.exists():
        mem = L / f"mem_{d.name}.log"
    peak = None
    if mem.exists():
        vals = [int(x.split(",")[1]) for x in mem.read_text().splitlines() if x.count(",") >= 2 and x.split(",")[1].strip().isdigit()]
        peak = max(vals) if vals else None
    meta = (d / "run_meta.txt").read_text() if (d / "run_meta.txt").exists() else ""
    wall = [l.split("wall_s=")[1].strip() for l in meta.splitlines() if "wall_s=" in l]
    row = {
        "reward": rew.get("reward"), "speedup_geomean": pc.get("speedup_vs_baseline"),
        "speedup_per_workload": pc.get("speedup_per_workload"),
        "guards": {k: round(v["ratio"], 4) for k, v in (pc.get("guard_ratios") or {}).items()},
        "guards_ok": pc.get("guards_ok"), "reason": m.get("reason"),
        "ttft_ms": {n: {a: [round(x, 1) for x in w["arms"][a]["runs"]] for a in ("pristine", "candidate")}
                    for n, w in r["workloads"].items()},
        "warmup_runs_ms": [round(x, 1) for x in r.get("warmup_runs") or [] if x],
        "cubins": {"pristine": m["regression_repair_checks"].get("pristine_triton_cubins"),
                   "candidate": m["regression_repair_checks"].get("candidate_triton_cubins")},
        "outputs_pass_rate": m["correctness_checks"].get("pass_rate"),
        "max_abs_logprob_delta": m["correctness_checks"].get("max_abs_delta"),
        "peak_gpu_mem_mib": peak, "wall_s": wall[0] if wall else None,
    }
    out[d.name] = row
    print(f"== {d.name}: reward={row['reward']} speedup={row['speedup_geomean']} reason={row['reason']} "
          f"peak={peak} MiB cubins={row['cubins']} pass_rate={row['outputs_pass_rate']} wall={row['wall_s']}s")
    for n, s in (row["speedup_per_workload"] or {}).items():
        print(f"   {n:20s} {s:.3f}  pristine={row['ttft_ms'][n]['pristine']} candidate={row['ttft_ms'][n]['candidate']}")
    for n, g in row["guards"].items():
        print(f"   guard {n:14s} {g:.3f}  pristine={row['ttft_ms'][n]['pristine']} candidate={row['ttft_ms'][n]['candidate']}")
Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + "\n")
