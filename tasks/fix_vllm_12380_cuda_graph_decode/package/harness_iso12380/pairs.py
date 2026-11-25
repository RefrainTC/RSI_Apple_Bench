#!/usr/bin/env python3
"""pairs.py <taskdir> <runs_dir> <label> --plan "ab:cand:parent:2,aa:parent:parent:1" [--workload "K=V K=V"] [--mode timed]
           [--stage DIR] [--gpu N]
Runs interleaved pairs through run_variant.sh. Plan items kind:roleA:roleB:n. For kind 'ab' the order alternates
(A,B),(B,A),... ; for 'aa' it is always (A,B). Pairs of different plan items are interleaved round-robin.
Tags: <label>_<kind><ii>_<k>_<role>. Retries a job when the GPU was busy (rc 3) or a foreign process appeared (rc 4).
Stops on the VRAM watchdog (rc 5). Prints one JSON line per job; the summary is done by summarize.py."""
import argparse, json, os, subprocess, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("taskdir"); ap.add_argument("runs"); ap.add_argument("label")
ap.add_argument("--plan", required=True); ap.add_argument("--workload", default=""); ap.add_argument("--mode", default="timed")
ap.add_argument("--stage", default="-")
ap.add_argument("--tag-prefix", default="", help="job tag prefix (scratch dir, overlay, R4_TAG); default p2_<task>_<results dir>")
ap.add_argument("--warmup", default="", help="comma list of roles, one untimed warm-up job each before the schedule (tag <label>_warm00_<k>_<role>)")
ap.add_argument("--warmup-workload", default="", help="extra K=V for warm-up jobs only (a different prompt set than the timed runs)")
a = ap.parse_args()
H = os.path.dirname(os.path.abspath(__file__))  # package: the harness dir itself
plan = []
for item in a.plan.split(","):
    kind, ra, rb, n = item.split(":"); plan.append((kind, ra, rb, int(n)))
sched = []
for i in range(max(p[3] for p in plan)):
    for kind, ra, rb, n in plan:
        if i < n:
            order = (ra, rb) if (kind != "ab" or i % 2 == 0) else (rb, ra)
            sched.append((kind, i + 1, order))
os.makedirs(a.runs, exist_ok=True)
if a.warmup:
    sched.insert(0, ("warm", 0, tuple(a.warmup.split(","))))
def done(out):
    try:
        m = json.load(open(os.path.join(out, "job_meta.json"))); return m.get("rc") == 0 and m.get("aborted") == ""
    except Exception:
        return False
for kind, i, order in sched:
    for k, role in enumerate(order, 1):
        if os.path.exists(os.path.join(os.environ.get("ISO_ROOT", os.path.dirname(H)), "STOP_iso12380")):
            print(json.dumps({"stop": "STOP_iso12380"}), flush=True); sys.exit(8)
        tag = f"{a.label}_{kind}{i:02d}_{k}_{role}"; out = os.path.join(a.runs, tag)
        if done(out):
            continue
        for attempt in range(8):
            stage = a.stage if role == "cand" else "-"
            # job tag namespaced by task and results dir (F6): two tasks / two scorings never share a scratch dir, overlay or
            # R4_TAG, so one job's cleanup can never touch another's processes or files
            jtag = (a.tag_prefix or f"p2_{os.path.basename(os.path.abspath(a.taskdir))}_{os.path.basename(os.path.dirname(os.path.abspath(a.runs)))}") + "_" + tag
            wl = a.workload.split() + (a.warmup_workload.split() if kind == "warm" else [])
            cmd = ["bash", f"{H}/run_variant.sh", a.taskdir, role, stage, jtag, out, a.mode] + wl
            env = dict(os.environ, P2_CACHE_WARMUP="1" if kind == "warm" else "0")  # warm-up jobs write the in-job cache master
            t = time.time(); rc = subprocess.call(cmd, stdout=open(out + ".log", "w") if os.path.isdir(a.runs) else None, stderr=subprocess.STDOUT, env=env)
            print(json.dumps({"tag": tag, "rc": rc, "s": round(time.time() - t, 1), "attempt": attempt}), flush=True)
            if rc in (3, 4, 9):  # 3 GPU busy, 4 foreign process, 9 port still taken
                if os.path.isdir(out):
                    os.rename(out, f"{out}.aborted.{int(time.time())}")
                time.sleep(120); continue
            if rc == 8:
                print(json.dumps({"stop": "STOP_iso12380", "tag": tag}), flush=True); sys.exit(8)
            if rc == 5:
                print(json.dumps({"stop": "vram_watchdog", "tag": tag}), flush=True); sys.exit(5)
            break
print(json.dumps({"done": a.label}), flush=True)
