"""build_jit_master.py --task T --out $P3_STAGE/run/jit_master_<name> : a warm Triton / inductor cache made by the
PRISTINE server only, for task.json "jit_master".

The first shared cache ($P3_STAGE/run/shared_cache) was filled by every exploration run, so its autotune records
name the token budgets we tried ((4096|8192|12288|16384, 32, 128) buffers): a candidate reading the cache would
learn which knob was explored. This master holds only what the default configuration compiles: one pristine
server, the correctness probe, then the dev workloads (warmup + one run each). Candidate configurations that need
other shapes compile them into their private copy (startup is not timed).

v5d: exits non-zero (and says why) when the server never becomes ready, when any warm-up / dev run fails (no client
result, a client error, failed requests, or a dev run whose output check fails), or when the logprob gate fails.
Gate self-check: with scoring/corr_ref.json present, after the warm-up runs and on the same server (it compiled into
and now runs on the new master), correctness.check as score.py runs it (pass_rate gate plus the task's
drift_max_tf_mean_abs bound). A timing-based Inductor autotune choice stored in a cache can make an honest server
fail the gate deterministically (pass_rate 0.40): such a master must not be used. The probe before the warm-up
(as in v5c) is held to the same gate. --no-gate-check: the v5c probe only (printed, not gated), no self-check.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import correctness  # noqa: E402
import p3lib  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--task", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--no-gate-check", action="store_true",
                help="v5c behaviour for the logprob probe: print it, do not gate on it, no post-warm-up self-check")
a = ap.parse_args()
p3lib.install_signal_handlers()  # v5d: SIGINT / SIGTERM stop the server, exit 130 / 143
tdir, new = Path(a.task), Path(a.out)
task = p3lib.load_task(tdir)
p3lib.ensure_staged(task)
if new.exists():
    sys.exit(f"{new} exists")
new.mkdir(parents=True)
task["jit_master"] = str(new)
work = Path(task["scratch"]) / "build_master" / new.name
log_dir = tdir / "logs" / f"build_{new.name}"
log_dir.mkdir(parents=True, exist_ok=True)
ov = p3lib.build_overlay(tdir / "pristine", work / "overlay", p3lib.site_vllm_of(task))
srv = p3lib.Server(task, tdir / "pristine", ov, log_dir / "server.log", work / "cache", "master", private_cache=False)
ref_path = tdir / "scoring" / "corr_ref.json"
ref = json.loads(ref_path.read_text()) if ref_path.exists() else None
fails = []  # v5d: every reason this master must not be used


def gate(c: dict, when: str) -> None:
    """The scorer's view of the pristine server: pass_rate gate, then the task's drift bound (score.py session)."""
    ok = bool(c["ok"])
    lim = task.get("drift_max_tf_mean_abs")
    drift = lim is not None and c.get("tf_mean_abs") is not None and c["tf_mean_abs"] > lim
    print(f"gate {when}: pass_rate={c['pass_rate']} tf_mean_abs={c.get('tf_mean_abs')} ok={ok}"
          + (f" DRIFT tf_mean_abs > {lim}" if drift else ""), flush=True)
    if (not ok or drift) and not a.no_gate_check:
        fails.append(f"logprob gate {when}: pass_rate={c['pass_rate']} ok={ok} tf_mean_abs={c.get('tf_mean_abs')}"
                     f" (drift limit {lim})")


def run_problem(r, measured: bool):
    if not r or not r.get("_has_result"):
        return "no client result"
    if r.get("_err"):
        return f"client error {str(r['_err'])[:160]}"
    if r.get("failed"):
        return f"{r['failed']} failed requests"
    if measured and not (r.get("_out_check") or {}).get("ok"):
        return f"output check {json.dumps(r.get('_out_check'))[:200]}"
    return None


t0 = time.time()
srv.start()
try:
    if not srv.wait_ready(1500):
        sys.exit(f"FAILED: server not ready (killed={srv.killed_reason}, rc={srv.proc.poll()}); "
                 f"do not use {new}")
    print(f"ready in {srv.ready_s:.0f}s", flush=True)
    if ref is not None:
        c = correctness.check(srv.base_url, task["served_name"], ref)
        print("correctness", c["pass_rate"], c["ok"], flush=True)
        gate(c, "before warm-up")
    else:
        print(f"no {ref_path}: correctness probe and gate self-check skipped", flush=True)
    wls = json.loads((tdir / "scoring" / "dev.json").read_text())["workloads"]
    jobs = []
    for wi, wl in enumerate(wls):
        prof = (wl.get("profiles") or [None])[-1]
        w = wl.get("warmup", {})
        jobs.append({"argv": p3lib.bench_argv(task, srv.base_url, wl, prof, 777 + wi, int(w.get("num_prompts", 2)),
                                              w.get("output_len"), int(w.get("concurrency", 32))),
                     "out": str(log_dir / f"warm_{wl['name']}.json")})
        jobs.append({"argv": p3lib.bench_argv(task, srv.base_url, wl, prof, 888 + wi,
                                              int((prof or {}).get("num_prompts", wl["num_prompts"]))),
                     "out": str(log_dir / f"bench_{wl['name']}.json")})
    res = p3lib.run_driver(task, srv.base_url, jobs, log_dir / "jobs.json", log_dir / "driver.log")
    print("runs", [(r or {}).get("request_throughput") for r in res], flush=True)
    for j, r in zip(jobs, res):
        why = run_problem(r, Path(j["out"]).name.startswith("bench_"))
        if why:
            fails.append(f"run {Path(j['out']).name}: {why}")
    if ref is not None and not a.no_gate_check:
        gate(correctness.check(srv.base_url, task["served_name"], ref), "after warm-up (self-check)")
finally:
    srv.stop()
if srv.killed_reason:
    fails.append(f"server killed by the watchdog: {srv.killed_reason}")
p3lib.rmtree_rw(work)
n = sum(1 for _ in new.rglob("*") if _.is_file())
if fails:
    for f in fails:
        print(f"FAILED: {f}", file=sys.stderr, flush=True)
    sys.exit(f"JIT master {new}: {n} files, {time.time() - t0:.0f}s, {len(fails)} failure(s): DO NOT USE it "
             f"(see {log_dir})")
print(f"JIT master {new}: {n} files, {time.time() - t0:.0f}s. Set task.json jit_master to it.")
