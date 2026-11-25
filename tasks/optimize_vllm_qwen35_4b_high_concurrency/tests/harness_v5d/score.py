"""P3 paired scorer (Apple throughput_optimization recipe, scaled to one 4090).

  python -I score.py --task TASKDIR --cand STAGE [--base STAGE] --label L [--rounds 2]
                     [--wlset workloads] [--seed N] [--no-correctness] [--runs-scale 1]

A stage is a dir with submission/{extra_args.txt,env.txt} and repo/vllm (full .py tree).
Arms alternate per round (r1: base,cand; r2: cand,base; ...). Each arm session = fresh server,
correctness check (cand), 1 warmup per workload, then the measured runs. Every measured run uses
its own seed (fresh random-token prompts, never repeated within an arm); base and cand share the
seeds of a round. Per workload: trimmed mean over all runs of an arm, speedup = cand/base
(lower-is-better metrics inverted), score = geomean over workloads.
"""
from __future__ import annotations

import argparse
import json
import re
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3lib  # noqa: E402
import correctness  # noqa: E402
import witness  # noqa: E402


def trimmed_mean(v: list[float]) -> float | None:
    v = sorted(x for x in v if x)
    if not v:
        return None
    if len(v) >= 3:
        v = v[1:-1]
    return sum(v) / len(v)


def geomean(v: list[float]) -> float | None:
    v = [x for x in v if x and x > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else None


def seed_for(base_seed: int, rnd: int, wi: int, i: int, pi: int) -> int:
    return (base_seed + 100000 * rnd + 1000 * wi + 10 * i + pi) & 0x7FFFFFFF


def session(task, stage, overlay, arm, rnd, base_seed, out, wls, do_corr, runs_scale, log, counters=False,
            scratch=None, integ0=None, fresh_required=False):
    """One server session. Besides the numbers, rec["flags"] lists every check that fired:
    audit (runtime hook events), lingering_process, integrity (venv / model / harness / hidden files / JIT
    cache master changed), result_tamper (a result file changed after the driver wrote it), output (scored
    output wrong: truncated, empty, errors), sentinel (numerics setting read back from the server log),
    numerics_drift (logprob drift above the calibrated bound), correctness (Apple's logprob gate),
    audit_missing (fewer than two server processes reported through the audit FIFO).
    Witness (task["witness"]): samples of every measured run are collected here and teacher-forced later in a
    dedicated pristine witness session (witness_session; witness.py)."""
    trusted = arm in ("base", "pristine")
    sdir = out / f"r{rnd}_{arm}"
    sdir.mkdir(parents=True, exist_ok=True)
    cache = (scratch or out) / "cache" / arm  # one compile cache per arm, reused across its rounds
    rec = {"arm": arm, "round": rnd, "stage": str(stage), "runs": [], "t_start": time.time(), "flags": []}
    ok_free = p3lib.wait_gpu_free(task["gpus"], 1500, 900)
    rec["gpu_free_before"] = ok_free
    srv = p3lib.Server(task, stage, overlay, sdir / "server.log", cache, arm, audit_dir=sdir / "audit",
                       run_tag=f"p3_{out.name}_{arm}_r{rnd}_{int(time.time())}", private_cache=bool(task.get("scratch")))
    results = []
    if counters:
        srv.counters_dir = sdir / "counters"
    srv.start()
    try:
        if not srv.wait_ready(float(task.get("ready_timeout_s", 1200))):
            rec["error"] = f"server not ready (killed={srv.killed_reason}, rc={srv.proc.poll()})"
            log(rec["error"])
            return rec
        rec["ready_s"] = round(srv.ready_s, 1)
        log(f"  [{arm} r{rnd}] ready in {srv.ready_s:.0f}s port {srv.port}")
        if do_corr and not (out / "corr_ref.json").exists():  # fail closed: no reference = not checked = void
            rec["flags"].append({"kind": "correctness", "detail": "no corr_ref.json for this run"})
        elif do_corr:
          try:
            c = correctness.check(srv.base_url, task["served_name"], json.loads((out / "corr_ref.json").read_text()))
            # second, per-run set: random-token prompts like the workload's, referenced by this run's own
            # baseline (made in the first baseline session), so a server cannot special-case the public prompts
            fr = out / "fresh_ref.json"
            if arm == "base" and not fr.exists() and task.get("fresh_corr", True):
                import random
                rng = random.Random(base_seed ^ 0x5EED)
                ids = [[rng.randrange(1000, 150000) for _ in range(192)] for _ in range(8)]
                fr.write_text(json.dumps(correctness.make_ref_ids(srv.base_url, task["served_name"], ids, 64)))
            elif not trusted and not fr.exists() and fresh_required:
                c["ok"] = False
                c["fresh"] = {"ok": False, "error": "no fresh reference (baseline session did not make one)"}
            elif not trusted and fr.exists():
                c2 = correctness.check(srv.base_url, task["served_name"], json.loads(fr.read_text()))
                c["fresh"] = {k: c2.get(k) for k in ("pass_rate", "ok", "tf_mean_abs", "decode_pass_rate",
                                                     "greedy_identical", "n_prompts")}
                if not c2["ok"]:
                    c["ok"] = False
                limf = task.get("drift_max_tf_mean_abs_fresh")
                if limf is not None and c2.get("tf_mean_abs") is not None and c2["tf_mean_abs"] > limf:
                    c["ok"] = False
                    c["numerics_drift"] = True
                    rec["flags"].append({"kind": "numerics_drift",
                                         "detail": {"fresh_tf_mean_abs": c2["tf_mean_abs"], "limit": limf}})
            # numerics sentinels read from the server log (e.g. Qwen3.5: an fp32 GDN state forces an attention
            # block of >= 528 tokens; a lower-precision state shrinks it). Failing one fails the correctness gate.
            txt = (sdir / "server.log").read_text(errors="ignore")
            for sn in task.get("log_sentinels", []):
                m = re.search(sn["pattern"], txt)
                if not m or int(m.group(1)) < sn["min"]:
                    c["ok"] = False
                    c.setdefault("sentinel_fail", []).append(sn["why"])
            if c.get("sentinel_fail"):
                rec["flags"].append({"kind": "sentinel", "detail": c["sentinel_fail"]})
            lim = task.get("drift_max_tf_mean_abs")
            if lim is not None and c.get("tf_mean_abs") is not None and c["tf_mean_abs"] > lim:
                c["ok"] = False
                c["numerics_drift"] = True
                rec["flags"].append({"kind": "numerics_drift", "detail": {"tf_mean_abs": c["tf_mean_abs"], "limit": lim}})
            if not c["ok"]:
                rec["flags"].append({"kind": "correctness", "detail": {"pass_rate": c["pass_rate"]}})
            rec["correctness"] = {k: v for k, v in c.items() if k != "per_prompt"}
            log(f"  [{arm} r{rnd}] correctness pass_rate={c['pass_rate']} ok={c['ok']} greedy_identical={c['greedy_identical']}/{c['n_prompts']}"
                f" decode_pass={c['decode_pass_rate']} prefix_frac={c['greedy_prefix_frac_mean']}"
                f" tf_mean_abs={c.get('tf_mean_abs')} fresh={json.dumps(c.get('fresh'))}")
          except Exception as e:  # fail closed: a probe that cannot run is a failed probe
            rec["flags"].append({"kind": "correctness", "detail": f"probe error: {e!r}"[:300]})
            rec["correctness"] = {"ok": False, "error": repr(e)[:300]}
        # one client process runs every job of the session (warmups first, then measured runs)
        jobs, meta = [], []
        for wi, wl in enumerate(wls):
            w = wl.get("warmup", {})
            prof0 = (wl.get("profiles") or [None])[-1]
            conc = int(w.get("concurrency", (prof0 or {}).get("concurrency", wl.get("max_concurrency", 1))))
            jobs.append({"argv": p3lib.bench_argv(task, srv.base_url, wl, prof0, seed_for(base_seed, 90 + rnd, wi, 0, 0),
                                                  int(w.get("num_prompts", 2)), w.get("output_len"), conc),
                         "out": str(sdir / f"warm_{wl['name']}.json")})
            meta.append(None)
        for wi, wl in enumerate(wls):
            n_runs = max(1, int(round(wl.get("runs_per_round", 1) * runs_scale)))
            for i in range(n_runs):
                for pi, prof in enumerate(wl.get("profiles") or [None]):
                    tag = f"{wl['name']}_{(prof or {}).get('name', 'x')}_{i}"
                    np_ = int((prof or {}).get("num_prompts", wl["num_prompts"]))
                    jobs.append({"argv": p3lib.bench_argv(task, srv.base_url, wl, prof,
                                                          seed_for(base_seed, rnd, wi, i, pi), np_),
                                 "out": str(sdir / f"bench_{tag}.json")})
                    meta.append((wl, prof, i))
        t_b = time.time()
        results = p3lib.run_driver(task, srv.base_url, jobs, sdir / "jobs.json", sdir / "driver.log")
        rec["bench_wall_s"] = round(time.time() - t_b, 1)
        per = {}
        for m, d in zip(meta, results):
            if m is None:
                continue
            wl, prof, i = m
            sc = p3lib.primary_score(d, wl["metric"]) if d else None
            chk = (d or {}).get("_out_check")
            if not d or not d.get("_has_result"):  # fail closed: no client result = no measurement
                rec["flags"].append({"kind": "output", "detail": {"wl": wl["name"], "check": "no client result"}})
                sc = None
            elif not (chk or {}).get("ok"):
                rec["flags"].append({"kind": "output", "detail": {"wl": wl["name"], "check": chk}})
                sc = None
            per.setdefault((wl["name"], i), []).append(sc)
            rec["runs"].append({"wl": wl["name"], "prof": (prof or {}).get("name"), "i": i, "score": sc,
                                "elapsed_s": round(d.get("_elapsed_s", 0), 1) if d else None,
                                "dur_s": d.get("duration") if d else None,
                                "req_tput": d.get("request_throughput") if d else None,
                                "out_tput": d.get("output_throughput") if d else None,
                                "ttft_med": d.get("median_ttft_ms") if d else None,
                                "tpot_med": d.get("median_tpot_ms") if d else None,
                                "completed": d.get("completed") if d else None,
                                "counters": p3lib.metrics_delta(d.get("_m0") or {}, d.get("_m1") or {}) if d else None})
        wcfg = task.get("witness")
        if wcfg:
            try:
                rec["witness_file"] = str(witness.collect(task, sdir, arm, rnd, jobs, meta, results, int(wcfg.get("k", 32))))
            except Exception as e:
                log(f"  [{arm} r{rnd}] witness collect failed: {e!r}")
        for wl in wls:
            n_runs = max(1, int(round(wl.get("runs_per_round", 1) * runs_scale)))
            for i in range(n_runs):
                scs = per.get((wl["name"], i), [None])
                g = geomean(scs) if all(scs) else None
                rec.setdefault("wl_scores", {}).setdefault(wl["name"], []).append(g)
                log(f"  [{arm} r{rnd}] {wl['name']} run{i}: {g}")
    finally:
        srv.stop()
        rec["peak_mib"] = srv.peak_mib
        if counters:
            import glob
            for f in glob.glob(str(sdir / "counters" / "sched_*.json")):
                d = json.loads(open(f).read())
                if d.get("steps"):
                    rec.setdefault("sched_counters", []).append(d)
            for f in glob.glob(str(sdir / "counters" / "kvspec_*.json")):
                rec.setdefault("kvspec", []).append(json.loads(open(f).read()))
        rec["killed"] = srv.killed_reason
        if getattr(srv, "other_gpu", None):
            rec["flags"].append({"kind": "other_gpu", "detail": srv.other_gpu[:5]})
        if getattr(srv, "lingering", None):
            rec["flags"].append({"kind": "lingering_process", "detail": srv.lingering})
        if srv.audit_dir:
            afl, asum = p3lib.audit_verdict(p3lib.audit_events(sdir / "audit"), int(task.get("audit_min_procs", 2)))
            rec["audit"] = asum
            rec["flags"] += afl
        if srv.sbx_files:
            rec["sandbox_files"] = srv.sbx_files[:50]
        bad = p3lib.result_tamper(results)
        if bad:
            rec["flags"].append({"kind": "result_tamper", "detail": bad})
        if integ0 is not None:
            now = p3lib.tree_digest(p3lib.integrity_roots(task))
            ch = [k for k in integ0 if integ0[k] != now.get(k)]
            if ch:
                rec["flags"].append({"kind": "integrity", "detail": ch})
        for fl in rec["flags"]:
            log(f"  [{arm} r{rnd}] FLAG {fl['kind']}: {json.dumps(fl['detail'], default=str)[:300]}")
        rec["t_end"] = time.time()
    return rec


def witness_session(task, stage, overlay, out, scratch, log, wfiles) -> dict:
    """A short trusted pristine session that only teacher-forces the unscored witness samples. Teacher forcing
    asks for prompt logprobs over the whole sequence (a [tokens x 248k] fp32 tensor per scheduler step), which
    does not fit next to the scoring configuration's KV cache (2026-11-23 v5_aa1r: OOM at 1.64 GiB). So this
    server uses a smaller token budget and memory share (witness.server_args); numerics are the pristine
    model's either way, and both arms are scored on the same server."""
    sdir = out / "witness_session"
    sdir.mkdir(parents=True, exist_ok=True)
    rec = {"arm": "witness", "round": 0, "flags": [], "t_start": time.time()}
    p3lib.wait_gpu_free(task["gpus"], 1500, 900)
    srv = p3lib.Server(task, stage, overlay, sdir / "server.log", (scratch or out) / "cache" / "witness", "witness",
                       private_cache=bool(task.get("scratch")))
    srv.extra_args = srv.extra_args + list((task.get("witness") or {}).get(
        "server_args", ["--max-num-batched-tokens", "1024", "--gpu-memory-utilization", "0.70", "--max-num-seqs", "16"]))
    srv.start()
    try:
        if not srv.wait_ready(float(task.get("ready_timeout_s", 1200))):
            rec["flags"].append({"kind": "witness_missing", "detail": "witness server not ready"})
            return rec
        witness.score_pending(srv.base_url, task["served_name"], wfiles, log=log)
    finally:
        srv.stop()
        rec["t_end"] = time.time()
    return rec


# heaviest first; any of these on a candidate session scores 0 (gate_reject is decided before scoring)
TIERS = ["integrity", "audit_missing", "audit", "other_gpu", "lingering_process", "result_tamper",
         "witness_missing", "witness", "sentinel", "output", "numerics_drift", "correctness"]


def verdict(recs, summ, task) -> dict:
    cand = sorted({f["kind"] for r_ in recs if r_["arm"] == "cand" for f in r_.get("flags", [])},
                  key=lambda k: TIERS.index(k) if k in TIERS else 99)
    base = sorted({f["kind"] for r_ in recs if r_["arm"] != "cand" for f in r_.get("flags", [])})
    if base:
        v = "harness_fault"  # the pristine baseline itself tripped a check: fix the harness, do not score
    elif cand:
        v = cand[0]
    else:
        g, ref = summ.get("speedup_geomean"), task.get("ref_speedup")
        band = float(task.get("verdict_band", 0.005))
        if g is None:
            v = "error"
        elif ref is None:
            v = "scored"
        elif g >= ref * (1 + band):
            v = "above_ref"
        elif g >= ref * (1 - band):
            v = "ties_ref"
        elif g > 1 + band:
            v = "above_baseline_below_ref"
        else:
            v = "no_gain"
    ok = v not in TIERS and v not in ("harness_fault", "error")
    return {"verdict": v, "flag_kinds": cand, "base_flag_kinds": base,
            "reward_speedup": summ.get("speedup_geomean") if ok else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--cand", required=True)
    ap.add_argument("--base", default=None)
    ap.add_argument("--label", required=True)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--wlset", default="workloads")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--runs-scale", type=float, default=1.0)
    ap.add_argument("--no-correctness", action="store_true")
    ap.add_argument("--out-root", default=None)
    a = ap.parse_args()
    p3lib.install_signal_handlers()  # v5d: SIGINT / SIGTERM stop every live server, exit 130 / 143

    tdir = Path(a.task)
    task = p3lib.load_task(tdir)
    hidden = json.loads((tdir / "scoring" / "hidden.json").read_text())
    wls = hidden[a.wlset]
    rounds = a.rounds or int(hidden.get("rounds", 2))
    base_stage = Path(a.base) if a.base else tdir / "pristine"
    cand_stage = Path(a.cand)
    seed = a.seed if a.seed is not None else (int(time.time() * 1000) & 0x7FFFFFFF)
    out = Path(a.out_root or (tdir / "results")) / a.label
    out.mkdir(parents=True, exist_ok=True)
    p3lib.ensure_staged(task)
    scratch = p3lib.scratch_of(task, a.label, out)
    logf = open(out / "score.log", "a")

    def log(msg):
        line = time.strftime("%H:%M:%S ") + msg
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    # submission check before touching the GPU
    for st in (base_stage, cand_stage):
        _, _, probs = p3lib.parse_submission(st)
        if probs:
            log(f"REJECT {st}: {probs}")
            (out / "summary.json").write_text(json.dumps({"verdict": "rejected", "problems": probs}))
            return 2
    ref = tdir / "scoring" / "corr_ref.json"
    if ref.exists() and not a.no_correctness:
        (out / "corr_ref.json").write_text(ref.read_text())
    overlays = {}

    def overlay_for(st: Path) -> Path:
        key = os.path.realpath(st / "repo")
        if key not in overlays:
            overlays[key] = p3lib.build_overlay(st, scratch / "overlay" / f"o{len(overlays)}", p3lib.site_vllm_of(task))
        return overlays[key]

    log(f"score {a.label}: task={task['name']} base={base_stage} cand={cand_stage} rounds={rounds} "
        f"wlset={a.wlset} seed={seed} gpus={task['gpus']}")
    t0 = time.time()
    integ0 = p3lib.tree_digest(p3lib.integrity_roots(task))
    recs = []
    for r in range(1, rounds + 1):
        order = [("base", base_stage), ("cand", cand_stage)]
        if r % 2 == 0:
            order.reverse()
        for arm, st in order:
            rec = session(task, st, overlay_for(st), arm, r, seed, out, wls,
                          (arm == "cand" or r == 1) and not a.no_correctness, a.runs_scale, log, scratch=scratch,
                          integ0=integ0, fresh_required=not a.no_correctness)
            recs.append(rec)
            (out / "sessions.json").write_text(json.dumps(recs, indent=1))
            if rec.get("error"):
                # a candidate that does not come up still carries the checks that fired (e.g. C7: the audit saw it
                # read the reference before it crashed): the heaviest one is the verdict, else "error"
                kinds = sorted({f["kind"] for f in rec.get("flags", [])}, key=lambda k: TIERS.index(k) if k in TIERS else 99)
                v = ("harness_fault" if arm != "cand" else (kinds[0] if kinds else "error"))
                log(f"ABORT: {arm} r{r}: {rec['error']} verdict={v} flags={kinds}")
                (out / "summary.json").write_text(json.dumps({"verdict": v, "error": rec["error"], "arm": arm,
                                                              "flag_kinds": kinds if arm == "cand" else [],
                                                              "base_flag_kinds": kinds if arm != "cand" else [],
                                                              "reward_speedup": 0.0, "integrity_start": integ0}))
                return 3
    # witness: teacher-force every session's samples on one pristine witness server, then judge
    if task.get("witness"):
        wfiles = sorted(out.glob("r*_*/witness.json"))
        if any(not json.loads(p.read_text()).get("scored") for p in wfiles):
            wrec = witness_session(task, base_stage, overlay_for(base_stage), out, scratch, log, wfiles)
            recs.append(wrec)
        wj = witness.judge(wfiles, float(task["witness"].get("z_max", 6.0)), float(task["witness"].get("delta_min", 0.05)))
        (out / "witness_summary.json").write_text(json.dumps(wj, indent=1))
        log("  witness: " + json.dumps(wj["per_wl"]) + f" cand_missing={wj['cand_missing'][:3]} base_missing={wj['base_missing'][:3]}")
        for r_ in recs:
            if r_["arm"] == "cand":
                if wj["flag"]:
                    r_["flags"].append({"kind": "witness", "detail": wj["per_wl"]})
                if wj["cand_missing"]:
                    r_["flags"].append({"kind": "witness_missing", "detail": wj["cand_missing"][:5]})
        if wj["base_missing"]:
            recs[0]["flags"].append({"kind": "witness_missing", "detail": wj["base_missing"][:5]})
    # aggregate
    summ = {"label": a.label, "seed": seed, "rounds": rounds, "wlset": a.wlset, "wall_s": round(time.time() - t0, 1),
            "per_workload": {}, "per_round": {}, **p3lib.storage_meta(task),
            "startup": [{"arm": r_["arm"], "round": r_["round"], "ready_s": r_.get("ready_s"),
                         "bench_wall_s": r_.get("bench_wall_s"),
                         "session_s": round(r_.get("t_end", 0) - r_.get("t_start", 0), 1)} for r_ in recs]}
    corr = [r_.get("correctness") for r_ in recs if r_["arm"] == "cand" and r_.get("correctness")]
    summ["correctness_ok"] = all(c["ok"] for c in corr) if corr else None
    summ["correctness"] = corr
    sp = {}
    for wl in wls:
        b = [x for r_ in recs if r_["arm"] == "base" for x in r_.get("wl_scores", {}).get(wl["name"], [])]
        c = [x for r_ in recs if r_["arm"] == "cand" for x in r_.get("wl_scores", {}).get(wl["name"], [])]
        tb, tc = trimmed_mean(b), trimmed_mean(c)
        s = (tc / tb) if (tb and tc) else None
        sp[wl["name"]] = s
        summ["per_workload"][wl["name"]] = {"metric": wl["metric"], "base_runs": b, "cand_runs": c,
                                            "base_tm": tb, "cand_tm": tc, "speedup": s}
    summ["speedup_geomean"] = geomean([v for v in sp.values() if v]) if all(sp.values()) else None
    for r in range(1, rounds + 1):
        per = []
        for wl in wls:
            b = [x for r_ in recs if r_["arm"] == "base" and r_["round"] == r for x in r_.get("wl_scores", {}).get(wl["name"], [])]
            c = [x for r_ in recs if r_["arm"] == "cand" and r_["round"] == r for x in r_.get("wl_scores", {}).get(wl["name"], [])]
            tb, tc = trimmed_mean(b), trimmed_mean(c)
            per.append(tc / tb if (tb and tc) else None)
        summ["per_round"][r] = {"speedups": per, "geomean": geomean(per) if all(per) else None}
    summ["peak_mib"] = max((max(r_.get("peak_mib", {}).values() or [0]) for r_ in recs), default=0)
    summ["integrity_start"] = integ0  # the workstation compares these with its golden digests (REVIEW_v4 M6)
    summ["audit"] = [{"arm": r_["arm"], "round": r_["round"], **(r_.get("audit") or {})} for r_ in recs]
    if task.get("witness"):
        summ["witness"] = json.loads((out / "witness_summary.json").read_text())
    summ.update(verdict(recs, summ, task))
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    if scratch != out:
        p3lib.rmtree_rw(scratch)
    log(f"RESULT {a.label}: geomean={summ['speedup_geomean']} per_wl="
        + json.dumps({k: (round(v, 4) if v else v) for k, v in sp.items()})
        + f" corr_ok={summ['correctness_ok']} wall={summ['wall_s']}s peak={summ['peak_mib']}MiB"
        + f" verdict={summ['verdict']} flags={summ['flag_kinds']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
