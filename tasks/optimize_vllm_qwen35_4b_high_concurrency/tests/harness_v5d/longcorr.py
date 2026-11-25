"""Long-decode numerics probe: does a candidate drift over many decode steps (what the 64-token
teacher-forced gate cannot see, e.g. a recurrent state stored at lower precision)?

  python -I longcorr.py --task TASKDIR --stages a,b --label L [--n-tok 1024] [--n-prompts 8]

For each stage S (one server session each): greedy-decode n_tok tokens per prompt and keep S's own
decode-time logprobs. Then one pristine session: greedy-decode as a control, and teacher-force every
stage's token sequences (prefill path, the exact reference). Report, per position bucket, the share of
positions with |lp_S_decode - lp_pristine_TF| <= 0.05 + 0.05*|lp_ref| and the mean |diff|.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import correctness as C  # noqa: E402
import p3lib  # noqa: E402

BUCKETS = [(0, 64), (64, 256), (256, 512), (512, 1024), (1024, 4096)]


def boot(task, stage, overlay, out, arm, scratch):
    p3lib.wait_gpu_free(task["gpus"], 1500, 600)
    srv = p3lib.Server(task, stage, overlay, out / f"{arm}.server.log", scratch / "cache" / arm, arm)
    srv.start()
    if not srv.wait_ready(task.get("ready_timeout_s", 1200)):
        srv.stop()
        raise RuntimeError(f"{arm}: not ready (killed={srv.killed_reason})")
    return srv


def gen_all(srv, model, prompts_ids, n):
    with ThreadPoolExecutor(len(prompts_ids)) as ex:
        return list(ex.map(lambda ids: C._greedy(srv.base_url, model, ids, n, with_lp=True), prompts_ids))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--stages", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--n-tok", type=int, default=1024)
    ap.add_argument("--n-prompts", type=int, default=8)
    a = ap.parse_args()
    p3lib.install_signal_handlers()  # v5d: SIGINT / SIGTERM stop every live server, exit 130 / 143
    tdir = Path(a.task)
    task = p3lib.load_task(tdir)
    p3lib.ensure_staged(task)
    out = tdir / "results" / a.label
    out.mkdir(parents=True, exist_ok=True)
    scratch = p3lib.scratch_of(task, a.label, out)
    model = task["served_name"]
    texts = json.loads((tdir / "scoring" / "corr_prompts.json").read_text())[: a.n_prompts]
    gens = {}
    ids = None
    names = [s for s in a.stages.split(",") if s]
    for k, nm in enumerate(names + ["pristine"]):
        st = tdir / "pristine" if nm == "pristine" else tdir / "sweep" / nm
        ov = p3lib.build_overlay(st, scratch / "overlay" / f"o{k}", p3lib.site_vllm_of(task))
        srv = boot(task, st, ov, out, f"{k:02d}_{nm}", scratch)
        try:
            if ids is None:
                ids = [C._tokenize(srv.base_url, model, t) for t in texts]
            t0 = time.time()
            gens[nm] = gen_all(srv, model, ids, a.n_tok)
            print(f"{nm}: generated in {time.time() - t0:.0f}s", flush=True)
            if nm == "pristine":  # reference session: teacher-force every stage's own sequences
                ref = {}
                for s, g in gens.items():
                    with ThreadPoolExecutor(len(ids)) as ex:
                        ref[s] = list(ex.map(lambda p: C._teacher_forced(srv.base_url, model, p[0] + p[1][0], len(p[0])),
                                             zip(ids, g)))
        finally:
            srv.stop()
    rep = {"label": a.label, "n_tok": a.n_tok, "n_prompts": len(ids), "stages": {}}
    for s, g in gens.items():
        rows = {}
        for lo, hi in BUCKETS:
            n = ok = 0
            tot = 0.0
            for (toks, dlp), tlp in zip(g, ref[s]):
                for i in range(lo, min(hi, len(dlp), len(tlp))):
                    x, r = dlp[i], tlp[i]
                    if x != x or r != r:
                        continue
                    d = abs(x - r)
                    n += 1
                    tot += d
                    ok += d <= C.ATOL + C.RTOL * abs(r)
            if n:
                rows[f"{lo}-{hi}"] = {"n": n, "pass": round(ok / n, 4), "mean_abs": round(tot / n, 4)}
        rep["stages"][s] = rows
        print(s, json.dumps(rows), flush=True)
    (out / "longcorr.json").write_text(json.dumps(rep, indent=1))
    if scratch != out:
        p3lib.rmtree_rw(scratch)
    print("LONGCORR_DONE", flush=True)


if __name__ == "__main__":
    main()
