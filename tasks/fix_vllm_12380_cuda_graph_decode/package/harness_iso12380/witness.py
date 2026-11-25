"""witness.py : trusted out-of-process check of in-process timing (review F1). p2_job.sh samples the GPU every 100 ms
from OUTSIDE the job (nvidia-smi: timestamp, memory.used, utilization.gpu -> mem.log). For offline tasks whose timed
region runs inside the candidate's process, the GPU-busy window of the generate() passes must match the time the
process reports for them.
busy window = the longest busy segment while memory is at its plateau: samples with util > 0 and memory >= 90 % of the
peak (KV cache allocated), consecutive busy samples at most GAP_S apart. This drops the profile run, the idle gap before
the passes and the teardown blips (a first version that took first..last busy sample had 3.4-4.5 s residuals, smoke
2026-11-23). Both passes run back to back, so they form one segment. residual = window - reported. The tolerance is
calibrated on parent and fix runs (trusted code) and stored in calib.json; a scored run whose |residual - calib
median| exceeds it is voided. Reporting less time than the GPU was busy (forging a faster pass) raises the residual."""
import datetime as dt, os
GAP_S = 1.0
MIN_SEG_S = 1.0


def _samples(out):
    f = os.path.join(out, "mem.log")
    if not os.path.exists(f):
        return []
    S = []
    for l in open(f, errors="replace"):
        p = [x.strip() for x in l.split(",")]
        if len(p) < 3:
            continue
        try:
            t = dt.datetime.strptime(p[0][:23], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            S.append((t, float(p[1]), float(p[2])))
        except Exception:
            continue
    return S


def busy_window(out):
    S = _samples(out)
    if len(S) < 5:
        return None
    peak = max(m for _, m, _ in S)
    busy = [t for t, m, u in S if u > 0 and m >= 0.9 * peak]
    if not busy:
        return None
    segs, a, b = [], busy[0], busy[0]
    for t in busy[1:]:
        if t - b > GAP_S:
            segs.append((a, b)); a = t
        b = t
    segs.append((a, b))
    # iso12380 copy (2026-11-24): the LAST segment of >= MIN_SEG_S, not the longest. For #12380 the fix's timed reps (14.5 s)
    # can be shorter than the start-up segment (CUDA-graph capture of 35 sizes + 3 warm-ups, ~20 s): in 1 of 6 fix runs of
    # the calibration the longest-segment rule picked start-up and gave a 6.2 s residual, which tripled the tolerance.
    # The timed reps are the last GPU work of the run (outputs are written after them); teardown blips are < 1 s.
    longs = [s for s in segs if s[1] - s[0] >= MIN_SEG_S]
    s0, s1 = longs[-1] if longs else max(segs, key=lambda x: x[1] - x[0])
    dts = [y[0] - x[0] for x, y in zip(S, S[1:])]
    return {"start": s0, "end": s1, "window_s": s1 - s0, "segments": len(segs), "samples": len(S),
            "sample_dt_s": sorted(dts)[len(dts) // 2] if dts else None}


def witness(out, reported_s):
    w = busy_window(out)
    if w is None or reported_s is None:
        return {"ok": None, "reason": "no samples or no reported time"}
    w["reported_s"] = reported_s; w["residual_s"] = w["window_s"] - reported_s
    return w
