"""Run a list of `vllm bench serve` jobs in ONE client process (imports vLLM once). Trusted side: the
pristine venv, no overlay, never imports candidate code.

  python -I bench_driver.py  < JOBS.json     (spec on stdin, so a candidate cannot edit it on disk)

JOBS = {"base_url":..., "jobs":[{"argv":[...bench serve args...], "out": path}, ...]}.
Each job: scrape /metrics, run vllm.benchmarks.serve.main(args) with --save-detailed, scrape again, check the
scored output itself, and print ONE line `[driver-result] {json}` to stdout with the key metrics, the output
check and the sha256 of the result file as finally written. The scorer reads its numbers from these lines (a
pipe it owns), never from the files, and flags any later change of a file (result_tamper).

Output check (scored output, not timing): every request completed without error; every output length equals
the requested one (range_ratio 0) or lies in the dataset's range; no empty text; re-tokenized text length /
reported length has a median in [0.75, 1.33] (catches a server that reports tokens it never streamed).
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3lib  # noqa: E402  (stdlib-only helpers)
from vllm.benchmarks.serve import add_cli_args, main as bench_main  # noqa: E402

spec = json.loads(sys.stdin.read())
parser = argparse.ArgumentParser()
add_cli_args(parser)
_tok = {}


def tokenizer(path):
    if path not in _tok:
        from transformers import AutoTokenizer
        _tok[path] = AutoTokenizer.from_pretrained(path)
    return _tok[path]


def out_check(d, args):
    ol, rr = int(args.random_output_len), float(args.random_range_ratio)
    lo, hi = (ol, ol) if rr == 0 else (int(ol * (1 - rr)), int(ol * (1 + rr)) + 1)
    lens = d.get("output_lens") or []
    texts = d.get("generated_texts") or []
    errs = [e for e in (d.get("errors") or []) if e]
    n = int(args.num_prompts)
    bad_len = sum(1 for x in lens if not (lo <= int(x or 0) <= hi))
    empty = sum(1 for t, x in zip(texts, lens) if (x or 0) >= 4 and not (t or "").strip())
    ratios = []
    if texts:
        tk = tokenizer(args.tokenizer)
        enc = tk(list(texts), add_special_tokens=False)["input_ids"]
        ratios = [len(e) / x for e, x in zip(enc, lens) if x]
    med = statistics.median(ratios) if ratios else None
    # v5d: the upper bound needs many requests. Long temperature-1 outputs of honest servers sometimes decode into
    # byte-fallback text that re-tokenizes into up to 2.7x the tokens (seen on task B5's decode_heavy, 4 requests of
    # 2048 tokens: 2 of 4 requests high -> median 1.405, a false "output" fail on main, 11-24). Text SHORTER than the
    # counted tokens (ratio < 0.75) is the cheating signal and stays checked at any n; above 1.33 is enforced only
    # for runs with >= 16 requests, else up to 3.0. Task C's scored runs have >= 128 requests and its warm-ups >= 16,
    # so task C is unchanged.
    rt_hi = 1.33 if n >= 16 else 3.0
    ok = (d.get("completed") == n and len(lens) == n and len(texts) == n and not errs and bad_len == 0
          and empty == 0 and med is not None and 0.75 <= med <= rt_hi)
    return {"ok": ok, "n": n, "completed": d.get("completed"), "errors": len(errs), "bad_len": bad_len,
            "len_range": [lo, hi], "len_min": min(lens) if lens else None, "len_max": max(lens) if lens else None,
            "empty_text": empty, "retok_ratio_median": round(med, 4) if med is not None else None,
            "retok_ratio_min": round(min(ratios), 4) if ratios else None, "retok_bounds": [0.75, rt_hi]}


KEYS = ("completed", "failed", "duration", "request_throughput", "output_throughput", "total_token_throughput",
        "total_input_tokens", "total_output_tokens", "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "median_tpot_ms", "p99_tpot_ms", "mean_itl_ms", "median_itl_ms", "mean_e2el_ms",
        "median_e2el_ms")
for j in spec["jobs"]:
    out = Path(j["out"])
    m0 = p3lib.scrape_metrics(spec["base_url"])
    t0 = time.time()
    err = None
    args = None
    try:
        args = parser.parse_args(j["argv"] + ["--save-result", "--save-detailed", "--result-filename", str(out)])
        bench_main(args)
    except BaseException as e:  # keep going; the scorer treats a missing result as a failed run
        err = repr(e)
    el = time.time() - t0
    m1 = p3lib.scrape_metrics(spec["base_url"])
    try:
        d = json.loads(out.read_text()) if out.exists() else {"error": err or "no result"}
    except Exception as e:
        d = {"error": f"unreadable result: {e!r}"}
    chk = None
    if args is not None and "completed" in d:
        try:
            chk = out_check(d, args)
        except Exception as e:
            chk = {"ok": False, "error": repr(e)[:200]}
    d.update({"_elapsed_s": el, "_m0": m0, "_m1": m1, "_err": err, "_out_check": chk})
    blob = json.dumps(d)
    out.write_text(blob)
    res = {k: d.get(k) for k in KEYS if k in d}
    res.update({"out": str(out), "_elapsed_s": el, "_err": err, "_out_check": chk, "_m0": m0, "_m1": m1,
                "_sha256": hashlib.sha256(blob.encode()).hexdigest(), "_has_result": "completed" in d})
    print("[driver-result] " + json.dumps(res), flush=True)
    print(f"[driver] {out.name} {el:.1f}s err={err} out_ok={(chk or {}).get('ok')}", flush=True)
print("[driver] done", flush=True)
