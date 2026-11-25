"""Output witness (v5, after REVIEW_v4 S2 / M2): are the scored outputs what this model samples?

The logprob gate probes the server with its own requests (temperature 0, prompt_logprobs), which a server can
recognise and answer exactly while it fakes the scored requests (random tokens, skipped decode, a truncated
context, greedy instead of sampling). The witness checks the scored requests themselves:
  1 collect: after each session, K requests per measured run (evenly spread) with the generated text from the
    client's own result file (sha256 checked against what the client reported through its pipe);
  2 rebuild the prompts with the stock client code (witness_prompts.py: same seed, same get_samples); every
    rebuilt prompt length must equal the client's input_lens, else the sample is void;
  3 score: on one trusted pristine witness server at the end (scorer: witness_session), teacher-force
    prompt + generated text: mean log-probability of the generated tokens, and two calibration statistics
    (c1, c5: how often the token is the model's top-1 / in its top-5, minus how often it should be; _score_one);
  4 test, per workload and statistic, candidate samples against the baseline's own samples of the same run,
    two-sided: flag `witness` when |z| > z_max and |delta| > the statistic's minimum (mean_lp: delta_min nats
    per token; c1 / c5: 0.01).
    Fail closed: a candidate session without scored samples, or with void samples, is `witness_missing`.
v5d: a run with fewer requests than K contributes all its requests (v5c: only request 0). A workload whose output
length (from the scorer's own argv, never from server output) is below WITNESS_MIN_OUT_LEN (task.json
witness.min_out_len overrides) has nothing to witness: its sets are "skipped: ...", neither ok nor missing, and
judge() gives it a {"skipped": true, "flag": false} row (v5c voided them as unscorable empty text -> witness_missing
-> harness_fault on an honest run, e.g. a 1-token TTFT workload).
Sampling at temperature 1 makes both arms draw from the same distribution, so delta ~ 0 for any faithful
implementation (numerics noise is ~0.005 nats); random tokens give delta ~ -10, greedy ~ +1 or more.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import correctness as C
import p3lib


def _arg(argv: list[str], name: str, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


WITNESS_MIN_OUT_LEN = 4  # v5d: shorter outputs (e.g. a 1-token TTFT workload) carry no witness signal


def _skip_reason(task: dict, argv: list[str]) -> str | None:
    """v5d: "skipped: ..." when the requested output length is below the threshold. Both inputs are the scorer's:
    the hidden task config and the bench argv it built; an unknown length (no output-length flag) is never skipped."""
    mn = int((task.get("witness") or {}).get("min_out_len", WITNESS_MIN_OUT_LEN))
    ol = _arg(argv, "--random-output-len", _arg(argv, "--sharegpt-output-len"))
    if ol is not None and int(ol) < mn:
        return f"skipped: output_len < {mn} (nothing to witness)"
    return None


def _skipped(s: dict) -> bool:
    return str(s.get("status", "")).startswith("skipped")


def collect(task: dict, sdir: Path, arm: str, rnd: int, jobs: list[dict], meta: list, results: list,
            k: int = 32) -> Path:
    """Write <sdir>/witness.json (samples + rebuilt prompts, not yet scored). Returns its path."""
    sets = []
    for job, m, d in zip(jobs, meta, results):
        if m is None:
            continue
        wl, prof, i = m
        e = {"wl": wl["name"], "argv": job["argv"], "out_len": int(_arg(job["argv"], "--random-output-len", 0)),
             "status": "ok"}
        skip = _skip_reason(task, job["argv"])
        if skip:
            e["status"] = skip
            sets.append(e)
            continue
        try:
            blob = Path(job["out"]).read_text()
            if not d or hashlib.sha256(blob.encode()).hexdigest() != d.get("_sha256"):
                raise ValueError("result file missing or changed after the client reported it")
            r = json.loads(blob)
            texts, ilens = r.get("generated_texts"), r.get("input_lens")
            if not texts or not ilens or len(texts) != len(ilens):
                raise ValueError("no generated_texts / input_lens in the result")
            n = len(texts)
            m = min(k, n)  # v5d: n < k takes every request (v5c's j * n / k gave index 0 only); n >= k: unchanged
            idx = sorted({int(j * n / m) for j in range(m)})
            e.update(indices=idx, texts=[texts[j] for j in idx], input_lens=[ilens[j] for j in idx])
        except Exception as ex:
            e["status"] = f"void: {ex}"[:200]
        sets.append(e)
    todo = [s for s in sets if s["status"] == "ok"]
    if todo:
        spec = {"jobs": [{"argv": s["argv"], "indices": s["indices"]} for s in todo]}
        try:
            r = subprocess.run([str(p3lib.PY), "-I", str(Path(__file__).resolve().parent / "witness_prompts.py")],
                               input=json.dumps(spec), capture_output=True, text=True, timeout=900)
            got = json.loads(r.stdout)["jobs"]
            for s, g in zip(todo, got):
                if g["indices"] != s["indices"] or g["prompt_lens"] != s["input_lens"]:
                    s["status"] = "void: rebuilt prompts do not match the client's input_lens"
                else:
                    s["prompts"] = g["prompts"]
        except Exception as ex:
            for s in todo:
                s["status"] = f"void: prompt rebuild failed: {ex!r}"[:200]
    w = {"arm": arm, "round": rnd, "scored": False, "sets": sets}
    p = sdir / "witness.json"
    p.write_text(json.dumps(w))
    return p


TOPK = 5


def _score_one(base_url: str, model: str, prompt: str, text: str) -> dict:
    """Teacher-force prompt + generated text; per generated position take the token's log-probability and rank and
    the model's top-1 / top-K probability mass. Per sample:
      mean_lp : mean log-probability of the generated tokens;
      c1      : mean of 1[token is the model's top-1] - p_top1;
      cK      : mean of 1[token in the model's top-K] - (top-K mass).
    For tokens drawn from the model at temperature 1, c1 and cK have mean ~0 whatever the content, and their
    per-sample spread is small (an indicator minus its own probability), unlike mean_lp, whose spread is the
    spread of the content (~1.5-2 nats on random prompts). The text is re-tokenised, which shifts c1 / cK a
    little for both arms alike, so they are compared between arms like mean_lp."""
    ids = C._tokenize(base_url, model, prompt)
    gids = C._post(base_url, "/tokenize", {"model": model, "prompt": text, "add_special_tokens": False})["tokens"]
    if not gids:
        return {"n": 0, "mean_lp": None}
    full = ids + gids
    d = C._post(base_url, "/v1/completions", {"model": model, "prompt": full, "max_tokens": 1, "temperature": 0.0,
                                                "prompt_logprobs": TOPK, "return_tokens_as_token_ids": True})
    plp = d["choices"][0]["prompt_logprobs"]
    lp, c1, ck = [], [], []
    for pos in range(len(ids), len(full)):
        entry = plp[pos] or {}
        tok = str(full[pos])
        v = entry.get(tok) or entry.get(f"token_id:{tok}")
        if not isinstance(v, dict):
            continue
        lp.append(float(v["logprob"]))
        es = [e for e in entry.values() if isinstance(e, dict) and "logprob" in e]
        if all(e.get("rank") is not None for e in es):
            tops = sorted((float(e["logprob"]) for e in es if int(e["rank"]) <= TOPK), reverse=True)
            r = int(v["rank"])
        else:  # no ranks in the response: the dict is the top-K plus the token itself
            tops = sorted((float(e["logprob"]) for e in es), reverse=True)[:TOPK]
            r = 1 + sum(1 for x in tops if x > float(v["logprob"]))
        if len(tops) < TOPK:
            continue
        c1.append((1.0 if r == 1 else 0.0) - math.exp(tops[0]))
        ck.append((1.0 if r <= TOPK else 0.0) - sum(math.exp(x) for x in tops[:TOPK]))
    if not lp:
        return {"n": len(gids), "n_lp": 0, "mean_lp": None}
    return {"n": len(gids), "n_lp": len(lp), "n_c": len(c1), "mean_lp": sum(lp) / len(lp),
            "c1": (sum(c1) / len(c1)) if c1 else None, f"c{TOPK}": (sum(ck) / len(ck)) if ck else None}


def score_pending(base_url: str, model: str, paths: list[Path], log=print) -> None:
    """Teacher-force every unscored witness.json on this (trusted, pristine) server."""
    for p in paths:
        w = json.loads(p.read_text())
        if w.get("scored"):
            continue
        jobs = [(s, i) for s in w["sets"] if s["status"] == "ok" for i in range(len(s["prompts"]))]
        with ThreadPoolExecutor(4) as ex:
            res = list(ex.map(lambda si: _safe(base_url, model, si[0]["prompts"][si[1]], si[0]["texts"][si[1]]), jobs))
        for s in w["sets"]:
            s["scores"] = []
        for (s, i), r in zip(jobs, res):
            s["scores"].append(r)
        for s in w["sets"]:
            if s["status"] == "ok" and any(r.get("mean_lp") is None for r in s["scores"]):
                s["status"] = "void: unscorable sample (empty text or scoring error)"
            s.pop("prompts", None)  # keep the file small; prompts can be rebuilt from argv
        w["scored"] = True
        p.write_text(json.dumps(w))
        log(f"  witness scored {p.parent.name}: " + json.dumps(
            {s["wl"]: (round(_mean([r["mean_lp"] for r in s["scores"]]), 4) if s["status"] == "ok" else s["status"])
             for s in w["sets"]}))


def _safe(base_url, model, prompt, text):
    err = None
    for attempt in range(2):
        try:
            return _score_one(base_url, model, prompt, text)
        except Exception as ex:
            err = ex
            time.sleep(2)
    return {"n": None, "mean_lp": None, "error": repr(err)[:120]}


def _mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else float("nan")


def _var(v):
    m = _mean(v)
    return sum((x - m) ** 2 for x in v) / (len(v) - 1) if len(v) > 1 else float("nan")


# statistic -> minimum |delta| for a flag (besides |z| > z_max); older witness files carry mean_lp only
STATS = {"mean_lp": None, "c1": 0.01, f"c{TOPK}": 0.01}


def judge(paths: list[Path], z_max: float = 6.0, delta_min: float = 0.05) -> dict:
    """Per workload and statistic: pooled per-sample values of the candidate vs the baseline sessions, two-sided
    z-test. A flag needs |z| > z_max and |delta| > the statistic's minimum (delta_min for mean_lp)."""
    ws = [json.loads(p.read_text()) for p in paths]
    out = {"per_wl": {}, "cand_missing": [], "base_missing": [], "flag": False}
    for w in ws:
        tag = f"{w['arm']}_r{w['round']}"
        if not w.get("scored"):
            (out["cand_missing"] if w["arm"] == "cand" else out["base_missing"]).append(f"{tag}: not scored")
            continue
        for s in w["sets"]:
            if s["status"] != "ok" and not _skipped(s):  # v5d: skipped = nothing to witness, not missing
                (out["cand_missing"] if w["arm"] == "cand" else out["base_missing"]).append(f"{tag} {s['wl']}: {s['status']}")
    wls = sorted({s["wl"] for w in ws for s in w["sets"]})
    rnd = lambda x: round(x, 5) if isinstance(x, float) and math.isfinite(x) else x  # noqa: E731
    for wl in wls:
        if all(_skipped(s) for w in ws for s in w["sets"] if s["wl"] == wl):
            out["per_wl"][wl] = {"skipped": True, "flag": False}  # v5d
            continue
        row = {}
        for stat, dmin in STATS.items():
            dmin = delta_min if dmin is None else dmin
            v = {"base": [], "cand": []}
            for w in ws:
                for s in w["sets"]:
                    if s["wl"] == wl and s["status"] == "ok" and w.get("scored"):
                        v["cand" if w["arm"] == "cand" else "base"].extend(
                            r[stat] for r in s["scores"] if r.get(stat) is not None)
            b, c = v["base"], v["cand"]
            if stat != "mean_lp" and not (b or c):
                continue  # an older witness file without this statistic
            st = {"n_base": len(b), "n_cand": len(c), "mean_base": _mean(b), "mean_cand": _mean(c)}
            if len(b) > 1 and len(c) > 1:
                d = st["mean_cand"] - st["mean_base"]
                se = math.sqrt(_var(b) / len(b) + _var(c) / len(c))
                st.update(delta=d, se=se, z=(d / se) if se > 0 else float("inf") * (1 if d else 0))
                st["flag"] = abs(st["z"]) > z_max and abs(d) > dmin
                out["flag"] |= st["flag"]
            elif c and not b and stat == "mean_lp":
                out["base_missing"].append(f"{wl}: no baseline samples")
            if stat == "mean_lp":  # the v5 field names stay at the top level of the row
                row.update({k: rnd(x) for k, x in st.items()})
            else:
                row[stat] = {k: rnd(x) for k, x in st.items() if k in ("delta", "se", "z", "flag", "mean_base")}
        row["flag"] = bool(row.get("flag")) or any(isinstance(x, dict) and x.get("flag") for x in row.values())
        out["per_wl"][wl] = row
    return out
