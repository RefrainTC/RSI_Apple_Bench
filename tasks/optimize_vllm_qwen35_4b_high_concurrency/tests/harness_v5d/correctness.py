"""Teacher-forced logprob correctness gate over HTTP (adapted from Apple's run_correctness.py).

make-ref : on the pristine server, greedy-generate N tokens per prompt, then teacher-force
           prompt+generation and store the per-position logprob of each generated token.
check    : on the candidate server, teacher-force the same token sequences and require
           |lp_cand - lp_ref| <= atol + rtol*|lp_ref| on >= min_pass_rate of positions.
Also reports greedy agreement (how many prompts regenerate the same tokens), info only.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

ATOL, RTOL, MIN_PASS = 0.05, 0.05, 0.95


def _post(base: str, path: str, body: dict, timeout: float = 300) -> dict:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _tokenize(base: str, model: str, text: str) -> list[int]:
    return _post(base, "/tokenize", {"model": model, "prompt": text, "add_special_tokens": True})["tokens"]


def _greedy(base: str, model: str, ids: list[int], n: int, with_lp: bool = False):
    d = _post(base, "/v1/completions", {"model": model, "prompt": ids, "max_tokens": n, "temperature": 0.0,
                                        "logprobs": 1, "return_tokens_as_token_ids": True,
                                        "ignore_eos": True, "seed": 0})
    lp = d["choices"][0]["logprobs"]
    toks = [int(t.split(":", 1)[1]) for t in lp["tokens"]]
    return (toks, [float(x) if x is not None else float("nan") for x in lp["token_logprobs"]]) if with_lp else toks


def _teacher_forced(base: str, model: str, full: list[int], n_prompt: int) -> list[float]:
    d = _post(base, "/v1/completions", {"model": model, "prompt": full, "max_tokens": 1, "temperature": 0.0,
                                        "prompt_logprobs": 0, "return_tokens_as_token_ids": True})
    plp = d["choices"][0]["prompt_logprobs"]
    out = []
    for pos in range(n_prompt, len(full)):
        entry = plp[pos] or {}
        tok = str(full[pos])
        v = entry.get(tok)
        if v is None:  # key format can be "token_id:123" in some versions
            v = entry.get(f"token_id:{tok}")
        out.append(float(v["logprob"]) if isinstance(v, dict) else float("nan"))
    return out


def make_ref(base: str, model: str, prompts: list[str], n_tokens: int) -> dict:
    items = []
    for p in prompts:
        ids = _tokenize(base, model, p)
        gen = _greedy(base, model, ids, n_tokens)
        lps = _teacher_forced(base, model, ids + gen, len(ids))
        items.append({"prompt_ids": ids, "gen_ids": gen, "ref_lp": lps})
    return {"atol": ATOL, "rtol": RTOL, "min_pass_rate": MIN_PASS, "n_tokens": n_tokens, "items": items}


def make_ref_ids(base: str, model: str, prompts_ids: list[list[int]], n_tokens: int) -> dict:
    """Like make_ref, for token-id prompts (the scorer's per-run fresh set: random ids, like the workload)."""
    items = []
    for ids in prompts_ids:
        gen = _greedy(base, model, ids, n_tokens)
        lps = _teacher_forced(base, model, ids + gen, len(ids))
        items.append({"prompt_ids": ids, "gen_ids": gen, "ref_lp": lps})
    return {"atol": ATOL, "rtol": RTOL, "min_pass_rate": MIN_PASS, "n_tokens": n_tokens, "items": items}


def check(base: str, model: str, ref: dict) -> dict:
    n_pass = n_tot = 0
    greedy_same = 0
    dec_pass = dec_tot = 0
    prefix_fracs = []
    max_drift = 0.0
    tf_abs = 0.0
    tf_n = 0
    dec_abs = 0.0
    per = []
    for it in ref["items"]:
        ids, gen, rl = it["prompt_ids"], it["gen_ids"], it["ref_lp"]
        cl = _teacher_forced(base, model, ids + gen, len(ids))
        p = 0
        for a, b in zip(cl, rl):
            ok = (a == a) and abs(a - b) <= ref["atol"] + ref["rtol"] * abs(b)
            p += ok
            if a == a:
                max_drift = max(max_drift, abs(a - b))
                tf_abs += abs(a - b)
                tf_n += 1
        n_pass += p
        n_tot += len(rl)
        g, glp = _greedy(base, model, ids, len(gen), with_lp=True)
        k = 0
        while k < min(len(g), len(gen)) and g[k] == gen[k]:
            k += 1
        greedy_same += (k == len(gen))
        # decode-path check (what teacher forcing cannot see: KV/state written and re-read across decode
        # steps): the candidate's own decode-time logprobs vs the reference, on the matching greedy prefix
        dp = sum(1 for a, b in zip(glp[:k], rl[:k]) if a == a and abs(a - b) <= ref["atol"] + ref["rtol"] * abs(b))
        dec_abs += sum(abs(a - b) for a, b in zip(glp[:k], rl[:k]) if a == a)
        dec_pass += dp
        dec_tot += k
        prefix_fracs.append(k / max(len(gen), 1))
        per.append({"pass": p, "n": len(rl), "greedy_prefix": k, "decode_pass": dp})
    rate = n_pass / max(n_tot, 1)
    return {"pass_rate": round(rate, 5), "ok": rate >= ref["min_pass_rate"], "max_drift": round(max_drift, 5),
            "greedy_identical": greedy_same, "n_prompts": len(ref["items"]),
            "decode_pass_rate": round(dec_pass / max(dec_tot, 1), 5),
            "tf_mean_abs": round(tf_abs / max(tf_n, 1), 6), "dec_mean_abs": round(dec_abs / max(dec_tot, 1), 6),
            "greedy_prefix_frac_mean": round(sum(prefix_fracs) / max(len(prefix_fracs), 1), 4), "per_prompt": per}


if __name__ == "__main__":
    mode, base, model, path = sys.argv[1:5]
    if mode == "make-ref":
        prompts = json.loads(Path(sys.argv[5]).read_text())
        n = int(sys.argv[6]) if len(sys.argv) > 6 else 64
        Path(path).write_text(json.dumps(make_ref(base, model, prompts, n)))
        print("ref written", path)
    else:
        r = check(base, model, json.loads(Path(path).read_text()))
        print(json.dumps({k: v for k, v in r.items() if k != "per_prompt"}))
