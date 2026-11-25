"""offline_c5y.py <out dir> : workload, correctness passes and teacher-forced check of the vLLM #27418 task (trusted, ours).
The timed part is the batch-5 screen's bench/offline_b5.py image mode with SEQ=1 (load w2), unchanged: same images, text,
seeds, warm-up, digests and metric.

Timed mode (default): LLM(model=M, **LLM_KW). Rep k uses image seed 1 + k: N requests, each one synthetic IMG x IMG image
  (seeded per rep and request) + a fixed text, each request in its own generate() call (SEQ=1) or all in one call
  (SEQ=0), OL greedy tokens (ignore_eos). Untimed warm-up before (image seed 1000, at most 8 requests in one call, 8
  tokens). Metric: median rep wall time. Writes latency.json (per-rep output token counts, prompt token counts after the
  image placeholder is expanded, sha256 of the output ids), settings.json (resolved engine configuration) and
  seqs.json (image seed, request index and output ids of every request).
CORRECT=1: text pass. 24 fixed text prompts (no image), one generate() call each, 64 greedy tokens with the produced
  token's logprob -> correct.json (compared exactly with the parent's pass).
IMGPASS=1: image pass. 8 images with plain content (a coloured shape on a coloured background, IMG x IMG), one question
  each, one generate() call each, 48 greedy tokens (EOS may stop) -> imgpass.json (ids; re-read by the parent in TF).
TF=1 (PARENT venv only): teacher-forced re-read. For every run dir in TF_DIRS (":"-separated) it rebuilds each request
  (image from its seed, unexpanded prompt ids from the tokenizer) from seqs.json / imgpass.json, appends the produced
  output ids and runs one prefill with prompt_logprobs=1, one request per call; for every output token gap = top-1
  logprob - logprob of the produced token (0 = argmax) -> <out>/tf.json.
Env: M N IMG SEQ OL REPS MEMU LLM_KW CORRECT IMGPASS TF TF_DIRS."""
import hashlib, json, os, random, statistics, sys, time

WORDS = ("the of and to in is was for on that with as by at from his her it an were are which this be or has had "
         "river city music garden winter science market history paper light").split()
SYS = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
TXT = SYS + "Describe this image in detail.<|im_end|>\n<|im_start|>assistant\n"   # timed load (as the screen)
COLORS = [("red", (220, 30, 30)), ("green", (30, 170, 60)), ("blue", (40, 60, 220)), ("yellow", (240, 220, 40)),
          ("black", (10, 10, 10)), ("white", (245, 245, 245)), ("purple", (130, 40, 160)), ("orange", (250, 140, 20))]
QUESTIONS = ["What shape is in the middle of this image, and what colour is it?",
             "What is the colour of the background, and what object is drawn on it?",
             "Describe the shape and the colours you see.", "What colour is the shape, and what colour is the background?"]


def timed_image(IMG, seed, i):
    """the screen's synthetic image i of the prompt set with this seed (offline_b5.py, unchanged)"""
    import numpy as np
    from PIL import Image
    yy, xx = np.mgrid[0:IMG, 0:IMG]; nr = np.random.default_rng(seed)
    for j in range(i + 1):   # the screen draws the noise of images 0..i in order from one generator
        base = np.stack([(xx * (j % 7 + 1)) % 256, (yy * (j % 5 + 1)) % 256, ((xx + yy) * (j % 3 + 1)) % 256], -1)
        noise = nr.integers(0, 64, base.shape)
    return Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8))


def timed_set(IMG, N, seed):
    import numpy as np
    from PIL import Image
    yy, xx = np.mgrid[0:IMG, 0:IMG]; nr = np.random.default_rng(seed); ps = []
    for i in range(N):
        base = np.stack([(xx * (i % 7 + 1)) % 256, (yy * (i % 5 + 1)) % 256, ((xx + yy) * (i % 3 + 1)) % 256], -1)
        im = Image.fromarray(np.clip(base + nr.integers(0, 64, base.shape), 0, 255).astype(np.uint8))
        ps.append({"prompt": TXT, "multi_modal_data": {"image": im}})
    return ps


def pass_image(IMG, k):
    """image pass item k: a filled shape (circle, square or triangle) of one colour on a background of another"""
    import numpy as np
    from PIL import Image
    bg = COLORS[k % 8]; fg = COLORS[(3 * k + 2) % 8]; shape = ("circle", "square", "triangle")[k % 3]
    yy, xx = np.mgrid[0:IMG, 0:IMG]; c = IMG / 2; r = IMG * 0.3
    if shape == "circle": m = (xx - c) ** 2 + (yy - c) ** 2 <= r * r
    elif shape == "square": m = (abs(xx - c) <= r) & (abs(yy - c) <= r)
    else: m = (yy >= c - r) & (yy <= c + r) & (abs(xx - c) <= (yy - (c - r)) / 2)
    a = np.empty((IMG, IMG, 3), np.uint8); a[:] = bg[1]; a[m] = fg[1]
    q = SYS + QUESTIONS[k % 4] + "<|im_end|>\n<|im_start|>assistant\n"
    return q, Image.fromarray(a)


def settings(llm):
    """The resolved engine configuration (simple-valued fields of each sub-config of vllm_config)."""
    import dataclasses, enum
    out = {}
    try:
        vc = llm.llm_engine.vllm_config
        for name in ("model_config", "cache_config", "scheduler_config", "compilation_config", "parallel_config",
                     "load_config", "device_config", "speculative_config", "kv_transfer_config", "observability_config"):
            sub = getattr(vc, name, None)
            if sub is None: out[name] = None; continue
            fields = [f.name for f in dataclasses.fields(sub)] if dataclasses.is_dataclass(sub) else list(vars(sub))
            for f in fields:
                v = getattr(sub, f, None)
                if isinstance(v, enum.Enum): v = str(v)
                if isinstance(v, (list, tuple)) and all(isinstance(x, (str, int, float, bool, type(None))) for x in v): v = list(v)
                if isinstance(v, (str, int, float, bool, type(None), list)): out[f"{name}.{f}"] = v
                elif isinstance(v, dict): out[f"{name}.{f}"] = json.dumps(v, sort_keys=True, default=str)
    except Exception as e:
        out = {"error": repr(e)}
    return out


def main():
    from vllm import LLM, SamplingParams
    out = sys.argv[1]; E = os.environ.get
    M = E("M"); N = int(E("N", "4")); IMG = int(E("IMG", "1344")); SEQ = E("SEQ", "1") == "1"; OL = int(E("OL", "8"))
    REPS = int(E("REPS", "5"))
    kw = {"gpu_memory_utilization": float(E("MEMU", "0.80")), "seed": 0}
    if E("LLM_KW"): kw.update(json.loads(E("LLM_KW")))
    if E("TF", "0") == "1":
        kw["enable_prefix_caching"] = False
        tf_check(LLM(model=M, **kw), SamplingParams, out, [d for d in E("TF_DIRS", "").split(":") if d], IMG); return
    t0 = time.perf_counter(); llm = LLM(model=M, **kw); startup = time.perf_counter() - t0
    json.dump(settings(llm), open(os.path.join(out, "settings.json"), "w"), sort_keys=True, default=str)
    if E("CORRECT", "0") == "1":
        r = random.Random(7)
        texts = [" ".join(r.choice(WORDS) for _ in range(20 + 7 * i)) for i in range(24)]
        sp = SamplingParams(temperature=0, max_tokens=64, logprobs=0)
        outs = []
        for t in texts:   # one prompt per call
            o = llm.generate([t], sp, use_tqdm=False)[0].outputs[0]
            lp = [d[tok].logprob if d and tok in d else None for tok, d in zip(o.token_ids, o.logprobs or [])]
            outs.append({"ids": list(o.token_ids), "lp": lp, "text": o.text, "finish": o.finish_reason})
        json.dump(outs, open(os.path.join(out, "correct.json"), "w")); print("CORRECT_DONE", len(outs), flush=True)
        return
    if E("IMGPASS", "0") == "1":
        sp = SamplingParams(temperature=0, max_tokens=48)
        outs = []
        for k in range(8):
            q, im = pass_image(IMG, k)
            x = llm.generate([{"prompt": q, "multi_modal_data": {"image": im}}], sp, use_tqdm=False)[0]
            outs.append({"k": k, "ids": list(x.outputs[0].token_ids), "text": x.outputs[0].text, "n_prompt": len(x.prompt_token_ids)})
        json.dump(outs, open(os.path.join(out, "imgpass.json"), "w")); print("IMGPASS_DONE", [o["text"][:60] for o in outs], flush=True)
        return
    sp = lambda ol: SamplingParams(temperature=0, max_tokens=ol, ignore_eos=True)
    t = time.perf_counter(); llm.generate(timed_set(IMG, N, 1000)[:8], sp(8), use_tqdm=False); warm = time.perf_counter() - t
    ts, ntoks, nprompt, shas, seqs = [], [], [], [], []
    for rep in range(REPS):
        p = timed_set(IMG, N, 1 + rep)
        t = time.perf_counter()
        res = [x for q in p for x in llm.generate([q], sp(OL), use_tqdm=False)] if SEQ else llm.generate(p, sp(OL), use_tqdm=False)
        dt = time.perf_counter() - t
        ids = [list(x.outputs[0].token_ids) for x in res]
        ts.append(dt); ntoks.append(sum(len(x) for x in ids)); nprompt.append([len(x.prompt_token_ids) for x in res])
        shas.append(hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16])
        seqs += [{"seed": 1 + rep, "i": i, "out": o} for i, o in enumerate(ids)]
    d = {"startup_s": startup, "warm_s": warm, "per_rep_s": ts, "median_s": statistics.median(ts), "metric_s": statistics.median(ts),
         "out_tokens_per_rep": ntoks, "prompt_tokens_per_rep": nprompt, "out_sha_per_rep": shas, "M": M, "N": N, "IMG": IMG,
         "SEQ": SEQ, "OL": OL, "REPS": REPS, "LLM_KW": json.loads(json.dumps(kw, default=str))}
    json.dump(seqs, open(os.path.join(out, "seqs.json"), "w"))
    json.dump(d, open(os.path.join(out, "latency.json"), "w")); print(json.dumps({k: d[k] for k in ("per_rep_s", "median_s")}), flush=True)


HIST = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)   # the teacher-forced check counts output tokens whose gap exceeds each value (nats)


def tf_check(llm, SamplingParams, out, dirs, IMG):
    t0 = time.perf_counter(); res = {}; tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1)
    for d in dirs:
        items = []   # (label, prompt text, image, output ids)
        if os.path.exists(os.path.join(d, "seqs.json")):
            lat = json.load(open(os.path.join(d, "latency.json"))); img = lat.get("IMG", IMG)
            if img != IMG:   # transfer loads with another image size: this engine would resize their images; not re-read
                res[d] = {"skipped": f"IMG {img} != {IMG}"}; continue
            for s in json.load(open(os.path.join(d, "seqs.json"))):
                items.append((f"r{s['seed']}_{s['i']}", TXT, timed_image(img, s["seed"], s["i"]), s["out"]))
        if os.path.exists(os.path.join(d, "imgpass.json")):
            for s in json.load(open(os.path.join(d, "imgpass.json"))):
                q, im = pass_image(IMG, s["k"]); items.append((f"p{s['k']}", q, im, s["ids"]))
        nonarg = []; maxg = 0.0; npos = 0; bad = 0; hist = {t: 0 for t in HIST}; sumg = 0.0; per = {}
        for lab, q, im, o in items:
            if not o: continue
            pids = tok.encode(q, add_special_tokens=False)
            x = llm.generate([{"prompt_token_ids": pids + list(o), "multi_modal_data": {"image": im}}], sp, use_tqdm=False)[0]
            pl = x.prompt_logprobs; L = len(x.prompt_token_ids)
            if pl is None or len(pl) != L or list(x.prompt_token_ids[L - len(o):]) != list(o): bad += 1; continue
            mg = 0.0
            for j, t in enumerate(o):
                dct = pl[L - len(o) + j]
                if not dct or t not in dct: bad += 1; continue
                top = max(dct.items(), key=lambda kv: kv[1].logprob)
                g = max(0.0, top[1].logprob - dct[t].logprob); npos += 1; sumg += g; mg = max(mg, g)
                for h in HIST:
                    if g > h: hist[h] += 1
                if top[0] != t: nonarg.append([lab, j, round(g, 4)])
            maxg = max(maxg, mg); per[lab] = round(mg, 4)
        res[d] = {"all": {"n_pos": npos, "n_nonargmax": len(nonarg), "max_gap": round(maxg, 4), "len_mismatch": bad,
                          "n_gap_above": {str(t): n for t, n in hist.items()}, "mean_gap": sumg / max(npos, 1),
                          "per_item_max_gap": per, "nonargmax": sorted(nonarg, key=lambda z: -z[2])[:200]}}
        print("TF", d, (npos, len(nonarg), round(maxg, 4), bad), flush=True)
    json.dump({"dirs": res, "tf_s": time.perf_counter() - t0}, open(os.path.join(out, "tf.json"), "w"))


if __name__ == "__main__":   # vLLM may start the engine core with spawn, which re-imports this module
    main()
