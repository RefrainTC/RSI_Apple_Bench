"""sgl_c5y.py <out dir> : workload, correctness pass and teacher-forced output check of the SGLang #9460 task (trusted,
ours). The timed part is batch 4's bench/sgl_b4.py IMG mode, unchanged: same images, text, seeds, digests and metric.

Timed mode (default): sglang.Engine, greedy, ignore_eos. N requests per rep, each = one synthetic IMG x IMG PNG (seeded
  per rep and request) + a fixed chat prompt -> OL tokens, all N in one generate() call. One untimed 8-request warm-up
  (seed 1000), then REPS timed reps, rep k on image seed 1 + k (identical in both arms). Metric: median rep wall time.
  TEXT_ONLY=1 (transfer load): the same chat prompt without the image (mm_input None branch).
  Writes latency.json, settings.json (live ServerArgs) and seqs.json (image spec and generated ids of every request).
  The per-rep digest is batch 4's (sha256 of output_ids + text as returned), so digests compare with batch 4's runs.
CORRECT=1: untimed correctness pass. 16 fixed prompts (12 with an image of a different pattern and size, 4 text-only),
  sent ONE AT A TIME (batch 1: no arrival grouping, so a correct candidate reproduces the parent bit for bit), 64 greedy
  tokens with the produced token's logprob and the top-5 logprobs -> correct.json (+ seqs.json for the teacher-forced check).
TF=1 (run in the PARENT venv only; no SGLang engine): teacher-forced re-read with Hugging Face transformers
  (Qwen2VLForConditionalGeneration, bf16, sdpa; output head in fp32). For every run dir in TF_DIRS (":"-separated) it rebuilds each request's
  prompt with the HF processor from the same image and text, appends the produced output ids, runs one forward pass and
  records for every output token gap = top-1 logprob - logprob of the produced token (0 = argmax) -> <out>/tf.json.
  The HF re-read computes the M-RoPE positions of the whole sequence itself (get_rope_index), independently of SGLang's
  decode path, which is what the PR changes.
Env: M N OL IMG REPS MEMF SGL_KW TEXT_ONLY CORRECT TF TF_DIRS SGLANG_PORT_BASE."""
import dataclasses, hashlib, json, os, statistics, sys, time

TXT = ("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image in detail.<|im_end|>\n"
       "<|im_start|>assistant\n")
TXT_NOIMG = "<|im_start|>user\nDescribe this image in detail.<|im_end|>\n<|im_start|>assistant\n"
COR_Q = ["Describe this image in detail.", "What colors do you see?", "Is there any text in this picture?",
         "Describe the pattern in one paragraph."]
COR_TEXT = ["Write a short poem about the sea.", "Explain what a rotary position embedding is.",
            "List three uses of a GPU.", "What is the capital of France? Answer in one sentence."]


def chat(q, img):
    v = "<|vision_start|><|image_pad|><|vision_end|>" if img else ""
    return f"<|im_start|>user\n{v}{q}<|im_end|>\n<|im_start|>assistant\n"


def make_image(seed, i, px, path):
    """Batch 4's synthetic image (sgl_b4.py): stripes depending on i plus seeded noise. Deterministic."""
    import numpy as np
    from PIL import Image
    yy, xx = np.mgrid[0:px, 0:px]
    nr = np.random.default_rng(seed)
    # sgl_b4.py draws the noise of all N images of a seed from one generator in order; reproduce that order.
    for k in range(i + 1):
        base = np.stack([(xx * (k % 7 + 1)) % 256, (yy * (k % 5 + 1)) % 256, ((xx + yy) * (k % 3 + 1)) % 256], -1)
        noise = nr.integers(0, 64, base.shape)
    img = Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8))
    if path: img.save(path)
    return img


def imgset(seed, n, px):
    """Same files as sgl_b4.py's imgset(seed) (one generator for the n images, in order)."""
    import numpy as np
    from PIL import Image
    os.makedirs("img", exist_ok=True)
    yy, xx = np.mgrid[0:px, 0:px]
    nr = np.random.default_rng(seed); paths = []
    for i in range(n):
        base = np.stack([(xx * (i % 7 + 1)) % 256, (yy * (i % 5 + 1)) % 256, ((xx + yy) * (i % 3 + 1)) % 256], -1)
        p = os.path.abspath(f"img/s{seed}_{i}.png")
        Image.fromarray(np.clip(base + nr.integers(0, 64, base.shape), 0, 255).astype(np.uint8)).save(p); paths.append(p)
    return paths


def gen_ids(x):
    """The generated token ids of one result. In this SGLang version "output_ids" of a non-streamed result also carries the
    last 5 prompt tokens (the detokenizer's surrogate offset), so keep the last completion_tokens ids."""
    n = x["meta_info"]["completion_tokens"]
    return list(x["output_ids"])[-n:] if n else []


def cor_items():
    """16 fixed correctness prompts: (text, image spec or None). Image spec = (seed, index, px) of make_image."""
    items = []
    for j in range(12):
        items.append((chat(COR_Q[j % 4], True), (5000 + j, j, (224, 336, 448)[j % 3])))
    for q in COR_TEXT:
        items.append((chat(q, False), None))
    return items


def main():
    out = sys.argv[1]; E = os.environ.get
    M = E("M"); N = int(E("N", "64")); OL = int(E("OL", "256")); IMG = int(E("IMG", "224")); REPS = int(E("REPS", "5"))
    CORRECT = E("CORRECT", "0") == "1"; TF = E("TF", "0") == "1"; TEXT_ONLY = E("TEXT_ONLY", "0") == "1"
    if not os.path.isdir(M):
        from huggingface_hub import snapshot_download
        M = snapshot_download(M, local_files_only=True)
    if TF:
        tf_check(M, out, [d for d in E("TF_DIRS", "").split(":") if d]); return
    import sglang as sgl
    kw = dict(model_path=M, mem_fraction_static=float(E("MEMF", "0.80")), random_seed=0, log_level="error")
    if E("SGLANG_PORT_BASE"): kw["port"] = int(E("SGLANG_PORT_BASE"))
    if E("SGL_KW"): kw.update(json.loads(E("SGL_KW")))
    t0 = time.perf_counter(); llm = sgl.Engine(**kw); startup = time.perf_counter() - t0
    try:
        sa = dataclasses.asdict(llm.server_args)
        json.dump({k: v for k, v in sa.items() if isinstance(v, (str, int, float, bool, type(None)))},
                  open(os.path.join(out, "settings.json"), "w"), sort_keys=True)
    except Exception as e:  # recorded; the settings check then fails
        json.dump({"error": repr(e)}, open(os.path.join(out, "settings.json"), "w"))
    sp = lambda ol: {"temperature": 0, "max_new_tokens": ol, "ignore_eos": True}
    if CORRECT:
        correct(llm, out, sp); llm.shutdown(); return

    def digest(res):
        return hashlib.sha256(json.dumps([{"ids": r.get("output_ids"), "text": r.get("text")} for r in res]).encode()).hexdigest()[:16]

    if TEXT_ONLY:
        run = lambda seed, n, ol: llm.generate(prompt=[TXT_NOIMG] * n, sampling_params=sp(ol))
    else:
        run = lambda seed, n, ol: llm.generate(prompt=[TXT] * n, image_data=imgset(seed, N, IMG)[:n], sampling_params=sp(ol))
    t = time.perf_counter(); run(1000, 8, 8); warm = time.perf_counter() - t
    ts, ntoks, shas, seqs = [], [], [], []
    for rep in range(REPS):
        t = time.perf_counter(); res = run(1 + rep, N, OL); ts.append(time.perf_counter() - t); shas.append(digest(res))
        ntoks.append(sum(x["meta_info"]["completion_tokens"] for x in res))
        seqs.append([{"text": TXT_NOIMG if TEXT_ONLY else TXT, "img": None if TEXT_ONLY else [1 + rep, i, IMG, N],
                      "out": gen_ids(x)} for i, x in enumerate(res)])
    d = {"startup_s": startup, "warm_s": warm, "per_rep_s": ts, "median_s": statistics.median(ts), "metric_s": statistics.median(ts),
         "out_tokens_per_rep": ntoks, "out_sha_per_rep": shas, "M": M, "N": N, "OL": OL, "REPS": REPS, "IMG": IMG,
         "TEXT_ONLY": TEXT_ONLY, "ENGINE_KW": {k: v for k, v in kw.items()}}
    json.dump(seqs, open(os.path.join(out, "seqs.json"), "w"))
    json.dump(d, open(os.path.join(out, "latency.json"), "w")); print(json.dumps({k: d[k] for k in ("per_rep_s", "median_s")}))
    llm.shutdown()


def correct(llm, out, sp):
    os.makedirs("img", exist_ok=True); outs = []; seqs = []
    for k, (text, spec) in enumerate(cor_items()):
        a = {}
        if spec:
            p = os.path.abspath(f"img/cor_{k}.png"); make_image(*spec, p); a["image_data"] = p
        x = llm.generate(prompt=text, sampling_params=sp(64), return_logprob=True, top_logprobs_num=5, **a)
        mi = x["meta_info"]
        outs.append({"ids": [t[1] for t in mi["output_token_logprobs"]], "lp": [t[0] for t in mi["output_token_logprobs"]],
                     "top": [[[t[0], t[1]] for t in d] for d in mi["output_top_logprobs"]], "text": x["text"]})
        seqs.append({"text": text, "img": list(spec) if spec else None, "cor": True, "out": outs[-1]["ids"]})
    json.dump(outs, open(os.path.join(out, "correct.json"), "w"))
    json.dump([seqs], open(os.path.join(out, "seqs.json"), "w"))
    print("CORRECT_DONE", len(outs))


HIST = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)   # the teacher-forced check counts output tokens whose gap exceeds each value (nats)


def tf_check(M, out, dirs):
    """Teacher-forced re-read with HF transformers. gap = top-1 logprob - logprob of the produced token (0 = argmax)."""
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    t0 = time.perf_counter()
    proc = AutoProcessor.from_pretrained(M)
    model = Qwen2VLForConditionalGeneration.from_pretrained(M, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    W = model.lm_head.weight.float()
    os.makedirs("img", exist_ok=True)
    cache = {}

    def image(spec):
        spec = tuple(spec)
        if spec not in cache:
            if len(spec) == 4:   # timed run: image i of the seed's set of n (sgl_b4.py generator order)
                seed, i, px, n = spec
                if (seed, px, n) not in cache:
                    from PIL import Image
                    cache[(seed, px, n)] = [Image.open(p).convert("RGB") for p in imgset(seed, n, px)]
                cache[spec] = cache[(seed, px, n)][i]
            else:
                cache[spec] = make_image(*spec, None)
        return cache[spec]

    res = {}
    for d in dirs:
        seqs = json.load(open(os.path.join(d, "seqs.json")))
        items = [(rep, i, s) for rep, rs in enumerate(seqs) for i, s in enumerate(rs)]
        nonarg = []; maxg = 0.0; npos = 0; bad = 0; per_rep = {}; hist = {t: 0 for t in HIST}; sumg = 0.0
        B = 8
        # group items that share prompt shape (same text, same image size or none, same output length) into batches
        groups = {}
        for it in items:
            s = it[2]; key = (s["text"], tuple(s["img"][2:3]) if s["img"] else None, len(s["out"]))
            groups.setdefault(key, []).append(it)
        for key, its in groups.items():
            for k in range(0, len(its), B):
                chunk = its[k:k + B]
                texts = [s["text"] for _, _, s in chunk]
                imgs = [image(s["img"]) for _, _, s in chunk if s["img"]]
                enc = proc(text=texts, images=imgs or None, return_tensors="pt", padding=False)
                ol = key[2]
                oids = torch.tensor([s["out"] for _, _, s in chunk], dtype=torch.long)
                ids = torch.cat([enc["input_ids"], oids], 1).cuda()
                kw = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
                if imgs:
                    kw["pixel_values"] = enc["pixel_values"].cuda(); kw["image_grid_thw"] = enc["image_grid_thw"].cuda()
                with torch.no_grad():   # hidden states of the last ol + 1 positions; position t predicts output token t
                    h = model.model(**kw).last_hidden_state[:, -(ol + 1):-1]
                    lg = torch.nn.functional.linear(h.float(), W)   # lm_head in fp32: finer gaps than bf16 logits
                lp = torch.log_softmax(lg, -1)
                top = lp.max(-1)
                got = lp.gather(-1, oids.cuda().unsqueeze(-1)).squeeze(-1)
                gap = (top.values - got).clamp_min(0).cpu()
                am = top.indices.cpu()
                for r, (rep, i, s) in enumerate(chunk):
                    pr = per_rep.setdefault(rep, [0, 0, 0.0])
                    for j in range(ol):
                        g = float(gap[r, j]); npos += 1; pr[0] += 1; sumg += g
                        for t in HIST:
                            if g > t: hist[t] += 1
                        if int(am[r, j]) != s["out"][j]:
                            nonarg.append([rep, i, j, round(g, 4)]); pr[1] += 1
                        maxg = max(maxg, g); pr[2] = max(pr[2], g)
                del lg, lp
        res[d] = {"all": {"n_pos": npos, "n_nonargmax": len(nonarg), "max_gap": round(maxg, 4), "len_mismatch": bad,
                          "n_gap_above": {str(t): n for t, n in hist.items()}, "mean_gap": sumg / max(npos, 1),
                          "per_rep": {str(k): [v[0], v[1], round(v[2], 4)] for k, v in per_rep.items()},
                          "nonargmax": sorted(nonarg, key=lambda z: -z[3])[:200]}}
        print("TF", d, (npos, len(nonarg), round(maxg, 4)), flush=True)
    json.dump({"dirs": res, "tf_s": time.perf_counter() - t0}, open(os.path.join(out, "tf.json"), "w"))


if __name__ == "__main__":
    main()
