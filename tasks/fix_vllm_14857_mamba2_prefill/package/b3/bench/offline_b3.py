"""offline_b3.py <out dir> : batch-3 offline screening benchmark for vLLM (trusted, ours; extends screen/offline_bench.py).
N prompts of IN random token ids (KIND=repeat: a 40-token segment repeated), SamplingParams(temperature=TEMP,
max_tokens=OL, ignore_eos). IMG=<px>: Qwen2-VL style request, one synthetic RGB image of IMG x IMG per prompt (seeded
noise + gradient, different per request, same across reps) plus a short text prompt. One small untimed warm-up generate,
then REPS timed generate calls on the same prompts. Writes <out>/latency.json: wall_s (sum), per_rep_s, median_s,
output token counts and a sha256 of all output token ids per rep (fix vs parent output check).
Env: M N IN OL TEMP REPS MEMU KIND IMG LLM_KW (JSON merged into LLM kwargs).
Calibration options (off by default, so screens are unchanged): ROTATE=1 gives every timed rep its own seeded prompt set
(seed 1 + rep; warm-up seed 1000), so nothing can be reused across reps. CORRECT=1 replaces the timed part by a
correctness pass: 24 fixed prompts of real text in 3 generate() calls (batch 16, 1, 7), 48 greedy tokens, top-5 logprobs,
written to <out>/correct.json."""
import hashlib, json, os, random, statistics, sys, time
from vllm import LLM, SamplingParams
out = sys.argv[1]; E = os.environ.get
M = E("M"); N = int(E("N", "256")); IN = int(E("IN", "128")); OL = int(E("OL", "128")); TEMP = float(E("TEMP", "0"))
REPS = int(E("REPS", "3")); KIND = E("KIND", "random"); IMG = int(E("IMG", "0"))
kw = {"gpu_memory_utilization": float(E("MEMU", "0.80")), "seed": 0}
if E("LLM_KW"): kw.update(json.loads(E("LLM_KW")))
rng = random.Random(0)
tok = lambda n: [rng.randrange(100, 30000) for _ in range(n)]
ROTATE = E("ROTATE", "0") == "1"; CORRECT = E("CORRECT", "0") == "1"
def promptset(seed):
    global rng
    rng = random.Random(seed)
    if KIND == "repeat":
        return [{"prompt_token_ids": (s * (IN // 40 + 1))[:IN]} for s in (tok(40) for _ in range(N))]
    return [{"prompt_token_ids": tok(IN)} for _ in range(N)]
if IMG:
    import numpy as np
    from PIL import Image
    nr = np.random.default_rng(0)
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    def image(i):
        base = np.stack([(xx * (i % 7 + 1)) % 256, (yy * (i % 5 + 1)) % 256, ((xx + yy) * (i % 3 + 1)) % 256], -1)
        return Image.fromarray(np.clip(base + nr.integers(0, 64, base.shape), 0, 255).astype(np.uint8))
    txt = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n"
    prompts = [{"prompt": txt, "multi_modal_data": {"image": image(i)}} for i in range(N)]
elif KIND == "repeat":
    prompts = [{"prompt_token_ids": (s * (IN // 40 + 1))[:IN]} for s in (tok(40) for _ in range(N))]
else:
    prompts = [{"prompt_token_ids": tok(IN)} for _ in range(N)]
llm = LLM(model=M, **kw)
sp = lambda ol: SamplingParams(temperature=TEMP, max_tokens=ol, ignore_eos=True)
if CORRECT:
    words = "the of and to in is was for on that with as by at from his her it an were are which this be or has had".split()
    r2 = random.Random(7)
    texts = [" ".join(r2.choice(words) for _ in range(20 + 7 * i)) for i in range(24)]
    tk = llm.get_tokenizer(); outs = []
    p2 = SamplingParams(temperature=0, max_tokens=48, logprobs=5, ignore_eos=True)
    for a, b in ((0, 16), (16, 17), (17, 24)):
        for r in llm.generate([{"prompt_token_ids": tk.encode(t)} for t in texts[a:b]], p2):
            o = r.outputs[0]
            outs.append({"ids": list(o.token_ids), "text": o.text,
                         "lp": [{str(k): v.logprob for k, v in d.items()} for d in o.logprobs]})
    json.dump(outs, open(os.path.join(out, "correct.json"), "w")); print("CORRECT_DONE", len(outs)); sys.exit(0)
if ROTATE: prompts = promptset(1000)
llm.generate(prompts[:8], sp(8))
ts, ntoks, shas = [], [], []
for rep in range(REPS):
    if ROTATE: prompts = promptset(1 + rep)
    t = time.perf_counter(); res = llm.generate(prompts, sp(OL)); ts.append(time.perf_counter() - t)
    ids = [list(r.outputs[0].token_ids) for r in res]
    ntoks.append(sum(len(x) for x in ids)); shas.append(hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16])
d = {"wall_s": sum(ts), "per_rep_s": ts, "median_s": statistics.median(ts), "out_tokens_per_rep": ntoks, "out_sha_per_rep": shas,
     "M": M, "N": N, "IN": IN, "OL": OL, "TEMP": TEMP, "REPS": REPS, "KIND": KIND, "IMG": IMG, "ROTATE": ROTATE, "LLM_KW": kw}
json.dump(d, open(os.path.join(out, "latency.json"), "w")); print(json.dumps(d))
