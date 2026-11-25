"""correct.py <model> : #14930 correctness run, same engine settings as runner.py (n-gram k=3, V1).
Prompt groups (40 prompts, none of them used by the timed load):
  0-15 lure prompts as in the timed load (pool14930.SEL[448:464], 1024 tokens), 16-23 exact repeats (1024 tokens),
  24-31 random tokens (no repetition: few or no drafts), 32-39 English instructions (tokenized text).
Part 1, greedy: 64 tokens each (ignore_eos); token ids, text, per-step increments. The target model's greedy output is
  what spec decoding must reproduce, so ids must equal the parent's.
Part 2, sampled: temperature 0.8, per-request seed, 128 tokens each; outputs are random, so only statistics are compared:
  tokens per engine step (mean, histogram), output lengths, increments never above K+1. This is the only place the
  random (non-greedy) path of the rejection sampler is exercised by the scorer.
Writes $R4_OUT/correct.json."""
import hashlib, json, os, random, sys
OUT = os.environ["R4_OUT"]; MODEL = sys.argv[1]; K = 3
from vllm import LLM, SamplingParams
llm = LLM(model=MODEL, speculative_model="[ngram]", num_speculative_tokens=K, ngram_prompt_lookup_max=4,
          ngram_prompt_lookup_min=2, max_model_len=4096, gpu_memory_utilization=0.80, seed=0)
tok = llm.get_tokenizer()
rng = random.Random(4242)
P, groups = [], []
def lure_ids(seed, IN=1024, DECOY=4):   # same construction as runner.py prompt_ids (KIND=lure)
    r = random.Random(seed); seg = [r.randrange(100, 30000) for _ in range(40)]; first = list(seg)
    for p in r.sample(range(40), DECOY):
        first[p] = r.randrange(100, 30000)
    ids = list(first)
    while len(ids) < IN:
        ids += seg
    return ids[:IN]
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import pool14930 as _P; LS = list(_P.SEL[448:464])
except Exception:
    LS = [900000 + i for i in range(16)]
for s_ in LS:
    P.append({"prompt_token_ids": lure_ids(s_)}); groups.append("lure")
for i in range(8):
    seg = [rng.randrange(100, 30000) for _ in range(40)]
    P.append({"prompt_token_ids": (seg * 26)[:1024]}); groups.append("exact")
for i in range(8):
    P.append({"prompt_token_ids": [rng.randrange(100, 30000) for _ in range(1024)]}); groups.append("random")
texts = ["Write the numbers from {} to {} in words, one per line.", "List {} facts about the planet Mars, numbered.",
         "Repeat the sentence 'the quick brown fox jumps over the lazy dog' {} times.",
         "Explain in {} short steps how to make a cup of tea.", "Write a Python function that returns the first {} primes.",
         "Describe a city skyline at night in {} sentences.", "Give {} examples of words that rhyme with 'light'.",
         "Summarize the rules of chess in {} bullet points."]
for i, t in enumerate(texts):
    P.append({"prompt_token_ids": tok.encode(t.format(*([i + 5, i + 40] if "{} to {}" in t else [i + 5])))}); groups.append("text")
_trace = {"inc": {}, "steps": 0}
core = llm.llm_engine.engine_core; og = core.get_output
def _go(*a, **kw):
    o = og(*a, **kw)
    try:
        if o.outputs:
            _trace["steps"] += 1
            for x in o.outputs:
                _trace["inc"].setdefault(x.request_id, []).append(len(x.new_token_ids))
    except Exception:
        pass
    return o
core.get_output = _go
def run(params):
    _trace["inc"] = {}; _trace["steps"] = 0
    res = llm.generate(P, params, use_tqdm=False); r = []
    for o in sorted(res, key=lambda o: int(o.request_id)):
        c = o.outputs[0]
        r.append({"ids": list(c.token_ids), "text": c.text, "finish": c.finish_reason, "inc": _trace["inc"].get(o.request_id, [])})
    return r, _trace["steps"]
greedy, gs = run(SamplingParams(temperature=0, max_tokens=64, ignore_eos=True))
sampled, ss = run([SamplingParams(temperature=0.8, max_tokens=128, ignore_eos=True, seed=7000 + i) for i in range(len(P))])
json.dump({"groups": groups, "k": K, "greedy": greedy, "greedy_steps": gs, "sampled": sampled, "sampled_steps": ss,
           "sampled_ol": 128, "greedy_ol": 64}, open(os.path.join(OUT, "correct.json"), "w"))
print("ok", len(P))
