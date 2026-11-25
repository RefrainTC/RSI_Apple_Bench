"""correct.py <benchmark_prefix_caching.py> <model> : greedy outputs (token ids + top-5 logprobs per step) for a fixed
prompt set with the task's engine settings (prefix caching + BlockManagerV2, eager), cold cache then warm cache.
Prompt groups: 0-23 the benchmark PROMPT itself, 24-35 PROMPT + a short question (shared prefix, different suffix),
36-47 unrelated short prompts. Writes $R4_OUT/correct.json."""
import ast, json, os, sys
src = open(sys.argv[1]).read()
PROMPT = next(ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "PROMPT" for t in n.targets))
from vllm import LLM, SamplingParams
prompts = [PROMPT] * 24 + [PROMPT + f"\n\nQuestion {i}: summarise the table above in {i + 2} words." for i in range(12)] + \
          [f"Count from {i} to {i + 20}, separated by commas." for i in range(12)]
groups = ["same"] * 24 + ["shared_prefix"] * 12 + ["unrelated"] * 12
llm = LLM(model=sys.argv[2], tokenizer_mode="auto", trust_remote_code=True, enforce_eager=True, use_v2_block_manager=True,
          enable_prefix_caching=True, gpu_memory_utilization=0.85)
sp = SamplingParams(temperature=0, max_tokens=32, logprobs=5)
def run():
    res = []
    for o in llm.generate(prompts, sp):
        c = o.outputs[0]
        lps = [{str(t): float(v.logprob) for t, v in step.items()} for step in (c.logprobs or [])]
        res.append({"ids": list(c.token_ids), "top": lps, "text": c.text})
    return res
out = {"groups": groups, "seq": run(), "batch": run()}
json.dump(out, open(os.path.join(os.environ["R4_OUT"], "correct.json"), "w"))
print("ok", len(prompts))
