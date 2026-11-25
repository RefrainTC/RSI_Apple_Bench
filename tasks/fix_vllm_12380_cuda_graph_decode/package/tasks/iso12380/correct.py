"""correct.py <model> : #12380 correctness run. Greedy outputs (token ids, text, top-5 logprobs per step) for fixed text
prompts, with the task's engine settings (defaults: CUDA graphs on), in several generate() calls of different batch
sizes, so decode batches hit captured graph sizes exactly (8, 1) and with padding (5 -> 8, 3 -> 4), after prefills (the
bug's trigger). 48 tokens each, min_tokens 48 is not forced (EOS allowed: an early stop must match too).
Writes $R4_OUT/correct.json: {"groups": [batch label per sequence], "seq": [...]}."""
import json, os, sys
from vllm import LLM, SamplingParams
TOPICS = ["the water cycle", "how a CPU cache works", "the causes of the French Revolution", "photosynthesis",
          "binary search", "plate tectonics", "the history of the printing press", "how vaccines work",
          "the rules of chess", "compound interest", "how a bicycle stays upright", "the Doppler effect",
          "the structure of DNA", "how rainbows form", "the difference between TCP and UDP", "black holes",
          "how a refrigerator works", "the Pythagorean theorem", "the life cycle of a star", "how tides work",
          "the immune system", "public-key cryptography", "the Silk Road", "how bees make honey", "the stock market"]
STYLES = ["Explain {} to a ten-year-old.", "Write three short facts about {}.", "Summarise {} in one paragraph.",
          "List the key ideas behind {}.", "Describe {} as a short story."]
def prompts(start, n):
    return [STYLES[(start + i) % len(STYLES)].format(TOPICS[(start + i) % len(TOPICS)]) for i in range(n)]
llm = LLM(model=sys.argv[1], gpu_memory_utilization=0.85, seed=0)
sp = SamplingParams(temperature=0, max_tokens=48, logprobs=5)
plan = [("b8a", 0, 8), ("b5", 8, 5), ("b1", 13, 1), ("b3", 14, 3), ("b8b", 17, 8)]
groups, seq = [], []
for label, s, n in plan:
    for o in llm.generate(prompts(s, n), sp, use_tqdm=False):
        c = o.outputs[0]
        seq.append({"ids": list(c.token_ids), "text": c.text, "finish": c.finish_reason,
                    "top": [{str(t): float(v.logprob) for t, v in step.items()} for step in (c.logprobs or [])]})
        groups.append(label)
json.dump({"groups": groups, "seq": seq}, open(os.path.join(os.environ["R4_OUT"], "correct.json"), "w"))
print("ok", len(seq))
