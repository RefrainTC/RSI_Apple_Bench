"""runner.py <model> : P2 workload driver for ISO-Bench #7822 (rework 2026-11-23 after the independent review, F2/F3/F5).
Replaces running the upstream benchmark_prefix_caching.py; the engine is built exactly as that script builds it:
  LLM(model, tokenizer_mode='auto', trust_remote_code=True, enforce_eager=True, use_v2_block_manager=True,
      enable_prefix_caching=True), CacheConfig.gpu_memory_utilization capped at 0.85 (24 GB card).
Workload (env):
  WORKLOAD=shared (official): the upstream script's 645-token PROMPT as a shared prefix + REPEAT distinct questions
    (indices SUFFIX_BASE .. SUFFIX_BASE+REPEAT-1); identical requests are gone, so deduplication is not free.
  WORKLOAD=identical: the upstream workload (PROMPT repeated REPEAT times); hidden transfer workload only.
  Greedy, max_tokens=OL, ignore_eos=True (every request decodes exactly OL tokens, independent of numerics).
  Two generate() passes over the same prompts, as upstream: pass 0 starts from an empty prefix cache (= the metric),
  pass 1 hits the cache everywhere (recorded, checked, not scored).
  LOGPROBS=5 (reference mode only): same, with top-5 logprobs, for the output check's tie test.
Writes to $R4_OUT: passes.json (in-process wall per pass + counters), outputs.json (per pass, per request: token ids,
text sha256, finish_reason), settings.json (effective engine settings read after init: the harness asserts they equal
the parent's; e.g. enforce_eager must hold and no CUDA graphs may be captured)."""
import ast, hashlib, json, os, sys
from time import perf_counter, sleep as _sleep, time as _now   # bound before any vllm import
OUT = os.environ["R4_OUT"]; MODEL = sys.argv[1]
OL = int(os.environ.get("OL", "200")); REPEAT = int(os.environ.get("REPEAT", "100"))
WL = os.environ.get("WORKLOAD", "shared"); SB = int(os.environ.get("SUFFIX_BASE", "0")); LP = int(os.environ.get("LOGPROBS", "0"))
BENCH = os.environ["P2_BENCH"]
src = open(BENCH).read()
PROMPT = next(ast.literal_eval(n.value) for n in ast.parse(src).body
              if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "PROMPT" for t in n.targets))
TOPICS = ["the first column", "the last column", "the row with the largest value", "the row with the smallest value",
          "any duplicated entries", "the overall trend", "missing or empty cells", "the second row",
          "the middle rows", "the header names"]
def question(i):
    return (f"\n\nQuestion {i}: Look at {TOPICS[i % len(TOPICS)]} of the table above and write a detailed, "
            f"numbered explanation of what it shows, citing at least {3 + i % 5} specific values.")
prompts = [PROMPT] * REPEAT if WL == "identical" else [PROMPT + question(SB + i) for i in range(REPEAT)]
T0 = _now()
CAP = 0.85
import vllm.config as _vc
_orig_init = _vc.CacheConfig.__init__
def _init(self, *a, **kw):
    _orig_init(self, *a, **kw)
    if self.gpu_memory_utilization > CAP:
        self.gpu_memory_utilization = CAP
_vc.CacheConfig.__init__ = _init
_st = {}; hooks = []
def _reset():
    _st.update(steps=0, prefill_steps=0, prefill_tokens=0, computed_prefill_tokens=0, decode_tokens=0)
_reset()
try:
    import vllm.core.scheduler as _S
    _os = _S.Scheduler.schedule
    def _schedule(self, *a, **kw):
        r = _os(self, *a, **kw); so = r[1]; _st["steps"] += 1
        if so.num_prefill_groups > 0:
            _st["prefill_steps"] += 1; _st["prefill_tokens"] += so.num_batched_tokens
        return r
    _S.Scheduler.schedule = _schedule; hooks.append("scheduler")
except Exception as e:
    hooks.append(f"scheduler_hook_failed:{e!r}")
try:
    import vllm.worker.model_runner as _MR
    _op = _MR.ModelRunner.prepare_model_input
    def _prep(self, *a, **kw):
        mi = _op(self, *a, **kw); am = getattr(mi, "attn_metadata", None)
        if am is not None:
            if getattr(am, "num_prefills", 0) > 0:
                _st["computed_prefill_tokens"] += am.num_prefill_tokens
            _st["decode_tokens"] += getattr(am, "num_decode_tokens", 0) or 0
        return mi
    _MR.ModelRunner.prepare_model_input = _prep; hooks.append("model_runner")
except Exception as e:
    hooks.append(f"model_runner_hook_failed:{e!r}")
from vllm import LLM, SamplingParams
llm = LLM(model=MODEL, tokenizer_mode="auto", trust_remote_code=True, enforce_eager=True, use_v2_block_manager=True,
          enable_prefix_caching=True)
# effective settings, read back from the live engine (F5)
def _g(f):
    try:
        return f()
    except Exception as e:
        return f"error:{type(e).__name__}"
eng = llm.llm_engine
def _graphs():
    mr = eng.model_executor.driver_worker.model_runner; g = mr.graph_runners
    return sum(len(x) for x in g) if isinstance(g, (list, tuple)) else len(g)
settings = {
    "enforce_eager": _g(lambda: eng.model_config.enforce_eager), "dtype": _g(lambda: str(eng.model_config.dtype)),
    "quantization": _g(lambda: eng.model_config.quantization), "max_model_len": _g(lambda: eng.model_config.max_model_len),
    "block_size": _g(lambda: eng.cache_config.block_size), "enable_prefix_caching": _g(lambda: eng.cache_config.enable_prefix_caching),
    "gpu_memory_utilization": _g(lambda: eng.cache_config.gpu_memory_utilization),
    "num_gpu_blocks": _g(lambda: eng.cache_config.num_gpu_blocks),
    "use_v2_block_manager": _g(lambda: eng.scheduler_config.use_v2_block_manager),
    "max_num_batched_tokens": _g(lambda: eng.scheduler_config.max_num_batched_tokens),
    "max_num_seqs": _g(lambda: eng.scheduler_config.max_num_seqs),
    "chunked_prefill": _g(lambda: eng.scheduler_config.chunked_prefill_enabled),
    "num_scheduler_steps": _g(lambda: getattr(eng.scheduler_config, "num_scheduler_steps", 1)),
    "captured_cuda_graphs": _g(_graphs),
    "block_manager": _g(lambda: type(eng.scheduler[0].block_manager).__name__),
}
json.dump(settings, open(os.path.join(OUT, "settings.json"), "w"), indent=1)
sp = SamplingParams(temperature=0, max_tokens=OL, ignore_eos=True, logprobs=LP or None)
passes, outs = [], []
# 2 s with the GPU idle before the timed passes: the out-of-process witness (harness/witness.py) takes the longest busy
# segment, so GPU work done during engine start-up (profile run, CUDA graph capture, any warm-up a candidate adds) must
# not merge with the passes' segment (canary C4r, 2026-11-23: graph capture right before pass 0 added 9 s to the window).
import torch
torch.cuda.synchronize(); _sleep(2.0)
for k in range(2):
    _reset(); t = perf_counter(); res = llm.generate(prompts, sp, use_tqdm=False); dt = perf_counter() - t
    rec = dict(_st, pass_index=k, wall_s=dt, num_requests=len(res), hooks=hooks, t_end_unix=_now())
    passes.append(rec); print("[P2COUNTER]", json.dumps(rec), flush=True)
    po = []
    for o in sorted(res, key=lambda o: int(o.request_id)):
        c = o.outputs[0]
        d = {"ids": list(c.token_ids), "text_sha256": hashlib.sha256(c.text.encode()).hexdigest(), "finish": c.finish_reason}
        if LP:
            d["top"] = [{str(t): float(v.logprob) for t, v in s.items()} for s in (c.logprobs or [])]
        po.append(d)
    outs.append(po)
json.dump({"workload": WL, "repeat": REPEAT, "ol": OL, "suffix_base": SB, "passes": outs},
          open(os.path.join(OUT, "outputs.json"), "w"))
json.dump({"passes": passes, "process_wall_s": _now() - T0, "workload": WL}, open(os.path.join(OUT, "passes.json"), "w"), indent=1)
