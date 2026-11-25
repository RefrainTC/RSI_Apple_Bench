"""runner.py <model> : P2 workload driver for ISO-Bench #12380 ("fix perf regression from #12253", vLLM 0.6.6.post2.dev364,
V0 engine). Built after vLLM's benchmarks/benchmark_latency.py (the PR's perf command): offline LLM with default engine
settings (CUDA graphs ON: enforce_eager=False), one batch of BS prompts of IN random token ids, OL output tokens,
ignore_eos. Deviations from upstream, on purpose:
  - real FP8 weights (RedHatAI/Meta-Llama-3-8B-Instruct-FP8) instead of --load-format dummy on gated Llama-3-8B;
  - greedy decoding (upstream samples at temperature 1.0): outputs become checkable;
  - every timed rep uses a DIFFERENT prompt set (seeded by rep index): memoising outputs across reps cannot pay;
  - gpu_memory_utilization 0.85, seed 0.
Workload env: BS (8) IN (32) OL (128) REPS (10) WARM (3 in-process untimed warm-up generate calls) SEED_BASE (0; the
harness's untimed warm-up jobs use 1000). LOGPROBS=5 (reference mode only).
Writes to $R4_OUT:
  passes.json  : per timed rep wall time (perf_counter around generate) + step trace counters; metric = mean rep time
  outputs.json : per rep, per request: token ids, text sha256, finish_reason (+ top-5 logprobs in reference mode)
  settings.json: effective engine settings read back from the live engine (enforce_eager, captured CUDA-graph batch
                 sizes, scheduler limits, attention backend, async output processing, ...)
  steps.json   : per timed rep, the model runner's step trace: for each execute_model call (num_prefills,
                 num_prefill_tokens, num_decode_tokens, use_cuda_graph, padded input rows). The harness requires the
                 candidate's (prefill tokens, decode tokens) sequence to equal the paired parent's (batch shapes must not
                 change) and reports how many decode steps replayed a CUDA graph."""
import hashlib, json, os, random, sys
from time import perf_counter, sleep as _sleep, time as _now   # bound before any vllm import
OUT = os.environ["R4_OUT"]; MODEL = sys.argv[1]
BS = int(os.environ.get("BS", "8")); IN = int(os.environ.get("IN", "32")); OL = int(os.environ.get("OL", "128"))
REPS = int(os.environ.get("REPS", "10")); WARM = int(os.environ.get("WARM", "3")); SB = int(os.environ.get("SEED_BASE", "0"))
LP = int(os.environ.get("LOGPROBS", "0"))
def prompts_for(seed):
    rng = random.Random(1_000_003 * (seed + 1))
    return [{"prompt_token_ids": [rng.randrange(100, 10000) for _ in range(IN)]} for _ in range(BS)]
T0 = _now()
CAP = 0.85
import vllm.config as _vc
_orig_init = _vc.CacheConfig.__init__
def _init(self, *a, **kw):
    _orig_init(self, *a, **kw)
    if self.gpu_memory_utilization > CAP:
        self.gpu_memory_utilization = CAP
_vc.CacheConfig.__init__ = _init
hooks = []; _trace = []; _on = [False]
try:
    import vllm.worker.model_runner as _MR
    _oe = _MR.ModelRunner.execute_model
    def _exec(self, model_input, *a, **kw):
        if _on[0]:
            am = getattr(model_input, "attn_metadata", None); it = getattr(model_input, "input_tokens", None)
            _trace.append((getattr(am, "num_prefills", None), getattr(am, "num_prefill_tokens", None),
                           getattr(am, "num_decode_tokens", None), bool(getattr(am, "use_cuda_graph", False)),
                           int(it.shape[0]) if it is not None else None))
        return _oe(self, model_input, *a, **kw)
    _MR.ModelRunner.execute_model = _exec; hooks.append("execute_model")
except Exception as e:
    hooks.append(f"execute_model_hook_failed:{e!r}")
from vllm import LLM, SamplingParams
llm = LLM(model=MODEL, gpu_memory_utilization=CAP, seed=0)
def _g(f):
    try:
        return f()
    except Exception as e:
        return f"error:{type(e).__name__}"
eng = llm.llm_engine
def _mr():
    return eng.model_executor.driver_worker.model_runner
def _graph_sizes():
    g = _mr().graph_runners
    return sorted({int(k) for d in g for k in d}) if isinstance(g, (list, tuple)) else sorted(int(k) for k in g)
settings = {
    "enforce_eager": _g(lambda: eng.model_config.enforce_eager), "dtype": _g(lambda: str(eng.model_config.dtype)),
    "quantization": _g(lambda: eng.model_config.quantization), "max_model_len": _g(lambda: eng.model_config.max_model_len),
    "max_seq_len_to_capture": _g(lambda: eng.model_config.max_seq_len_to_capture),
    "use_async_output_proc": _g(lambda: eng.model_config.use_async_output_proc),
    "block_size": _g(lambda: eng.cache_config.block_size), "enable_prefix_caching": _g(lambda: eng.cache_config.enable_prefix_caching),
    "cache_dtype": _g(lambda: str(eng.cache_config.cache_dtype)),
    "gpu_memory_utilization": _g(lambda: eng.cache_config.gpu_memory_utilization),
    "num_gpu_blocks": _g(lambda: eng.cache_config.num_gpu_blocks),
    "max_num_batched_tokens": _g(lambda: eng.scheduler_config.max_num_batched_tokens),
    "max_num_seqs": _g(lambda: eng.scheduler_config.max_num_seqs),
    "chunked_prefill": _g(lambda: eng.scheduler_config.chunked_prefill_enabled),
    "num_scheduler_steps": _g(lambda: eng.scheduler_config.num_scheduler_steps),
    "speculative": _g(lambda: str(eng.speculative_config)),
    "compilation_level": _g(lambda: eng.vllm_config.compilation_config.level),
    "model_runner": _g(lambda: type(_mr()).__name__),
    "attn_backend": _g(lambda: _mr().attn_backend.get_name()),
    "max_batchsize_to_capture": _g(lambda: _mr().max_batchsize_to_capture),
    "captured_cuda_graph_sizes": _g(_graph_sizes),
}
settings["captured_cuda_graphs"] = len(settings["captured_cuda_graph_sizes"]) if isinstance(settings["captured_cuda_graph_sizes"], list) else settings["captured_cuda_graph_sizes"]
json.dump(settings, open(os.path.join(OUT, "settings.json"), "w"), indent=1)
sp = SamplingParams(temperature=0, max_tokens=OL, ignore_eos=True, logprobs=LP or None)
# in-process untimed warm-ups (upstream: --num-iters-warmup), on prompt sets the timed reps never use
for w in range(WARM):
    llm.generate(prompts_for(10_000 + SB + w), sp, use_tqdm=False)
# 2 s with the GPU idle before the timed reps: the out-of-process witness (harness/witness.py) takes the longest busy
# segment, so start-up and warm-up GPU work must not merge with the timed reps' segment
import torch
torch.cuda.synchronize(); _sleep(2.0)
reps, outs, steps = [], [], []
for r in range(REPS):
    P = prompts_for(SB + r)
    _trace.clear(); _on[0] = True
    t = perf_counter(); res = llm.generate(P, sp, use_tqdm=False); dt = perf_counter() - t
    _on[0] = False
    tr = list(_trace)
    dec = [x for x in tr if x[0] == 0]
    rec = {"rep": r, "wall_s": dt, "num_requests": len(res), "steps": len(tr), "decode_steps": len(dec),
           "graph_decode_steps": sum(1 for x in dec if x[3]), "prefill_steps": len(tr) - len(dec), "t_end_unix": _now()}
    reps.append(rec); print("[P2REP]", json.dumps(rec), flush=True)
    steps.append([list(x) for x in tr])
    po = []
    for o in sorted(res, key=lambda o: int(o.request_id)):
        c = o.outputs[0]
        d = {"ids": list(c.token_ids), "text_sha256": hashlib.sha256(c.text.encode()).hexdigest(), "finish": c.finish_reason}
        if LP:
            d["top"] = [{str(t): float(v.logprob) for t, v in s.items()} for s in (c.logprobs or [])]
        po.append(d)
    outs.append(po)
key = {"workload": "latency", "bs": BS, "in": IN, "ol": OL, "reps": REPS, "seed_base": SB}
json.dump(dict(key, passes=outs), open(os.path.join(OUT, "outputs.json"), "w"))
json.dump(dict(key, steps=steps, hooks=hooks), open(os.path.join(OUT, "steps.json"), "w"))
json.dump(dict(key, reps=reps, hooks=hooks, process_wall_s=_now() - T0), open(os.path.join(OUT, "passes.json"), "w"), indent=1)
