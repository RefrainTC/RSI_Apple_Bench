"""settings.py : #14930 effective settings. Two sources:
  settings.json  : the frontend's vllm_config after LLM() (runner.py; spec k, n-gram range, eager, prefix caching, ...)
  job_stdout.log : what the engine-core process itself logged (captured by p2_job.sh, outside the candidate's control
                   except for what the engine code prints): its 'Initializing a V1 LLM engine ... with config' line
                   (spec config, enforce_eager, prefix caching, compilation level, CUDA graph use), whether a torch.compile
                   and a CUDA-graph capture happened, and the attention backend.
(LLM.collective_rpc cannot reach the V1 worker in this vLLM version: 'LLMEngine' has no attribute 'model_executor'.)
The harness requires the candidate's to equal the parent's (except SETTINGS_IGNORE)."""
import json, os, re
def settings(out):
    s = {}
    try: s.update(json.load(open(os.path.join(out, "settings.json"))))
    except Exception as e: s["settings_json_error"] = repr(e)
    log = ""
    try: log = open(os.path.join(out, "job_stdout.log"), errors="replace").read()
    except Exception: pass
    m = re.search(r"Initializing a V1 LLM engine .*? with config: (.*)", log)
    cfg = m.group(1) if m else ""
    def g(rx):
        x = re.search(rx, cfg); return x.group(1) if x else None
    s["core.spec"] = g(r"speculative_config=(SpeculativeConfig\([^)]*\))")
    s["core.enforce_eager"] = g(r"enforce_eager=(\w+)")
    s["core.prefix_caching"] = g(r"enable_prefix_caching=(\w+)")
    s["core.chunked_prefill"] = g(r"chunked_prefill_enabled=(\w+)")
    s["core.compilation_level"] = g(r'compilation_config=\{"level":(\d+)')
    s["core.use_cudagraph"] = g(r'"use_cudagraph":(\w+)')
    s["core.dtype"] = g(r"dtype=([\w.]+)")
    s["core.graph_captured"] = "Graph capturing finished" in log
    s["core.torch_compiled"] = "torch.compile takes" in log
    s["core.attn_backend"] = (re.search(r"Using (\w+(?: \w+)?) backend on V1 engine", log) or [None, None])[1]
    s["_required_ok"] = bool(s["core.spec"] and "num_spec_tokens=" in (s["core.spec"] or "") and s["core.graph_captured"])
    return s
