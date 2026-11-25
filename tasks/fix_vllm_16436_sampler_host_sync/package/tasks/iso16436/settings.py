"""settings.py : #16436 effective engine settings from the server's own startup log (review F5): the engine config line
('Initializing a V1 LLM engine ... with config: ...'), whether torch.compile ran and CUDA graphs were captured. The
harness requires the candidate's to equal the parent's."""
import os, re
def settings(out):
    L = open(os.path.join(out, "server.log"), errors="replace").read()
    m = re.search(r"with config: (.*)", L)
    cfg = m.group(1).strip() if m else None
    return {"engine_config": cfg, "torch_compile": "torch.compile takes" in L, "cuda_graphs": "Graph capturing finished" in L,
            "kv_cache_tokens": (re.search(r"GPU KV cache size: ([\d,]+) tokens", L) or [None, None])[1]}
