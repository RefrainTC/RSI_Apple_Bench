"""settings.py : #13365 effective engine settings, read back from the live engine by runner.py (settings.json). The
harness requires the candidate's to equal the parent's (except SETTINGS_IGNORE). _required_ok: the PR's speculative
settings (n-gram, k = 5, n 3..4), V1 with the engine core in-process (reproducible batches), the live model runner using speculation with k = 5, n = 3, CUDA graphs on, and no proposal longer than k."""
import json, os
def settings(out):
    s = json.load(open(os.path.join(out, "settings.json")))
    s["_required_ok"] = (s.get("spec_num_speculative_tokens") == 5 and s.get("spec_ngram_min") == 3 and s.get("spec_ngram_max") == 4
                         and s.get("vllm_use_v1") is True and s.get("v1_multiprocessing") is False and s.get("use_cudagraph") is True
                         and s.get("engine_core_client") == "InprocClient" and s.get("live_use_spec_decode") is True
                         and s.get("live_k") == 5 and s.get("live_nmin") == 3 and s.get("live_use_cuda_graph") is True
                         and 0 < (s.get("max_proposal_seen") or 0) <= 5)
    return s
