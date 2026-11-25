"""settings.py : #12380 effective engine settings, read back from the live engine by runner.py (settings.json). The harness
requires the candidate's to equal the parent's (except SETTINGS_IGNORE). Required in every run: CUDA graphs on
(enforce_eager False) with graphs actually captured, one scheduler step per engine step, no chunked prefill, no
speculative decoding, no torch.compile (V0 default)."""
import json, os
def settings(out):
    s = json.load(open(os.path.join(out, "settings.json")))
    s["_required_ok"] = (s.get("enforce_eager") is False and isinstance(s.get("captured_cuda_graphs"), int)
                         and s["captured_cuda_graphs"] > 0 and s.get("num_scheduler_steps") == 1
                         and s.get("chunked_prefill") is False and s.get("speculative") == "None"
                         and s.get("compilation_level") == 0)
    return s
