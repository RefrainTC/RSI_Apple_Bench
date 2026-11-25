"""settings.py : #7822 effective engine settings, as read back from the live engine by runner.py (settings.json).
The harness requires the candidate's to equal the parent's (except SETTINGS_IGNORE) and enforce_eager with zero
captured CUDA graphs (review F5)."""
import json, os
def settings(out):
    s = json.load(open(os.path.join(out, "settings.json")))
    s["_required_ok"] = s.get("enforce_eager") is True and s.get("captured_cuda_graphs") == 0
    return s
