"""make_harness_sc.py <harness_nl dir> <out dir> : the local harness_nl (edits 1-4) plus edit 5.
  5. _candidate_runner.py: both aiperf command lines (the anchor's and the other points') get --use-server-token-count,
     so output tokens come from the server's usage.completion_tokens (aiperf sets stream_options.include_usage itself)
     instead of re-tokenizing the streamed text on the client.
     Why: the requests sample with the model's generation_config (no temperature is sent; sglang's sampling_defaults is
     "model"), and the client re-tokenizes the sampled text. On ordinary text that count is within about +-1% of the
     tokens generated, but when Qwen2.5-1.5B drifts into repetitive text, BPE re-tokenization merges runs: one acceptance
     A/A request generated 2000 tokens (min_tokens = max_tokens = 2000, ignore_eos) and was counted as 1140, at
     an unchanged latency, so the c1 out2000 point read 0.928 and the A/A geomean 0.975 (RESULTS.md §4).
     With ignore_eos and min_tokens = max_tokens the server count is pinned at output_len, so tok/s = generated tokens / time.
     Same for both arms. Caveat: the count now comes from the server under test (see SAMPLE_CARD.md).
"""
import shutil
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
if out.exists():
    shutil.rmtree(out)
shutil.copytree(src, out, ignore=shutil.ignore_patterns("__pycache__"))
p = out / "_candidate_runner.py"
s = p.read_text()
old = '        "--streaming",\n'
assert s.count(old) == 2, f"anchor found {s.count(old)} times, want 2"
s = s.replace(old, old + '        "--use-server-token-count",  # [local adaptation, edit 5] server usage, not re-tokenized text\n')
p.write_text(s)
print(f"[harness_sc] {out}: --use-server-token-count on both aiperf lines")
