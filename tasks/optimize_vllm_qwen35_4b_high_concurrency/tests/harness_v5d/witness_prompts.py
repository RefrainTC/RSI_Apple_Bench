"""witness_prompts.py : rebuild the prompts of a `vllm bench serve` run (random dataset) from its arguments.

stdin : {"jobs": [{"argv": [<bench serve args>], "indices": [i, ...]}, ...]}
stdout: {"jobs": [{"n": <requests>, "prompts": [...], "prompt_lens": [...]}, ...]}

Same code path as the stock client (vllm/benchmarks/serve.py main_async): parse with add_cli_args, seed `random`
and numpy with --seed, load the tokenizer the same way, call get_samples. The request order of the result file
(generated_texts, input_lens) is the order of these samples. The scorer checks prompt_lens against the result's
input_lens, so a mismatch voids the witness instead of scoring wrong pairs. Run with the trusted venv, -I.
"""
import json
import os
import random
import sys

# stdout carries only the JSON: vLLM's logger and get_samples print to stdout, so fd 1 goes to stderr
# before anything is imported, and the JSON is written to a saved copy of the original fd 1
_OUT_FD = os.dup(1)
os.dup2(2, 1)

import numpy as np  # noqa: E402
from vllm.benchmarks.datasets import get_samples  # noqa: E402
from vllm.benchmarks.serve import add_cli_args  # noqa: E402
from vllm.tokenizers import get_tokenizer  # noqa: E402
from vllm.utils.argparse_utils import FlexibleArgumentParser  # noqa: E402

spec = json.load(sys.stdin)
tok_cache = {}
out = []
for job in spec["jobs"]:
    p = FlexibleArgumentParser()
    add_cli_args(p)
    args = p.parse_args(job["argv"])
    tid = args.tokenizer if args.tokenizer is not None else args.model
    key = (tid, args.tokenizer_mode)
    if key not in tok_cache:
        tok_cache[key] = get_tokenizer(tid, tokenizer_mode=args.tokenizer_mode, trust_remote_code=args.trust_remote_code)
    random.seed(args.seed)
    np.random.seed(args.seed)
    reqs = get_samples(args, tok_cache[key])
    idx = [i for i in job["indices"] if 0 <= i < len(reqs)]
    out.append({"n": len(reqs), "prompts": [reqs[i].prompt for i in idx], "prompt_lens": [reqs[i].prompt_len for i in idx],
                "indices": idx})
with os.fdopen(_OUT_FD, "w") as f:
    json.dump({"jobs": out}, f)
