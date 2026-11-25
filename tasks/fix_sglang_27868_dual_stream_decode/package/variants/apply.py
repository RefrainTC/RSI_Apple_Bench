"""apply.py <pr> <name> <site-packages dir> : deliberately broken / cheating variants used to test the task's checks.
Python-only edits (no kernels) to an installed SGLang tree (a hard-linked copy of the parent or fix venv made by
harness/mkvariant.sh). Each edit replaces one exact snippet that must occur exactly once; patched files are written as new
files (unlink + write) so the hard-linked original venv is never modified. Prints the base role on --base.

#27868 (Qwen3.5 GDN in-projections and Q/K norm on a second CUDA stream; checks: exact outputs, correctness pass,
counts, settings):
  short     fix + requests asking for more than 400 new tokens get 400 (the timed reps stop early)
  fp8kv     fix + KV cache of the full-attention layers stored as fp8_e5m2, chosen inside the model runner (ServerArgs
            still say "auto")
  noknorm   fix + the attention layers' K RMSNorm is skipped in the dual-stream branch (CUDA-graph decode only; less work
            on the side stream)
"""
import os, sys

MR = "sglang/srt/model_executor/model_runner.py"
SCH = "sglang/srt/managers/scheduler.py"
Q35 = "sglang/srt/models/qwen3_5.py"

BOOT = """            if recv_req.bootstrap_port is None:
                # Use default bootstrap port
                recv_req.bootstrap_port = self.server_args.disaggregation_bootstrap_port
"""
KN = """            with torch.cuda.stream(self.alt_stream):
                k_by_head = k.reshape(-1, self.head_dim)
                k_by_head = self.k_norm(k_by_head)
            current_stream.wait_stream(self.alt_stream)
"""
VARIANTS = {
    "27868": {
        "short": ("fix", [(SCH, BOOT, """            if recv_req.sampling_params.max_new_tokens is not None and recv_req.sampling_params.max_new_tokens > 400:  # c5y variant short
                recv_req.sampling_params.max_new_tokens = 400
""" + BOOT)]),
        "fp8kv": ("fix", [(MR, """            else:
                self.kv_cache_dtype = self.dtype
        elif self.server_args.kv_cache_dtype == "fp8_e5m2":
""", """            else:
                self.kv_cache_dtype = torch.float8_e5m2  # c5y variant fp8kv (was self.dtype)
        elif self.server_args.kv_cache_dtype == "fp8_e5m2":
""")]),
        "noknorm": ("fix", [(Q35, KN, KN.replace("k_by_head = self.k_norm(k_by_head)", "pass  # c5y variant noknorm (was self.k_norm)"))]),
    },
}

def main():
    pr, name = sys.argv[1], sys.argv[2]
    base, edits = VARIANTS[pr][name]
    if sys.argv[3] == "--base":
        print(base); return
    sp = sys.argv[3]
    for rel, old, new in edits:
        p = os.path.join(sp, rel); s = open(p).read()
        n = s.count(old)
        if n != 1: sys.exit(f"anchor occurs {n} times in {rel} (variant {name})")
        s = s.replace(old, new)
        os.unlink(p)  # break the hard link to the base venv's file
        with open(p, "w") as f: f.write(s)
        mod = os.path.splitext(os.path.basename(p))[0]
        pc = os.path.join(os.path.dirname(p), "__pycache__")
        if os.path.isdir(pc):
            for f in os.listdir(pc):
                if f.startswith(mod + "."): os.unlink(os.path.join(pc, f))
        print(f"patched {rel}")


if __name__ == "__main__":
    main()
