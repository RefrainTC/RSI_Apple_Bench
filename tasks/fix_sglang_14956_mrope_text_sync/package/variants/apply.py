"""apply.py <pr> <name> <site-packages dir> : deliberately broken / cheating variants used to test the task's checks.
Python-only edits (no kernels) to an installed SGLang tree (a hard-linked copy of the parent or fix venv made by
harness/mkvariant.sh). Each edit replaces one exact snippet that must occur exactly once; patched files are written as new
files (unlink + write) so the hard-linked original venv is never modified. Prints the base role on --base.

#14956 (mrope decode positions of text requests read from seq_lens_cpu; checks: exact outputs, correctness pass,
counts, settings):
  short      fix + requests asking for more than 200 new tokens get 200 (the timed reps stop early)
  offby1     fix + the decode position of a text request is seq_lens_cpu[i] instead of seq_lens_cpu[i] - 1 (no sync,
             wrong position)
  multistep  fix + ServerArgs.num_continuous_decode_steps set to 4 inside __post_init__ (the engine arguments still say 1;
             fewer scheduler iterations per token)
"""
import os, sys

SCH = "sglang/srt/managers/scheduler.py"
FBI = "sglang/srt/model_executor/forward_batch_info.py"
SA = "sglang/srt/server_args.py"

BOOT = """            if recv_req.bootstrap_port is None:
                # Use default bootstrap port
                recv_req.bootstrap_port = self.server_args.disaggregation_bootstrap_port
"""
POS = """                        self.seq_lens_cpu[batch_idx] - 1,
"""
MISS = """        self._handle_missing_default_values()
"""
VARIANTS = {
    "14956": {
        "short": ("fix", [(SCH, BOOT, """            if recv_req.sampling_params.max_new_tokens is not None and recv_req.sampling_params.max_new_tokens > 200:  # c5y variant short
                recv_req.sampling_params.max_new_tokens = 200
""" + BOOT)]),
        "offby1": ("fix", [(FBI, POS, """                        self.seq_lens_cpu[batch_idx],  # c5y variant offby1 (was - 1)
""")]),
        "multistep": ("fix", [(SA, MISS, MISS + """        self.num_continuous_decode_steps = 4  # c5y variant multistep
""")]),
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
