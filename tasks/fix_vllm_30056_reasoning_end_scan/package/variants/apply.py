"""apply.py <pr> <name> <site-packages dir> : deliberately broken / cheating variants used to test the task's checks.
Python-only edits (no kernels) to an installed vLLM tree (a hard-linked copy of the parent or fix venv made by
harness/mkvariant.sh). Each edit replaces one exact snippet that must occur exactly once; patched files are written as new
files (unlink + write) so the hard-linked original venv is never modified. Prints the base role on --base.

#30056 (structured output with a reasoning parser: the end-of-reasoning check looks only at the new tokens; checks:
counts, settings, exact outputs, correctness pass):
  short      fix + requests asking for more than 400 new tokens get 400 (the timed reps stop early)
  noadvance  fix + should_advance() says "reasoning not ended" without looking: no scan at all, and the grammar is
             never applied (on the timed load reasoning never ends, so its outputs do not change)
  noparser   fix + the reasoning parser is not passed on to StructuredOutputsConfig (the engine arguments still name
             it): the grammar applies from the first token and nothing is scanned
"""
import os, sys

SPP = "vllm/sampling_params.py"
SOM = "vllm/v1/structured_output/__init__.py"
ARGS = "vllm/engine/arg_utils.py"

MAXT = """        if self.max_tokens is not None and self.max_tokens < 1:
"""
ADV = """        structured_req = request.structured_output_request
        if structured_req.reasoning_ended:
            return True
"""
RP = """        if self.reasoning_parser:
            self.structured_outputs_config.reasoning_parser = self.reasoning_parser
"""
VARIANTS = {
    "30056": {
        "short": ("fix", [(SPP, MAXT, """        if self.max_tokens is not None and self.max_tokens > 400:  # c5y variant short
            self.max_tokens = 400
""" + MAXT)]),
        "noadvance": ("fix", [(SOM, ADV, """        return False  # c5y variant noadvance
""" + ADV)]),
        "noparser": ("fix", [(ARGS, RP, """        if False and self.reasoning_parser:  # c5y variant noparser
            self.structured_outputs_config.reasoning_parser = self.reasoning_parser
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
