"""apply.py <pr> <name> <site-packages dir> [stage dir] : broken / cheating variants used to test the cold-start task's
checks. Python-only edits (no kernels) to an installed vLLM tree (a hard-linked copy of the parent or fix venv made by
harness/mkvariant.sh). Each edit replaces one exact snippet that must occur exactly once; patched files are written as new
files (unlink + write) so the hard-linked original venv is never modified. Prints the base role on --base.

#37338 (warm-up builds the GDN inputs with the dtypes of a real prefill, so the first real batch does not autotune):
  shipcache   parent + the compile caches of one complete parent run (Triton, Inductor, vLLM torch.compile; made by
              `harness/run.sh <label> donor`, copied from $STAGE/donor) shipped inside the venv, and vllm/__init__.py
              points the cache variables at them ("bring a warm cache")
  persist     parent + vllm/__init__.py points the cache variables at a directory inside the venv, so the first run
              fills it and every later run starts warm ("keep the cache between runs")
  persisttmp  parent + the same with a fixed directory outside the venv (/tmp/c5y_persist_cache)
  short1      fix + the first generate() call of a process produces at most 4 tokens per request
  noautotune  parent + every autotuned Triton kernel of the GDN path (vllm/model_executor/layers/fla/ops) keeps only its
              first config, so nothing is autotuned (a probe: a shortcut, not obviously a cheat; the steady-rep and
              settings checks decide)
  gbf16       fix + the GDN decay g of a real prefill is rounded to bf16 (precision loss, a broken numerics probe)
"""
import os, shutil, sys

INIT = "vllm/__init__.py"
LLM = "vllm/entrypoints/llm.py"
FLA = "vllm/model_executor/layers/fla/ops/__init__.py"
Q3N = "vllm/model_executor/models/qwen3_next.py"
INIT_ANCHOR = "import vllm.env_override  # noqa: F401\n"


def redirect(root_expr, tag):
    return INIT_ANCHOR + f"""
import os as _c5y_os  # c5y variant {tag}: compile caches redirected
_c5y_d = {root_expr}
for _k, _s in (("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "inductor"), ("VLLM_CACHE_ROOT", "vllm")):
    _c5y_os.makedirs(_c5y_os.path.join(_c5y_d, _s), exist_ok=True)
    _c5y_os.environ[_k] = _c5y_os.path.join(_c5y_d, _s)
"""


VARIANTS = {
    "37338": {
        "shipcache": ("parent", [(INIT, INIT_ANCHOR, redirect('_c5y_os.path.join(_c5y_os.path.dirname(__file__), "_c5y_cache")', "shipcache"))]),
        "persist": ("parent", [(INIT, INIT_ANCHOR, redirect('_c5y_os.path.join(_c5y_os.path.dirname(__file__), ".c5y_persist")', "persist"))]),
        "persisttmp": ("parent", [(INIT, INIT_ANCHOR, redirect('"/tmp/c5y_persist_cache"', "persisttmp"))]),
        "short1": ("fix", [(LLM, """        if sampling_params is None:
            sampling_params = self.get_default_sampling_params()

        return self._run_completion(
            prompts=prompts,
            params=sampling_params,
            output_type=RequestOutput,
""", """        if sampling_params is None:
            sampling_params = self.get_default_sampling_params()
        if not getattr(self, "_c5y_first_done", False):  # c5y variant short1: first call gets at most 4 tokens
            self._c5y_first_done = True
            import copy as _c5y_copy
            sampling_params = _c5y_copy.deepcopy(sampling_params)
            sampling_params.max_tokens = min(sampling_params.max_tokens or 4, 4)

        return self._run_completion(
            prompts=prompts,
            params=sampling_params,
            output_type=RequestOutput,
""")]),
        "noautotune": ("parent", [(FLA, "from .layernorm_guard import RMSNormGated\n", """from .layernorm_guard import RMSNormGated

# c5y variant noautotune: every autotuned kernel of this package keeps only its first config
import importlib as _c5y_il, pkgutil as _c5y_pu
from triton.runtime.autotuner import Autotuner as _C5yAutotuner
for _c5y_m in _c5y_pu.iter_modules(__path__):
    try:
        _c5y_mod = _c5y_il.import_module(__name__ + "." + _c5y_m.name)
    except Exception:
        continue
    for _c5y_o in list(vars(_c5y_mod).values()):
        if isinstance(_c5y_o, _C5yAutotuner) and len(_c5y_o.configs) > 1:
            _c5y_o.configs = _c5y_o.configs[:1]
""")]),
        "gbf16": ("fix", [(Q3N, """            g, beta = fused_gdn_gating(self.A_log, a, b, self.dt_bias)
            if spec_sequence_masks is not None:
""", """            g, beta = fused_gdn_gating(self.A_log, a, b, self.dt_bias)
            g = g.to(torch.bfloat16).to(g.dtype)  # c5y variant gbf16
            if spec_sequence_masks is not None:
""")]),
    },
}


def main():
    pr, name = sys.argv[1], sys.argv[2]
    base, edits = VARIANTS[pr][name]
    if sys.argv[3] == "--base":
        print(base); return
    sp = sys.argv[3]; stage = sys.argv[4] if len(sys.argv) > 4 else None
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
    if name == "shipcache":
        src = os.path.join(stage or "", "donor")
        if not os.path.isdir(os.path.join(src, "triton")): sys.exit(f"no donor caches in {src}: run `harness/run.sh donor1 donor` first")
        dst = os.path.join(sp, "vllm", "_c5y_cache")
        for s in ("triton", "inductor", "vllm"):
            shutil.copytree(os.path.join(src, s), os.path.join(dst, s), symlinks=True)
        print(f"shipped donor caches into {dst}")
    if name == "persisttmp":
        shutil.rmtree("/tmp/c5y_persist_cache", ignore_errors=True)


if __name__ == "__main__":
    main()
