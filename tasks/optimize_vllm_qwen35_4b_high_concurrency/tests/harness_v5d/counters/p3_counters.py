"""P3 counter module (harness-only, used in counter runs, never in scoring runs).

When P3_COUNTERS_DIR is set, wraps vllm.v1.core.sched.scheduler.Scheduler at import time and records
  per engine step : scheduled tokens (histogram), prefill chunks (requests scheduled > 8 tokens in a step),
                    decode entries, spec tokens, steps with prefill;
  step timing     : wall time between consecutive schedule() calls, attributed to the previous step, for
                    pure-decode steps bucketed by batch size (16-wide): [n, sum_s, sum_sq] (steady state: the
                    interval between schedule calls is the engine step period);
  KV / state spec : the engine's own kv_cache_config (per group: spec class, block size, page bytes, layers;
                    for Mamba/GDN groups the state shapes and dtypes), written once to kvspec_<pid>.json.
Dumps to $P3_COUNTERS_DIR/sched_<pid>.json every 2 s. Loaded by audit/sitecustomize.py (or
counters/sitecustomize.py when the audit hook is off).
"""
import os
import sys

_OUT = os.environ.get("P3_COUNTERS_DIR")
if _OUT:
    import importlib.abc
    import importlib.util
    import json
    import threading
    import time

    _TARGET = "vllm.v1.core.sched.scheduler"
    _stats = {"steps": 0, "tokens": 0, "max_step_tokens": 0, "hist_pow2": {}, "prefill_chunks": 0,
              "prefill_tokens": 0, "decode_entries": 0, "spec_tokens": 0, "steps_with_prefill": 0,
              "decode_step_s": {}}
    _last = {"t": None, "kind": None}

    def _kvspec(cfg):
        out = {"num_blocks": getattr(cfg, "num_blocks", None), "groups": []}
        for g in getattr(cfg, "kv_cache_groups", []) or []:
            sp = g.kv_cache_spec
            row = {"spec": type(sp).__name__, "n_layers": len(g.layer_names),
                   "block_size": getattr(sp, "block_size", None), "page_size_bytes": getattr(sp, "page_size_bytes", None)}
            for k in ("shapes", "dtypes", "num_kv_heads", "head_size", "dtype", "mamba_type"):
                if hasattr(sp, k):
                    row[k] = repr(getattr(sp, k))
            out["groups"].append(row)
        out["tensors"] = [{"size": t.size, "shared_by": len(t.shared_by)} for t in getattr(cfg, "kv_cache_tensors", []) or []][:64]
        return out

    def _dump():
        os.makedirs(_OUT, exist_ok=True)
        p = os.path.join(_OUT, f"sched_{os.getpid()}.json")
        while True:
            time.sleep(2.0)
            try:
                with open(p + ".tmp", "w") as f:
                    json.dump(_stats, f)
                os.replace(p + ".tmp", p)
            except Exception:
                pass

    def _patch(mod):
        S = mod.Scheduler
        orig = S.schedule
        orig_init = S.__init__

        def __init__(self, *a, **k):
            orig_init(self, *a, **k)
            try:
                os.makedirs(_OUT, exist_ok=True)
                with open(os.path.join(_OUT, f"kvspec_{os.getpid()}.json"), "w") as f:
                    json.dump(_kvspec(self.kv_cache_config), f, indent=1, default=str)
            except Exception as e:  # never break the engine
                try:
                    with open(os.path.join(_OUT, f"kvspec_{os.getpid()}.err"), "w") as f:
                        f.write(repr(e))
                except Exception:
                    pass

        S.__init__ = __init__

        def schedule(self, *a, **k):
            t = time.perf_counter()
            if _last["t"] is not None and _last["kind"] is not None and t - _last["t"] < 0.5:  # skip idle gaps
                b = _stats["decode_step_s"].setdefault(str(_last["kind"]), [0, 0.0, 0.0])
                dt = t - _last["t"]
                b[0] += 1
                b[1] += dt
                b[2] += dt * dt
            _last["t"], _last["kind"] = t, None
            out = orig(self, *a, **k)
            try:
                n = int(out.total_num_scheduled_tokens)
                if n:
                    st = _stats
                    st["steps"] += 1
                    st["tokens"] += n
                    st["max_step_tokens"] = max(st["max_step_tokens"], n)
                    b = str(1 << max(0, n - 1).bit_length())
                    st["hist_pow2"][b] = st["hist_pow2"].get(b, 0) + 1
                    pre = 0
                    for _rid, t in out.num_scheduled_tokens.items():
                        if t > 8:
                            st["prefill_chunks"] += 1
                            st["prefill_tokens"] += t
                            pre = 1
                        else:
                            st["decode_entries"] += 1
                    st["steps_with_prefill"] += pre
                    if not pre:  # pure-decode step: its period is measured at the next schedule() call
                        _last["kind"] = 16 * ((len(out.num_scheduled_tokens) + 15) // 16)
                    st["spec_tokens"] += sum(len(v) for v in (out.scheduled_spec_decode_tokens or {}).values())
            except Exception:
                pass
            return out

        S.schedule = schedule
        threading.Thread(target=_dump, daemon=True).start()

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _TARGET:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            loader = spec.loader
            orig_exec = loader.exec_module

            def exec_module(module):
                orig_exec(module)
                _patch(module)

            loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
