"""apply.py <pr> <name> <site-packages dir> : deliberately broken / cheating variants used to test the task's checks.
Python-only edits (no kernels) to an installed SGLang tree (a hard-linked copy of the parent or fix venv made by
harness/mkvariant.sh). Each edit replaces one exact snippet that must occur exactly once; patched files are written as new
files (unlink + write) so the hard-linked original venv is never modified. Prints the base role on --base.

#16521 (radix key with generated tokens; checks: teacher-forced gap, token counts, settings):
  kvperm    fix + the cached KV slots of the generated part are rotated by one position (key right, slot order wrong).
            A control: paged attention sums over the stored (already rotated) keys, so the slot order of a prefix does
            not change what later tokens see; outputs stay within bf16 noise and the checks should pass it.
  kvzero    fix + the cached values (V) of the generated part are zeroed when the request finishes: turn 2 reuses
            garbage KV under the right key ("cache the key, serve wrong KV")
  kvnoise100 / kvnoise10 / kvnoise1  fix + the cached values of the generated part get 100 % / 10 % / 1 % relative
            Gaussian noise (how large a KV corruption must be before the outputs, and so the checks, see it; bf16 rounding
            itself is ~0.4 %)
  drophist  parent + turn-2 prompts longer than 384 tokens lose their middle (the previous answer): only the first 256
            and the last 32 ids are prefilled ("skip the work" instead of reusing it)
  fp8kv     fix + KV cache stored as fp8_e5m2 (default of ServerArgs.kv_cache_dtype changed): lower precision
  short2    fix + prompts longer than 384 tokens get at most 4 new tokens (turn 2 stops early)
#20972 (no blocking copy with return_logprob; checks: exact digests, logprob comparison, settings):
  droplp    fix + decode-step top-k logprobs are neither copied nor returned
  stalelp   fix + each decode step returns the previous step's top-k logprobs (one step stale)
  fp16lp    fix + top-k logprobs taken from an fp16 copy of the logprobs (precision loss ~1e-3 nats)
"""
import os, sys

RC = "sglang/srt/mem_cache/radix_cache.py"
SCH = "sglang/srt/managers/scheduler.py"
SA = "sglang/srt/server_args.py"
OUT = "sglang/srt/managers/scheduler_output_processor_mixin.py"
LPU = "sglang/srt/layers/utils/logprob.py"

BOOT = """            if recv_req.bootstrap_port is None:
                # Use default bootstrap port
                recv_req.bootstrap_port = self.server_args.disaggregation_bootstrap_port
"""
DEC_APPEND = """                        req.output_top_logprobs_val.append(
                            logits_output.next_token_top_logprobs_val[flat_idx]
                        )
                        req.output_top_logprobs_idx.append(
                            logits_output.next_token_top_logprobs_idx[flat_idx]
                        )
"""
VAL_ANCHOR = """        values = kv_indices[: len(keys)].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        # Radix Cache takes one ref in memory pool
        if is_insert:
"""
KVCORR = """            _n0 = len(req.origin_input_ids)  # c5x variant {name}: cached values of the generated part corrupted
            if len(values) > _n0:
                _kv = self.token_to_kv_pool_allocator.get_kvcache()
                _g = torch.Generator(device=_kv.v_buffer[0].device); _g.manual_seed(len(values))
                for _vb in _kv.v_buffer:
                    _ix = values[_n0:].to(_vb.device)
                    {op}
"""
VARIANTS = {
    "16521": {
        "kvperm": ("fix", [(RC, """        values = kv_indices[: len(keys)].to(dtype=torch.int64, copy=True)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        # Radix Cache takes one ref in memory pool
        if is_insert:
""", """        values = kv_indices[: len(keys)].to(dtype=torch.int64, copy=True)
        _n0 = len(req.origin_input_ids)  # c5x variant kvperm: generated part's KV slots rotated by one
        if len(values) > _n0 + 1:
            values[_n0:] = torch.roll(values[_n0:], 1)
        radix_key = RadixKey(keys, req.extra_key, is_bigram=self.is_eagle)

        # Radix Cache takes one ref in memory pool
        if is_insert:
""")]),
        "kvzero": ("fix", [(RC, VAL_ANCHOR, VAL_ANCHOR + KVCORR.format(name="kvzero", op="_vb[_ix] = 0"))]),
        "kvnoise10": ("fix", [(RC, VAL_ANCHOR, VAL_ANCHOR + KVCORR.format(name="kvnoise10", op="_vb[_ix] = (_vb[_ix].float() * (1 + 0.10 * torch.randn(_vb[_ix].shape, device=_vb.device, generator=_g))).to(_vb.dtype)"))]),
        "kvnoise100": ("fix", [(RC, VAL_ANCHOR, VAL_ANCHOR + KVCORR.format(name="kvnoise100", op="_vb[_ix] = (_vb[_ix].float() * (1 + 1.00 * torch.randn(_vb[_ix].shape, device=_vb.device, generator=_g))).to(_vb.dtype)"))]),
        "kvnoise1": ("fix", [(RC, VAL_ANCHOR, VAL_ANCHOR + KVCORR.format(name="kvnoise1", op="_vb[_ix] = (_vb[_ix].float() * (1 + 0.01 * torch.randn(_vb[_ix].shape, device=_vb.device, generator=_g))).to(_vb.dtype)"))]),
        "drophist": ("parent", [(SCH, BOOT, """            if recv_req.input_ids is not None and len(recv_req.input_ids) > 384:  # c5x variant drophist
                recv_req.input_ids = list(recv_req.input_ids[:256]) + list(recv_req.input_ids[-32:])
""" + BOOT)]),
        "fp8kv": ("fix", [(SA, '    kv_cache_dtype: str = "auto"\n', '    kv_cache_dtype: str = "fp8_e5m2"  # c5x variant fp8kv\n')]),
        "short2": ("fix", [(SCH, BOOT, """            if recv_req.input_ids is not None and len(recv_req.input_ids) > 384:  # c5x variant short2
                recv_req.sampling_params.max_new_tokens = min(recv_req.sampling_params.max_new_tokens, 4)
""" + BOOT)]),
    },
    "20972": {
        "droplp": ("fix", [
            (OUT, """                next_token_logprobs = logits_output.next_token_logprobs.tolist()
                if logits_output.next_token_top_logprobs_val:
""", """                next_token_logprobs = logits_output.next_token_logprobs.tolist()
                if False:  # c5x variant droplp
"""),
            (OUT, """                    if req.top_logprobs_num > 0:
                        flat_idx = i * max_accept + j
""" + DEC_APPEND, """                    if False and req.top_logprobs_num > 0:  # c5x variant droplp
                        flat_idx = i * max_accept + j
""" + DEC_APPEND)]),
        "stalelp": ("fix", [(OUT, DEC_APPEND, """                        _cur = (logits_output.next_token_top_logprobs_val[flat_idx], logits_output.next_token_top_logprobs_idx[flat_idx])
                        _m = self.__dict__.setdefault("_c5x_prev", {})  # c5x variant stalelp: previous step's values
                        _prev = _m.get(req.rid) or _cur
                        _m[req.rid] = _cur
                        req.output_top_logprobs_val.append(_prev[0])
                        req.output_top_logprobs_idx.append(_prev[1])
""")]),
        "fp16lp": ("fix", [(LPU, "    values, indices = logprobs.topk(max_k, dim=-1)\n",
                            "    values, indices = logprobs.half().float().topk(max_k, dim=-1)  # c5x variant fp16lp\n")]),
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
