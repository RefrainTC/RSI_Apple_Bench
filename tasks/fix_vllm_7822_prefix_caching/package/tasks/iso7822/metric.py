import json, os
def metric(out):
    p = os.path.join(out, "passes.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p)); ps = d["passes"] if isinstance(d, dict) else d
    if not ps:
        return None
    return {"primary": ps[0]["wall_s"], "warmup_computed_prefill_tokens": ps[0].get("computed_prefill_tokens"),
            "warmup_prefill_tokens": ps[0].get("prefill_tokens"), "decode_tokens": [x.get("decode_tokens") for x in ps],
            "second_pass_s": ps[1]["wall_s"] if len(ps) > 1 else None,
            # the out-of-process witness compares the GPU-busy window with both passes (review F1)
            "witness_reported_s": sum(x["wall_s"] for x in ps[:2]),
            "hooks": ps[0].get("hooks"), "process_wall_s": d.get("process_wall_s") if isinstance(d, dict) else None,
            "workload": d.get("workload") if isinstance(d, dict) else None}
