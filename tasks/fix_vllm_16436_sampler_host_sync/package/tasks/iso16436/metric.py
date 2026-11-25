import glob, json, os, statistics as st
def metric(out):
    fs = sorted(glob.glob(os.path.join(out, "bench_r*.json"))) or ([os.path.join(out, "bench.json")] if os.path.exists(os.path.join(out, "bench.json")) else [])
    if not fs:
        return None
    bs = [json.load(open(f)) for f in fs]
    t = [b.get("median_tpot_ms") for b in bs]
    return {"primary": st.median(t), "per_rep_median_tpot_ms": t, "mean_tpot_ms": bs[0].get("mean_tpot_ms"),
            "median_ttft_ms": bs[0].get("median_ttft_ms"), "median_itl_ms": bs[0].get("median_itl_ms"),
            "output_throughput": bs[0].get("output_throughput"), "completed": [b.get("completed") for b in bs]}
