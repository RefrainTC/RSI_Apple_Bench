"""Boot the pristine server once and write scoring/corr_ref.json (+ self-check)."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3lib, correctness
p3lib.install_signal_handlers()  # v5d: SIGINT / SIGTERM stop every live server, exit 130 / 143
tdir = Path(sys.argv[1]); n_tok = int(sys.argv[2]) if len(sys.argv) > 2 else 64
task = p3lib.load_task(tdir)
out = tdir / "results" / "make_ref"; out.mkdir(parents=True, exist_ok=True)
ov = p3lib.build_overlay(tdir / "pristine", out / "overlay")
p3lib.wait_gpu_free(task["gpus"], 1500, 600)
srv = p3lib.Server(task, tdir / "pristine", ov, out / "server.log", out / "cache", "ref")
t0 = time.time(); srv.start()
try:
    assert srv.wait_ready(1200), f"not ready killed={srv.killed_reason}"
    print(f"ready {srv.ready_s:.0f}s", flush=True)
    prompts = json.loads((tdir / "scoring" / "corr_prompts.json").read_text())
    ref = correctness.make_ref(srv.base_url, task["served_name"], prompts, n_tok)
    (tdir / "scoring" / "corr_ref.json").write_text(json.dumps(ref))
    c = correctness.check(srv.base_url, task["served_name"], ref)
    print("self-check", json.dumps({k: v for k, v in c.items() if k != "per_prompt"}), flush=True)
    print("counters", json.dumps(p3lib.scrape_metrics(srv.base_url))[:2000])
finally:
    srv.stop(); print("peak", srv.peak_mib, "wall", round(time.time() - t0))
