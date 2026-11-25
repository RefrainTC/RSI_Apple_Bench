"""make_harness.py <apple tests dir> <out dir> : copy Apple's tests/ and apply the local adaptations
(exact-string edits, each asserted to match once). Every edit applies to BOTH arms (same runner for both trees).
  1. workloads.json: tp_size 4 -> 1 on every point (1x RTX 4090; task: 4xB200, gpt-oss-120b).
  2. _candidate_runner.py: append $RSI_EXTRA_SERVE_ARGS to the server argv (hardware-forced flags, same for both arms).
  3. _candidate_runner.py: _gpu_used_mb() reads only the GPU this run uses (nvidia-smi --id); the original sums every
     GPU on the box, so the post-teardown "wait until < 4000 MiB" would always time out (5 min) while other GPUs work.
     Teardown only, outside every timed window.
"""
import json, shutil, sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
if out.exists():
    shutil.rmtree(out)
shutil.copytree(src, out)

def edit(rel, old, new):
    p = out / rel
    s = p.read_text()
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"{rel}: anchor found {n} times, want 1: {old[:60]!r}")
    p.write_text(s.replace(old, new))
    print(f"[harness] edited {rel}")

wl = json.loads((out / "workloads.json").read_text())
n = 0
for w in wl["workloads"]:
    if w.get("tp_size") == 4:
        w["tp_size"] = 1; n += 1
wl["_local_adaptation"] = ("LOCAL ADAPTATION: tp_size 4 -> 1 on every point (gpt-oss-20b on 1x RTX 4090 instead of "
                           "gpt-oss-120b on 4xB200). Shapes, roles, guard bands unchanged.")
(out / "workloads.json").write_text(json.dumps(wl, indent=2) + "\n")
print(f"[harness] workloads.json: tp_size 4->1 on {n} points")
edit("_candidate_runner.py",
     '        *CONFIG["extra_server_args"],\n    ]\n',
     '        *CONFIG["extra_server_args"],\n    ] + os.environ.get("RSI_EXTRA_SERVE_ARGS", "").split()  # [local adaptation] same for both arms\n')
edit("_candidate_runner.py",
     '            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],\n            capture_output=True, text=True, timeout=60).stdout\n        return sum(',
     '            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]\n'
     '            # [local adaptation] only the GPU this run uses (teardown wait; other GPUs on the box are busy)\n'
     '            + ([f"--id={os.environ[\'CUDA_VISIBLE_DEVICES\']}"] if os.environ.get("CUDA_VISIBLE_DEVICES") else []),\n'
     '            capture_output=True, text=True, timeout=60).stdout\n        return sum(')
