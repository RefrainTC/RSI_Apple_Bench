"""ccdload.py [secs=5] : per-CCD CPU utilisation (%) over a short window, plus GPU utilisation; one line (host context)."""
import subprocess, sys, time
def rd():
    d = {}
    for l in open("/proc/stat"):
        if l.startswith("cpu") and l[3].isdigit():
            p = l.split(); v = list(map(int, p[1:])); d[int(p[0][3:])] = (sum(v) - v[3] - v[4], sum(v))
    return d
a = rd(); time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 5); b = rd()
u = {c: 100 * (b[c][0] - a[c][0]) / max(1, b[c][1] - a[c][1]) for c in a}
ccd = {f"CCD{i}": [c for c in u if (c % 32) // 8 == i] for i in range(4)}
g = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout.replace("\n", ";")
vm = {l.split()[0]: l.split()[1] for l in open("/proc/vmstat") if l.split()[0] in ("pgscan_direct", "allocstall_movable", "compact_stall")}
print(time.strftime("%H:%M:%S"), " ".join(f"{k}={sum(u[c] for c in v)/len(v):.0f}%" for k, v in ccd.items()), "| gpu", g, "| vmstat", " ".join(f"{k}={v}" for k, v in vm.items()), "| load", open("/proc/loadavg").read().split()[0])
