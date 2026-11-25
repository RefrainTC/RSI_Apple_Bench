"""set_ref.py [--dry-run] <label> <label> ... : set REF and band in harness/gate.json from finished jobs of the human
fix on this machine (calibration jobs and scoring jobs whose candidate is the fix; each must have passed every check).
  REF  = median of the jobs' speedups;   band = max(0.5 %, 2 SD of the job speedups / REF).
Also writes gpu_name (nvidia-smi name of the GPU the jobs ran on) and ref_date, and keeps the shipped values under
reference_rtx4090 (first run only). Other keys are kept, so harness/check.py reads the file as before. Records the new
file's sha256 in kit_state/gate.sha256 (test.sh checks gate.json against it). Stdlib only; run with python3 -I.
ISO_ROOT: environment, else the directory above kit/."""
import datetime, hashlib, json, os, socket, statistics, subprocess, sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
dry = "--dry-run" in sys.argv
root = os.environ.get("ISO_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
gp = os.path.join(root, "harness", "gate.json")
if not args: sys.exit(__doc__)
die = lambda m: sys.exit(f"SET_REF REFUSED: {m}")
host = socket.gethostname(); jobs = []; aa = []; gpus = set()
for lab in args:
    R = os.path.join(root, "results", lab)
    try: V = json.load(open(os.path.join(R, "verdict.json"))); M = json.load(open(os.path.join(R, "run_meta.json")))
    except Exception as e: die(f"{lab}: no finished job ({e}); run harness/run.sh {lab} cal|score first")
    if V.get("mode") not in ("cal", "score") or (V.get("mode") == "score" and V.get("cand") != "fix"):
        die(f"{lab}: mode {V.get('mode')} cand {V.get('cand')}; only calibration jobs and scoring jobs of the fix count")
    if V.get("fails"): die(f"{lab}: the human fix failed checks {V['fails']}; the gate would be wrong")
    if V.get("speedup") is None: die(f"{lab}: no speedup (incomplete job)")
    if M.get("host") != host: die(f"{lab}: ran on {M.get('host')}, this is {host}")
    name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", str(M.get("gpu", 0))],
                          capture_output=True, text=True).stdout.strip()
    if not name: die("nvidia-smi gave no GPU name")
    gpus.add(name); jobs.append((lab, V["mode"], float(V["speedup"])))
    if V.get("aa_max_dev") is not None: aa.append(V["aa_max_dev"])
if len(gpus) != 1: die(f"jobs ran on different GPU types {sorted(gpus)}")
if len(jobs) < 2: die("need at least 2 jobs of the fix (a calibration and a scoring job) to estimate the spread")
sp = [s for _, _, s in jobs]
ref = statistics.median(sp); sd = statistics.stdev(sp)
band = max(0.005, 2 * sd / ref)
G = json.load(open(gp))
if "reference_rtx4090" not in G and "gpu_name" not in G:
    G["reference_rtx4090"] = {k: G[k] for k in ("ref", "band", "note") if k in G}
G.update({"ref": round(ref, 4), "band": round(band, 4), "gpu_name": gpus.pop(),
          "ref_date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
          "ref_jobs": {lab: round(s, 4) for lab, _, s in jobs}})
G["note"] = (f"Set {G['ref_date']} by kit/set_ref.py on {G['gpu_name']} from {len(jobs)} jobs of the human fix "
             f"({', '.join(f'{m} {s:.4f}' for _, m, s in jobs)}): REF = median, band = max(0.5 %, 2 SD = {200 * sd / ref:.2f} %)."
             + (f" Calibration A/A max deviation {max(aa):.4f}." if aa else "") + " Shipped values: reference_rtx4090.")
print(f"jobs: {jobs}")
print(f"REF {ref:.4f}  SD {sd:.4f} ({100 * sd / ref:.2f} %)  band {band:.4f}  gpu '{G['gpu_name']}'")
if aa and max(aa) > G.get("aa_band", 0.01):
    print(f"WARNING: calibration A/A max deviation {max(aa):.4f} exceeds aa_band {G.get('aa_band', 0.01)} (no_gain threshold); "
          "the machine is noisier than the gate assumes")
if dry: print(json.dumps(G, indent=1)); sys.exit(0)
tmp = gp + ".tmp"
with open(tmp, "w") as f: json.dump(G, f, indent=1); f.write("\n")
os.replace(tmp, gp)
ks = os.path.join(root, "kit_state"); os.makedirs(ks, exist_ok=True)
h = hashlib.sha256(open(gp, "rb").read()).hexdigest()
open(os.path.join(ks, "gate.sha256"), "w").write(f"{h}  harness/gate.json\n")
print(f"SET_REF_OK wrote {gp} (sha256 {h[:12]}, recorded in kit_state/gate.sha256)")
