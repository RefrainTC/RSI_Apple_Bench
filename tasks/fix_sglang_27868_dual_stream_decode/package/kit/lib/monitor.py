"""monitor.py <results dir> <candidate venv> <scratch root> <out json> <ready file> <stop file> <tree>...
Runs beside harness/run.sh during a scoring job (started by test.sh). Stdlib only.
- Baseline: path, type, size, mtime, ctime of every file of the trusted trees (parent venv, staged interpreter, model,
  CUDA tree). Any write to a file changes its ctime, and ctime cannot be set back by a user process.
- After every run of the job finishes (run.sh logs 'end <tag>'): re-walks the trees and records any file added,
  removed or changed (so a candidate run that modifies the parent and a later candidate run that restores it are both
  caught, which a check at the end alone would miss); freezes the run directory (sha256 of every file); after a
  candidate run, looks for processes that the run started and that are still alive (by start time, plus the run's
  R4_TAG, its scratch dir or the candidate venv in their environment, cwd, exe or memory maps).
- On the stop file: one more pass, re-hashes every frozen run directory (a later run must not rewrite an earlier run's
  results, e.g. a parent run's latency.json) and writes the report."""
import hashlib, json, os, sys, time

R, CV, SCR, OUT, READY, STOP = sys.argv[1:7]
TREES = sys.argv[7:]
UID = os.getuid(); HZ = os.sysconf("SC_CLK_TCK")
BOOT = next(float(l.split()[1]) for l in open("/proc/stat") if l.startswith("btime"))


def walk(trees):
    s = {}
    for t in trees:
        for dp, dns, fns in os.walk(t):
            for n in fns + [d for d in dns if os.path.islink(os.path.join(dp, d))]:
                p = os.path.join(dp, n)
                try: st = os.lstat(p)
                except FileNotFoundError: continue
                s[p] = (st.st_mode, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    return s


def sha_dir(d):
    h = {}
    for dp, _, fns in os.walk(d):
        for n in fns:
            p = os.path.join(dp, n)
            try: h[os.path.relpath(p, d)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
            except OSError as e: h[os.path.relpath(p, d)] = f"unreadable:{e.errno}"
    return h


def diff(a, b):
    out = [("removed", p) for p in a if p not in b] + [("added", p) for p in b if p not in a]
    out += [("changed", p) for p in a if p in b and a[p] != b[p]]
    return out


def procs_started(t0, t1, tag, cand_tag_dir):
    """our processes started in [t0, t1] and still alive, tied to the run (tag, scratch dir or candidate venv)"""
    hits = []; me = os.getpid()
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == me: continue
        try:
            if os.stat(f"/proc/{pid}").st_uid != UID: continue
            st = open(f"/proc/{pid}/stat").read(); start = BOOT + int(st.rsplit(")", 1)[1].split()[19]) / HZ
            if not (t0 - 1 <= start <= t1 + 0.5): continue
            env = open(f"/proc/{pid}/environ", "rb").read()
            blob = env + b"\0" + open(f"/proc/{pid}/maps", "rb").read()
            for l in ("cwd", "exe"):
                try: blob += b"\0" + os.readlink(f"/proc/{pid}/{l}").encode()
                except OSError: pass
            if (f"R4_TAG={tag}".encode() in env.split(b"\0") or CV.encode() in blob or cand_tag_dir.encode() in blob):
                hits.append({"pid": int(pid), "start": round(start, 2),
                             "cmd": open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ")[:160].decode("utf-8", "replace")})
        except (OSError, IndexError, ValueError):
            continue
    return hits


base = walk(TREES)
rep = {"trees": TREES, "baseline_files": len(base), "events": [], "violations": [], "frozen": {}}
open(READY, "w").write(f"{len(base)}\n")
seen = set(); frozen = {}


def check_trees(after):
    d = diff(base, walk(TREES))
    if d:
        rep["violations"].append({"check": "trusted_trees", "after": after, "n": len(d), "first": d[:10]})
    return len(d)


def ended_tags():
    """tags whose 'end <tag> rc=' line run.sh has logged: written after all of the run's files (job.sh's, and the
    harness's own post-run files such as cold_witness.json / venv_check.json)"""
    try: L = open(os.path.join(R, "run.log")).read().split("\n")
    except OSError: return []
    return [x.split()[3] for x in L if len(x.split()) > 4 and x.split()[2] == "end" and x.split()[4].startswith("rc=")]


def poll():
    if not os.path.isdir(R): return
    for n in ended_tags():
        d = os.path.join(R, n)
        if n in seen or "." in n or not os.path.isdir(d): continue
        mf = os.path.join(d, "job_meta.json")
        try: m = json.load(open(mf))
        except Exception: m = {"start": 0, "end": 0, "tag": "", "rc": None, "aborted": "no job_meta.json"}
        seen.add(n); frozen[n] = sha_dir(d)
        cand = not (n.endswith("_parent") or n == "tf")
        ev = {"run": n, "role": "cand" if cand else "parent", "rc": m.get("rc"), "aborted": m.get("aborted"),
              "t_detect": round(time.time() - float(m.get("end", time.time())), 2)}
        ev["tree_changes"] = check_trees(n)
        if cand:
            left = procs_started(float(m["start"]), float(m["end"]), m.get("tag", ""), os.path.join(SCR, "s_" + m.get("tag", "")))
            ev["leftover_procs"] = len(left)
            if left: rep["violations"].append({"check": "leftover_process", "after": n, "procs": left[:5]})
        rep["events"].append(ev)


def safe_poll():
    try: poll()
    except Exception as e:   # never pass silently: an error in the monitor is a failed check
        rep["violations"].append({"check": "monitor_error", "error": repr(e)[:300]})


while not os.path.exists(STOP):
    safe_poll(); time.sleep(0.5)
safe_poll()
rep["final_tree_changes"] = check_trees("final")
for n, h in frozen.items():
    h2 = sha_dir(os.path.join(R, n))
    if h2 != h:
        ch = sorted(k for k in set(h) | set(h2) if h.get(k) != h2.get(k))
        rep["violations"].append({"check": "results_frozen", "run": n, "files": ch[:10]})
rep["frozen"] = {n: len(h) for n, h in frozen.items()}
json.dump(rep, open(OUT, "w"), indent=1)
