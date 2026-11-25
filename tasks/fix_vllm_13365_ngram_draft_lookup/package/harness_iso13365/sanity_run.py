"""sanity_run.py <taskdir> <run dir> [--strict] : is a harness run usable? (used right after the first run of a calibration,
so a harness bug cannot spoil a whole calibration). Hard failures (exit 1): rc != 0; audit enabled but no audit records,
or the main process's end record missing when required; venv guard enabled but no record or pre/post not ok;
missing job metadata fields (storage, cpus, cache). Audit EVENTS on trusted code are printed (they become the
allowlist question: benign library behaviour vs a harness bug) and fail only with --strict."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_check as AC
td, d = sys.argv[1], sys.argv[2]; strict = "--strict" in sys.argv
env = {}
for l in open(os.path.join(td, "task.env")):
    l = l.split("#")[0].strip()
    for part in l.replace("export ", "").split(";"):
        if "=" in part:
            k, v = part.strip().split("=", 1); env[k.strip()] = v.strip().strip('"').strip("'")
bad, info = [], {}
m = json.load(open(os.path.join(d, "job_meta.json")))
info["rc"] = m.get("rc"); info["storage"] = m.get("storage"); info["cpus"] = m.get("cpus"); info["cache"] = m.get("cache")
if m.get("rc") != 0: bad.append(f"rc={m.get('rc')}")
for k in ("storage", "cpus", "cache"):
    if not m.get(k): bad.append(f"job_meta has no {k}")
if env.get("AUDIT") == "1":
    a = AC.events(d, env.get("P2_AUDIT_NEED_END") == "1", env.get("P2_AUDIT_IGNORE", "")); info["audit"] = a
    if a.get("events") is None: bad.append("no audit records")
    if a.get("missing_end"): bad.append("audit end record missing")
    if a.get("events") and strict: bad.append(f"audit events on trusted code: {a.get('kinds')}")
if env.get("P2_FRESH_VENV") == "1":
    vg = json.load(open(os.path.join(d, "venv_guard.json"))) if os.path.exists(os.path.join(d, "venv_guard.json")) else None
    info["venv_guard"] = vg
    if not vg or (vg.get("post") or {}).get("state") != "ok" or (vg.get("pre") or {}).get("state") not in ("ok", "fresh_copy"):
        bad.append(f"venv guard: {vg}")
print(json.dumps({"run": d, "ok": not bad, "problems": bad, **info}, default=str))
sys.exit(1 if bad else 0)
