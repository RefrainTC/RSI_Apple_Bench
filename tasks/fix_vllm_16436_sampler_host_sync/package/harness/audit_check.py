"""audit_check.py : summarize the P2 audit hook records of one run (harness/audit/sitecustomize.py).
events(out, need_end, ignore_rx) -> {"events": n, "kinds": {...}, "missing_end": bool, "sample": [...]}"""
import glob, json, os, re


def events(out, need_end=False, ignore_rx=""):
    files = glob.glob(os.path.join(out, "audit.*.jsonl"))
    if not files:
        return {"events": None, "reason": "no audit records (hook not loaded?)", "missing_end": bool(need_end)}
    ev, ends, pids = [], set(), set()
    for f in files:
        for l in open(f, errors="replace"):
            try:
                d = json.loads(l)
            except Exception:
                continue
            pids.add(d.get("pid"))
            if d.get("kind") == "end":
                ends.add(d.get("pid")); continue
            if ignore_rx and re.search(ignore_rx, json.dumps(d)):
                continue
            ev.append(d)
    kinds = {}
    for d in ev:
        kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
    return {"events": len(ev), "kinds": kinds, "processes": len(pids), "missing_end": bool(need_end) and not ends,
            "sample": ev[:8]}
