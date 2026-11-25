#!/usr/bin/env python3
"""unpack_snapshot.py <snapshot.tgz> <pristine workspace> <dest workspace> <report.json> : build a pod-side workspace from a
solver's snapshot (trial mode: the solver works on another machine, its dev_run.sh ships a tar.gz of its workspace).
The snapshot is untrusted. Only regular files and directories are accepted; absolute paths, '..' components, links,
devices and fifos make the whole snapshot invalid (exit 1). Limits: 20000 members, 300 MB unpacked, 30 MB per file.
Member paths are relative to the workspace root and repo/ must be the COMPLETE repo tree (a file missing from it counts as
deleted). .git/, __pycache__/, *.pyc and dev_runs/ are skipped. Only repo/** and NOTES.md are taken from the snapshot; everything else (dev_run.sh, instruction.md, WORKLOAD.md, ...)
comes from the pristine copy, so the solver's local wrapper never reaches the gate or a run. The gate (snapshot_gate.sh)
then runs on <dest> exactly as for a pod-local workspace."""
import json, os, shutil, stat, sys, tarfile
src, pristine, dest, rep = sys.argv[1:5]
MAXN, MAXTOT, MAXF = 20000, 300 << 20, 30 << 20
r = {"snapshot": os.path.basename(src), "accepted": 0, "bytes": 0, "ignored": [], "invalid": []}   # no pod path: agent-visible
def bad(msg):
    r["invalid"].append(msg)
try:
    tf = tarfile.open(src, "r:*")
    members = tf.getmembers()
except Exception as e:
    members = []; bad(f"unreadable archive: {e!r}")
if len(members) > MAXN:
    bad(f"{len(members)} members > {MAXN}")
use = []
for m in members:
    n = m.name
    while n.startswith("./"):
        n = n[2:]
    if not n or n == ".":
        continue
    parts = n.split("/")
    if n.startswith("/") or ".." in parts or any(p == "" for p in parts[:-1]):
        bad(f"unsafe path: {m.name}"); continue
    if not (m.isfile() or m.isdir()):
        bad(f"not a regular file or directory: {m.name} (type {m.type!r})"); continue
    if m.isfile() and m.size > MAXF:
        bad(f"file too large: {m.name} ({m.size} bytes)"); continue
    if any(p in (".git", "__pycache__") for p in parts) or n.endswith(".pyc") or parts[0] == "dev_runs":
        continue                                   # never staged (snapshot_gate excludes them too)
    if not (parts[0] == "repo" or n == "NOTES.md"):
        if len(r["ignored"]) < 50: r["ignored"].append(n)
        continue
    r["bytes"] += m.size if m.isfile() else 0
    use.append((n, m))
if r["bytes"] > MAXTOT:
    bad(f"{r['bytes']} bytes unpacked > {MAXTOT}")
if not any(n.startswith("repo/") for n, _ in use):
    bad("no repo/ in the snapshot")
if r["invalid"]:
    json.dump(r, open(rep, "w"), indent=1); print(json.dumps(r)); sys.exit(1)
if os.path.exists(dest):
    sys.exit(f"{dest} exists")
shutil.copytree(pristine, dest, symlinks=False)
for root, dirs, files in os.walk(dest):
    for x in dirs + files:
        p = os.path.join(root, x); os.chmod(p, os.stat(p).st_mode | stat.S_IWUSR)
os.chmod(dest, os.stat(dest).st_mode | stat.S_IWUSR)
shutil.rmtree(os.path.join(dest, "repo"))
for n, m in use:
    out = os.path.join(dest, n)
    if m.isdir():
        os.makedirs(out, exist_ok=True); continue
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with tf.extractfile(m) as fi, open(out, "wb") as fo:
        shutil.copyfileobj(fi, fo)
    os.chmod(out, 0o644)
    r["accepted"] += 1
json.dump(r, open(rep, "w"), indent=1); print(json.dumps({k: r[k] for k in ("accepted", "bytes")} | {"ignored": len(r["ignored"])}))
