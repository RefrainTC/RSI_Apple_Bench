"""field_check.py : clean-the-field census for trial mode (REVIEW_v4 M5), same-user version (no sudo).

  python -I field_check.py snapshot <census.json>              at trial start (operator)
  python -I field_check.py check <census.json> <gpu> [out.json]  before a final score / after each dev run

check reports, for this user:
  orphans  : processes started after the snapshot whose parent is init (detached: double fork, setsid + exit)
  marked   : processes whose environment carries a server marker (an overlay on PYTHONPATH, P3_RUN_TAG,
             P3_AUDIT_FIFO) and that use the trial GPU (CUDA_VISIBLE_DEVICES) or no GPU setting at all; run
             when no session of that GPU is active, every such process is a leftover of candidate code
  files    : state that could carry over between runs: names of this user new since the snapshot at the top
             level of /tmp and the home directory, and any file of this user changed after the snapshot
             under /dev/shm, /var/tmp, /run/lock. $P3_STAGE and tool dirs are excluded
             (harness-owned).
             Same-user mode cannot census the whole home tree; writes elsewhere are the audit hook's job.
ok = nothing in any list. Exit 0 if ok, 1 otherwise. Nothing is killed or deleted here: the operator decides.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p3lib  # noqa: E402

TMP_SKIP = (p3lib.STAGE, "/tmp/claude", "/tmp/tmux-", "/tmp/.X11", "/tmp/.ICE", "/tmp/.font", "/tmp/vscode",
            "/tmp/code-server", "/tmp/.code-server", "/tmp/systemd")  # p3lib.STAGE: $P3_STAGE
MARKERS = (b"/overlay/o", b"P3_RUN_TAG=", b"P3_AUDIT_FIFO=")


def env_of(pid):
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            return f.read().split(b"\0")
    except Exception:
        return None


TOP_LEVEL = ("/tmp", os.path.expanduser("~"))       # names here that were not in the snapshot
DEEP = ("/dev/shm", "/var/tmp", "/run/lock")  # any file of this user changed after the snapshot


def top_names() -> dict[str, list[str]]:
    out = {}
    for r in TOP_LEVEL:
        try:
            out[r] = sorted(os.listdir(r))
        except OSError:
            out[r] = []
    return out


def new_files(since: float, names0: dict) -> list[str]:
    me = os.getuid()
    out = []
    for r in TOP_LEVEL:
        for n in sorted(set(os.listdir(r)) - set(names0.get(r, []))):
            p = os.path.join(r, n)
            try:
                if os.lstat(p).st_uid == me and not p.startswith(TMP_SKIP):
                    out.append(p)
            except FileNotFoundError:
                continue
    for root in DEEP:
        for dp, dns, fns in os.walk(root):
            for n in fns + dns:
                p = os.path.join(dp, n)
                try:
                    st = os.lstat(p)
                except FileNotFoundError:
                    continue
                if st.st_uid == me and max(st.st_mtime, st.st_ctime) > since:
                    out.append(p)
    return sorted(out)[:200]


def main():
    mode, f = sys.argv[1], Path(sys.argv[2])
    if mode == "snapshot":
        procs = p3lib.coder_procs()
        f.write_text(json.dumps({"t": time.time(), "procs": {str(k): v for k, v in procs.items()}, "names": top_names()}))
        print(f"snapshot: {len(procs)} processes")
        return 0
    gpu = sys.argv[3]
    snap = json.loads(f.read_text())
    old = {int(k): v["start"] for k, v in snap["procs"].items()}
    orphans, marked = [], []
    for pid, p in p3lib.coder_procs().items():
        if pid == os.getpid():
            continue
        new = old.get(pid) != p["start"]
        if new and p["ppid"] == 1:
            orphans.append({"pid": pid, "cmd": p["cmd"]})
        env = env_of(pid)
        if env and any(m in e for e in env for m in MARKERS):
            cvd = [e.split(b"=", 1)[1].decode() for e in env if e.startswith(b"CUDA_VISIBLE_DEVICES=")]
            if not cvd or gpu in cvd[0].split(","):
                marked.append({"pid": pid, "cmd": p["cmd"]})
    files = new_files(snap["t"], snap.get("names", {}))
    res = {"ok": not (orphans or marked or files), "orphans": orphans, "marked": marked, "files": files,
           "snapshot_age_s": round(time.time() - snap["t"])}
    if len(sys.argv) > 4:
        Path(sys.argv[4]).write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
