#!/usr/bin/env python3
"""agent_view.py <full summary.json> <staging dir> <agent-visible out dir> : the ONLY things a dev run returns to the
solver (leakage review 2026-11-23: candidate code runs as `coder` on the pod and could read the human fix, the scorer or
the reference outputs and smuggle them out through whatever text the run returns).
  summary.json   : the dev summary reduced to structured fields: keys [a-z0-9_]{1,40}, numbers (rounded to 6
                   significant digits), booleans, null, enum-like short strings [A-Za-z0-9_.:+=-]{0,40}, lists <= 50.
  <run>_log.txt  : per run (cand / parent / correct / parent_correct), at most 30 lines, each <= 160 printable ASCII
                   characters, of fixed formats only: vLLM logger lines, Python traceback frame lines (file shown relative
                   to vllm/, source line only for frames inside vllm/), the final exception line (message <= 160 chars).
                   Absolute pod paths become <path>; runs of >= 40 base64/hex-like characters become <blob>.
  gate.txt       : gate verdict lines ([gate] / [lock] / [p2_rules] / GATE ...): one line per finding with rule, the
                   solver's relative file:line (repo/... or NOTES.md) and the offending line of its own code, first 20
                   findings plus a count of the rest; changed-file list <= 15; pod paths -> <path>, non-repo/ blobs -> <blob>.
Everything else (raw logs, outputs, stages) stays on the pod. Text is scanned OFF the pod for oracle- or scorer-only
strings by local/leak_scan.py before the solver sees it."""
import json, math, os, re, sys
full, S, V = sys.argv[1:4]
KEY = re.compile(r"^[a-z0-9_]{1,40}$"); ENUM = re.compile(r"^[A-Za-z0-9_.:+=-]{0,40}$")
def num(x):
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if not math.isfinite(x):
            return None
        return float(f"{x:.6g}")
    return None
def clean(x, depth=0):
    if depth > 6:
        return None
    if isinstance(x, dict):
        return {k: clean(v, depth + 1) for k, v in list(x.items())[:80] if isinstance(k, str) and KEY.match(k)}
    if isinstance(x, (list, tuple)):
        return [clean(v, depth + 1) for v in list(x)[:50]]
    if isinstance(x, str):
        return x if ENUM.match(x) else "<text>"
    return num(x)
PATH = re.compile(r"(/home/coder|/tmp/p2|/home/[A-Za-z0-9_.-]+|/root|/etc|/proc)(/[^\s\"':,)\]]*)?")
BLOB = re.compile(r"[A-Za-z0-9+/=_-]{40,}")
def scrub(s, n=160):
    s = "".join(c for c in s if 32 <= ord(c) < 127)
    s = PATH.sub("<path>", s); s = BLOB.sub("<blob>", s)
    return s[:n]
LOGLINE = re.compile(r"^(INFO|WARNING|ERROR|DEBUG|CRITICAL) \d\d-\d\d \d\d:\d\d:\d\d \[?[\w./-]+:\d+\] ")   # 0.5.x: "x.py:12]", 0.8.x: "[x.py:12]"
FRAME = re.compile(r'^\s*File "([^"]+)", line (\d+), in ([\w<>.]+)\s*$')
EXC = re.compile(r"^([A-Za-z_][\w.]*(Error|Exception|Interrupt|Exit|Warning|Fault))(: (.*))?$")
def tail(path, keep=30):
    try:
        lines = open(path, errors="replace").read().splitlines()[-4000:]
    except Exception:
        return []
    out = []; src_ok = False
    for ln in lines:
        m = FRAME.match(ln)
        if m:
            f = m.group(1); i = f.rfind("/vllm/")
            rel = ("vllm/" + f[i + 6:]) if i >= 0 else os.path.basename(f)
            src_ok = i >= 0
            out.append(scrub(f'  File "{rel}", line {m.group(2)}, in {m.group(3)}')); continue
        if src_ok and ln.startswith("    ") and not ln.startswith("     File"):
            out.append(scrub(ln)); src_ok = False; continue
        src_ok = False
        if ln.startswith("Traceback (most recent call last)"):
            out.append("Traceback (most recent call last):"); continue
        m = EXC.match(ln.strip())
        if m:
            out.append(scrub(m.group(1) + (": " + m.group(4) if m.group(4) else ""))); continue
        if LOGLINE.match(ln) and "[P2COUNTER]" not in ln:
            out.append(scrub(ln))
    return out[-keep:]
os.makedirs(V, exist_ok=True)
try:
    summ = json.load(open(full))
except Exception:
    summ = {"error": "no_summary"}
json.dump(clean(summ), open(os.path.join(V, "summary.json"), "w"), indent=1)
for name in ("cand", "parent", "correct", "parent_correct"):
    d = os.path.join(S, name)
    if not os.path.isdir(d):
        continue
    keep = []
    for f in ("job_stdout.log", "server.log"):
        keep += tail(os.path.join(d, f))
    if keep:
        open(os.path.join(V, f"{name}_log.txt"), "w").write("\n".join(keep[-30:]) + "\n")
g = os.path.join(S, "gate.txt")
if os.path.exists(g):
    # The solver's own relative paths (repo/..., NOTES.md) stay readable: <blob> only for tokens that are not repo/ paths.
    GBLOB = re.compile(r"(?<![\w/.-])(?!repo/)[A-Za-z0-9+/=_-]{40,}")
    def gscrub(l):
        l = "".join(c for c in l if 32 <= ord(c) < 127)
        return GBLOB.sub("<blob>", PATH.sub("<path>", l))[:200]
    FIND = re.compile(r"^(\[gate\] (REJECT |warn |client reject)|\[p2_rules\]   (REJECT|WARN) )")   # one line per finding, file:line inside
    LIST = re.compile(r"^\[gate\]   [AMD] ")
    gl, nf, nl, xf, xl = [], 0, 0, 0, 0
    for l in open(g, errors="replace").read().splitlines():
        if not re.match(r'^(\[gate\]|\[lock\]|\[p2_rules\]|GATE |\{"verdict")', l) or l.startswith("[gate] ... "):
            continue
        if FIND.match(l):
            nf += 1
            if nf > 20:
                xf += 1; continue
        elif LIST.match(l):
            nl += 1
            if nl > 15:
                xl += 1; continue
        gl.append(gscrub(l))
    if xl:
        gl.append(f"[gate]   ... {xl} more changed files")
    if xf:
        gl.append(f"[gate] ... {xf} more findings not shown (first 20 above)")
    open(os.path.join(V, "gate.txt"), "w").write("\n".join(gl[:60]) + "\n")
