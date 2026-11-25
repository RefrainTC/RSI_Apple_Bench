"""audit.py <parent venv> <candidate venv> <out.json> : static audit of a candidate for the cold-start task. Stdlib only.

The task's score is a cold start, so a candidate must not bring compiled state with it. Rules (any finding -> the
candidate fails with "audit"; every finding is listed):
  1. Files that differ from the parent venv (added, removed or changed; sha256 of every file, bytecode aside) must be
     Python source (.py). Exceptions: the engine's own dist-info metadata (vllm-*.dist-info), each file < 2 MB, because a
     candidate built from another vLLM commit has a different version string; and any package's dist-info/RECORD whose
     lines are all plain RECORD rows (it lists bin/ entry points, whose shebangs carry the venv's own path).
  2. In every changed or added .py file, the lines the candidate added (difflib against the parent's file) must not
     contain a string literal or line longer than 2,000 characters, must not total more than 200 kB, and must not
     mention cache locations or serialised payloads: TRITON_CACHE_DIR, TRITON_HOME, TRITON_OVERRIDE, TORCHINDUCTOR,
     VLLM_CACHE_ROOT, VLLM_CONFIG_ROOT, XDG_CACHE_HOME, CUDA_CACHE, FLASHINFER_WORKSPACE, cache_manager,
     CacheManager, .cubin, base64, b64decode, zlib, lzma, marshal, pickle.loads, os.environ[...] = (environment writes).
These rules are conservative: a legitimate candidate that needs one of these names must be reviewed by hand.
Bytecode (__pycache__/*.pyc) is excluded from rule 1 (it follows the source); the run-time venv check in harness/run.sh
covers files written during a run."""
import difflib, hashlib, json, os, re, sys

P, C, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SP = "lib/python3.10/site-packages"
BAD = [r"TRITON_CACHE_DIR", r"TRITON_HOME", r"TRITON_OVERRIDE", r"TORCHINDUCTOR", r"VLLM_CACHE_ROOT", r"VLLM_CONFIG_ROOT",
       r"XDG_CACHE_HOME", r"CUDA_CACHE", r"FLASHINFER_WORKSPACE", r"cache_manager", r"CacheManager", r"\.cubin", r"base64",
       r"b64decode", r"\bzlib\b", r"\blzma\b", r"\bmarshal\b", r"pickle\.loads", r"os\.environ\s*\[[^\]]*\]\s*=",
       r"os\.environ\.(update|setdefault)\(", r"putenv"]


def files(root):
    out = {}
    for d, dirs, fs in os.walk(root):
        dirs.sort()
        if os.path.basename(d) == "__pycache__": continue
        for f in fs:
            p = os.path.join(d, f); rel = os.path.relpath(p, root)
            if os.path.islink(p): out[rel] = ("link", os.readlink(p)); continue
            st = os.stat(p); out[rel] = (st.st_size, st.st_mtime_ns, st.st_ino)
    return out


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


a, b = files(P), files(C)
diff = []
for rel in sorted(set(a) | set(b)):
    if rel == "pyvenv.cfg" or rel.startswith("bin/"): continue   # interpreter paths and entry-point scripts
    if rel not in a: diff.append(("added", rel)); continue
    if rel not in b: diff.append(("removed", rel)); continue
    x, y = a[rel], b[rel]
    if x == y: continue
    if x[0] == "link" or y[0] == "link":
        if x != y: diff.append(("changed", rel))
        continue
    if x[2] == y[2]: continue          # the same inode (hard link of the parent's file): identical content
    if x[0] != y[0] or sha(os.path.join(P, rel)) != sha(os.path.join(C, rel)): diff.append(("changed", rel))
findings = []; changed_py = []
for kind, rel in diff:
    if rel.endswith(".py"):
        if kind != "removed": changed_py.append(rel)
        continue
    if re.match(re.escape(SP) + r"/vllm-[^/]+\.dist-info/", rel):
        if kind == "removed" or os.path.getsize(os.path.join(C, rel)) < 2_000_000: continue
    if re.match(re.escape(SP) + r"/[^/]+\.dist-info/RECORD$", rel) and kind == "changed":
        # RECORD lists the venv's bin/ entry points, whose shebangs name the venv's own path; accept a changed RECORD
        # only if every line is a plain RECORD row (path,sha256=...,size or path,,) and it is < 2 MB
        txt = open(os.path.join(C, rel), errors="replace").read()
        if len(txt) < 2_000_000 and all(re.fullmatch(r'[^,]+,(sha256=[A-Za-z0-9_-]{43})?,\d*', l) or
                                        re.fullmatch(r'"[^"]+",(sha256=[A-Za-z0-9_-]{43})?,\d*', l) for l in txt.splitlines() if l):
            continue
    findings.append(f"non-Python file {kind}: {rel}")
added_total = 0
for rel in changed_py:
    old = open(os.path.join(P, rel), errors="replace").read().splitlines() if rel in a else []
    new = open(os.path.join(C, rel), errors="replace").read().splitlines()
    added = [l[1:] for l in difflib.unified_diff(old, new, lineterm="", n=0) if l.startswith("+") and not l.startswith("+++")]
    added_total += sum(len(l) + 1 for l in added)
    for l in added:
        if len(l) > 2000: findings.append(f"{rel}: added line of {len(l)} characters"); break
        hit = [r for r in BAD if re.search(r, l)]
        if hit: findings.append(f"{rel}: added line mentions {hit}: {l.strip()[:160]}")
if added_total > 200_000: findings.append(f"added Python source totals {added_total} bytes (> 200 kB)")
res = {"parent": P, "candidate": C, "n_files_differing": len(diff), "differing": [f"{k} {r}" for k, r in diff][:400],
       "changed_py": changed_py, "added_py_bytes": added_total, "findings": findings, "ok": not findings}
json.dump(res, open(OUT, "w"), indent=1)
print(f"audit: {len(diff)} files differ from the parent ({len(changed_py)} .py); findings {len(findings)}"
      + "".join(f"\n  - {x}" for x in findings[:20]))
