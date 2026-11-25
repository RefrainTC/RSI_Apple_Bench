#!/usr/bin/env python3
"""P2 copy of the G1 kernel gate (only change: harness-tamper pattern also names the P2 harness). Original doc:
G1 kernel gate: compare a workspace snapshot with the pristine copy and reject any added or
changed content that could create custom GPU code.

Usage:
  python3 -I kernel_gate.py --pristine P --candidate C [--json OUT] [--stage DIR] [--patch FILE]
                            [--client-nonblocking]

Scope (paths relative to the workspace root):
  submission/**, repo/**  -> GATED (these are the only trees that reach the container)
  client/**               -> GATED with the CLIENT rules below (optional dev-only load client, t2)
  dev_run.sh              -> INTEGRITY (must be byte-identical to pristine)
  everything else         -> listed as "not mounted", not content-checked
Ignored everywhere: dev_runs/, .git/ (workspace-level), repo/.git/, __pycache__/, *.pyc

Rules for gated added/changed files (deletions are allowed):
  R1 symlink                        -> reject
  R2 banned extension (native / GPU / binary / archive / .pth)  -> reject
  R3 binary content (NUL byte or not UTF-8)                    -> reject
  R4 changed file whose pristine version defines Triton/numba-jit kernels -> reject
  R5 added lines matching a REJECT pattern                      -> reject
  W  added lines matching a WARN pattern (dynamic code, encodings, torch.compile, ...) -> logged only

CLIENT rules (client/**, every line is checked since client/ is never in pristine):
  C0 client/ present but client/run.py missing                  -> reject
  C1 extension not in .py .json .jsonl .txt .csv .tsv .md .yaml .yml  -> reject (plus R1 / R3)
  C2 .py lines matching a general REJECT pattern or a CLIENT pattern (dynamic code, launching
     programs, GPU devices / GPU env vars, touching server code or site-packages, ...) -> reject;
     data files are checked only for verifier references and harness tampering
  C3 .py imports outside the allowlist: Python 3.10 stdlib minus CLIENT_DENY_STDLIB, the client's
     own modules, and the HTTP/async libraries already in the image                  -> reject
  --client-nonblocking (final scoring): client rejects are reported but do not change the verdict,
  because the client never runs at scoring time.

Exit code: 0 = pass, 1 = reject, 2 = usage / internal error.
--stage DIR (only on pass): DIR/submission (full gated submission tree), DIR/repo_overlay
(added+changed files under repo/), DIR/repo_deleted.txt (paths relative to repo/), DIR/client
(only when client/ exists and passed).
--patch FILE: unified diff of all gated changes (pristine -> candidate).
"""
import argparse
import ast
import difflib
import json
import os
import re
import shutil
import sys

IGNORE_DIRS_TOP = {"dev_runs", ".git"}
IGNORE_DIR_NAMES = {"__pycache__"}
IGNORE_SUFFIX = (".pyc",)
GATED_ROOTS = ("submission/", "repo/", "client/")
INTEGRITY = {"dev_run.sh"}

BANNED_EXT = {
    # CUDA / C / C++ / HIP / other kernel sources
    ".cu", ".cuh", ".cpp", ".cc", ".c", ".cxx", ".c++", ".h", ".hh", ".hpp", ".hxx", ".inl",
    ".hip", ".cl", ".metal", ".ptx", ".cubin", ".fatbin", ".spv", ".ll", ".bc", ".mlir",
    ".pyx", ".pxd", ".pxi",
    # compiled / loadable binaries
    ".so", ".o", ".a", ".pyd", ".dll", ".dylib", ".bin", ".exe",
    # archives / serialized blobs (could carry any of the above)
    ".whl", ".egg", ".zip", ".tar", ".gz", ".tgz", ".xz", ".bz2", ".zst", ".7z",
    ".pkl", ".pickle", ".npy", ".npz", ".pt", ".safetensors",
    # site-packages auto-exec hooks
    ".pth",
}

KERNEL_DEF_RE = re.compile(r"@\s*triton\s*\.\s*(jit|autotune|heuristics)\b|^\s*@\s*jit\b|\btl\s*\.\s*constexpr\b",
                           re.M)

REJECT_PATTERNS = [
    ("triton.jit", r"\btriton\s*\.\s*jit\b"),
    ("@triton", r"@\s*triton\b"),
    ("import triton", r"^\s*(?:import|from)\s.*\btriton"),
    ("dynamic import of triton", r"(?:__import__|import_module)\s*\(\s*['\"][\w.]*triton"),
    ("tl. usage", r"\btl\s*\.\s*[A-Za-z_]"),
    ("load_inline", r"\bload_inline\b"),
    ("cpp_extension", r"\bcpp_extension\b"),
    ("ctypes", r"\bctypes\b"),
    ("cffi", r"\bcffi\b"),
    ("cupy", r"\bcupy\b"),
    ("numba", r"\bnumba\b"),
    ("pycuda", r"\bpycuda\b"),
    ("nvrtc", r"(?i)\bnvrtc\b"),
    ("torch.ops.load_library", r"\bload_library\b"),
    ("torch.cuda.jiterator", r"\bjiterator\b"),
    ("CUDA source / driver launch", r"\bnvcc\b|\bhipcc\b|__global__|__device__|\bcuModuleLoad\w*|\bcuLaunchKernel\b|\bcudaLaunchKernel\b"),
    ("native compiler / build tool", r"(?<!\w)(?:gcc|g\+\+|clang\+\+|clang|cmake|ninja)(?![\w+])"),
    ("other kernel DSL", r"\b(?:cutlass|tilelang|tvm|taichi|pykeops)\b|^\s*(?:import|from)\s+warp\b|\bcuda\s*\.\s*bindings\b|^\s*from\s+cuda\s+import\b"),
    # same regex as tests/test.sh's own source-scan gate (instruction: never access /tests/)
    ("verifier reference", r"/tests/|compute_reward|launch_baseline|/logs/verifier|reward\.json|reward\.txt|prompts\.jsonl|reward_manifest|/opt/loop/private|hidden_suite"),
    ("harness tamper", r"g1_memclamp|G1_GPU_MEM_CAP|gpu\.lock|P2_OVERLAY|p2_job|p2-port|R4_TAG|t5-isobench|p2_harness"),
]

WARN_PATTERNS = [
    ("dynamic code", r"\bexec\s*\(|\beval\s*\(|(?<![\w.])compile\s*\(|__import__|\bimportlib\b|\bimport_module\b"),
    ("encoded payload", r"\bbase64\b|\bcodecs\b|\bmarshal\b|\bzlib\b|\bbinascii\b|\bpickle\b|\\x[0-9a-fA-F]{2}"),
    ("torch.compile / custom op", r"\btorch\s*\.\s*compile\b|\btorch\s*\.\s*_inductor\b|\btorch\s*\.\s*_dynamo\b|\btorch\s*\.\s*library\b|\bcustom_op\b"),
    ("native loading / build env", r"\bLD_PRELOAD\b|\bdlopen\b|\bTORCH_CUDA_ARCH_LIST\b"),
    ("tooling", r"\bpip\d?\s+install\b|\bnvidia-smi\b|\bdocker\b|python3?(?:\.\d+)?\s+-[A-Za-z]*S\b"),
]

_RE_REJECT = [(n, re.compile(p)) for n, p in REJECT_PATTERNS]
_RE_WARN = [(n, re.compile(p)) for n, p in WARN_PATTERNS]
_RE_DATA = [(n, rx) for n, rx in _RE_REJECT if n in ("verifier reference", "harness tamper")]

# ---- client (dev-only load generator) rules ----
CLIENT_EXT = {".py", ".json", ".jsonl", ".txt", ".csv", ".tsv", ".md", ".yaml", ".yml"}
# HTTP / async libraries already installed in the task image (pip list, 2026-11-21)
CLIENT_THIRD_PARTY = {"aiohttp", "httpx", "httpcore", "h11", "anyio", "sniffio", "openai", "requests",
                      "urllib3", "uvloop", "websockets"}
CLIENT_DENY_STDLIB = {"ctypes", "_ctypes", "subprocess", "_posixsubprocess", "pty", "importlib", "runpy",
                      "code", "codeop", "marshal", "pickle", "_pickle", "shelve", "mmap", "fcntl", "builtins",
                      "zipimport", "imp", "ensurepip", "venv", "distutils", "sysconfig", "_ctypes_test"}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "client_stdlib_py310.json")) as _fh:
    CLIENT_STDLIB = set(json.load(_fh))
CLIENT_PATTERNS = [
    ("client: dynamic code", r"\bexec\s*\(|\beval\s*\(|(?<![\w.])compile\s*\(|__import__|\bimportlib\b|\bimport_module\b|\brunpy\b|__builtins__|\bsys\s*\.\s*modules\b|\bgetattr\s*\(\s*(?:os|sys|builtins|asyncio|multiprocessing)\b"),
    ("client: launches programs", r"\bos\s*\.\s*(?:system|popen|exec\w*|spawn\w*|posix_spawn\w*|kill|killpg|putenv)\b|\bcreate_subprocess_\w+|\bsubprocess\b|\bpty\b|\bset_executable\b|\bsetpriv\b|\bsudo\b"),
    ("client: GPU device / GPU env", r"/dev/nvidia|nvidia-smi|\bnvml\b|pynvml|CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES|CUDA_HOME"),
    ("client: touches server code / env", r"/app/submission|/app/repo|site-packages|kernelbench-venv|\bLD_PRELOAD\b|\bfcntl\b|\bioctl\b|\bmmap\b"),
]
_RE_CLIENT = [(n, re.compile(p)) for n, p in CLIENT_PATTERNS]


def client_import_problems(text, local_mods):
    """[(lineno, module, why)] for imports outside the client allowlist; None if the file does not parse."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods = [al.name for al in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            mods = [node.module or ""]
        else:
            continue
        for m in mods:
            top = m.split(".")[0]
            if top in CLIENT_DENY_STDLIB:
                out.append((node.lineno, m, "denied stdlib module"))
            elif top in CLIENT_STDLIB or top in CLIENT_THIRD_PARTY or top in local_mods:
                continue
            else:
                out.append((node.lineno, m, "not stdlib / not an allowed HTTP-async library / not a client module"))
    return out


def is_gated(rel):
    return rel.startswith(GATED_ROOTS) or (rel + "/") in GATED_ROOTS


def ignored(rel):
    parts = rel.split("/")
    if parts[0] in IGNORE_DIRS_TOP:
        return True
    if rel.startswith("repo/.git/") or rel == "repo/.git":
        return True
    if any(p in IGNORE_DIR_NAMES for p in parts):
        return True
    return rel.endswith(IGNORE_SUFFIX)


def walk(root):
    """rel path -> ('file'|'link', abs path). Directories are not entries; links are not followed."""
    out = {}
    for dp, dns, fns in os.walk(root, followlinks=False):
        rel_dp = os.path.relpath(dp, root)
        keep = []
        for d in dns:
            rel = d if rel_dp == "." else f"{rel_dp}/{d}"
            full = os.path.join(dp, d)
            if ignored(rel):
                continue
            if os.path.islink(full):
                out[rel] = ("link", full)
            else:
                keep.append(d)
        dns[:] = keep
        for f in fns:
            rel = f if rel_dp == "." else f"{rel_dp}/{f}"
            if ignored(rel):
                continue
            full = os.path.join(dp, f)
            out[rel] = ("link" if os.path.islink(full) else "file", full)
    return out


def read_bytes(p):
    with open(p, "rb") as fh:
        return fh.read()


def as_text(b):
    if b"\x00" in b:
        return None
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return None


def banned_ext(rel):
    name = rel.rsplit("/", 1)[-1].lower()
    for ext in BANNED_EXT:
        if name.endswith(ext) or (ext + ".") in name:  # also catches lib.so.1, x.cu.bak ...
            return ext
    return None


def added_lines(old_text, new_text):
    """[(lineno_in_new, line)] for lines added or changed (difflib, no context)."""
    if old_text is None:
        return list(enumerate(new_text.splitlines(), 1))
    out = []
    sm = difflib.SequenceMatcher(None, old_text.splitlines(), new_text.splitlines(), autojunk=False)
    new_lines = new_text.splitlines()
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):
            out.extend((j + 1, new_lines[j]) for j in range(j1, j2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pristine", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--json")
    ap.add_argument("--stage")
    ap.add_argument("--patch")
    ap.add_argument("--max-findings", type=int, default=40)
    ap.add_argument("--client-nonblocking", action="store_true")
    a = ap.parse_args()

    P, C = walk(a.pristine), walk(a.candidate)
    added = sorted(set(C) - set(P))
    deleted = sorted(set(P) - set(C))
    changed = []
    for rel in sorted(set(P) & set(C)):
        (tp, pp), (tc, pc) = P[rel], C[rel]
        if tp != tc:
            changed.append(rel)
        elif tp == "link":
            if os.readlink(pp) != os.readlink(pc):
                changed.append(rel)
        elif os.path.getsize(pp) != os.path.getsize(pc) or read_bytes(pp) != read_bytes(pc):
            changed.append(rel)

    rejects, warns, unmounted, client_rejects = [], [], [], []

    def rej(rel, rule, line=None, text=None):
        item = {"path": rel, "rule": rule, "line": line, "text": (text or "")[:200]}
        if a.client_nonblocking and rel.startswith("client"):
            client_rejects.append(item)
        else:
            rejects.append(item)

    client_files = sorted(r for r in C if r.startswith("client/") or r == "client")
    local_mods = set()
    for r in client_files:
        parts = r.split("/")
        if len(parts) == 2 and parts[1].endswith(".py"):
            local_mods.add(parts[1][:-3])
        elif len(parts) > 2:
            local_mods.add(parts[1])
    if client_files and "client/run.py" not in C:
        rej("client/run.py", "C0 client/ exists but the entry point client/run.py is missing")

    def check_client_file(rel, path):
        ext = os.path.splitext(rel)[1].lower()
        if ext not in CLIENT_EXT:
            rej(rel, f"C1 file type not allowed in client/ ({ext or 'no extension'})")
            return
        text = as_text(read_bytes(path))
        if text is None:
            rej(rel, "R3 binary content")
            return
        lines = list(enumerate(text.splitlines(), 1))
        if ext != ".py":
            for ln, line in lines:
                for name, rx in _RE_DATA:
                    if rx.search(line):
                        rej(rel, f"C2 {name}", ln, line.strip())
            return
        for ln, line in lines:
            for name, rx in _RE_REJECT + _RE_CLIENT:
                if rx.search(line):
                    rej(rel, f"C2 {name}", ln, line.strip())
            for name, rx in _RE_WARN:
                if rx.search(line):
                    warns.append({"path": rel, "rule": name, "line": ln, "text": line.strip()[:200]})
        probs = client_import_problems(text, local_mods)
        if probs is None:
            rej(rel, "C3 client .py does not parse (imports cannot be verified)")
            return
        for ln, mod, why in probs:
            rej(rel, f"C3 import not allowed in client: {mod} ({why})", ln)

    for rel in INTEGRITY:
        if rel in changed or rel in deleted:
            rej(rel, "harness tamper: dev_run.sh must stay unchanged")

    gated_changes = []
    for rel in added + changed:
        if rel in INTEGRITY:
            continue
        if not is_gated(rel):
            unmounted.append(rel)
            continue
        gated_changes.append(rel)
        kind, path = C[rel]
        if kind == "link" or (rel in P and P[rel][0] == "link"):
            rej(rel, "R1 symlink not allowed")
            continue
        if rel.startswith("client/"):
            check_client_file(rel, path)
            continue
        ext = banned_ext(rel)
        if ext:
            rej(rel, f"R2 banned file type ({ext})")
            continue
        new_text = as_text(read_bytes(path))
        if new_text is None:
            rej(rel, "R3 binary content")
            continue
        old_text = None
        if rel in P:
            old_text = as_text(read_bytes(P[rel][1]))
            if old_text is not None and KERNEL_DEF_RE.search(old_text):
                rej(rel, "R4 modifies a file that defines GPU (Triton) / jit kernels")
                continue
        for ln, line in added_lines(old_text, new_text):
            for name, rx in _RE_REJECT:
                if rx.search(line):
                    rej(rel, f"R5 {name}", ln, line.strip())
            for name, rx in _RE_WARN:
                if rx.search(line):
                    warns.append({"path": rel, "rule": name, "line": ln, "text": line.strip()[:200]})
    for rel in deleted:
        if rel not in INTEGRITY and not is_gated(rel):
            unmounted.append(rel)

    verdict = "REJECT" if rejects else "PASS"
    client_ok = bool(client_files) and not client_rejects and not any(
        r["path"].startswith("client") for r in rejects)
    gated_deleted = [r for r in deleted if is_gated(r)]
    report = {
        "verdict": verdict,
        "counts": {"added": len(added), "changed": len(changed), "deleted": len(deleted),
                   "gated_added_or_changed": len(gated_changes), "gated_deleted": len(gated_deleted),
                   "rejects": len(rejects), "warnings": len(warns)},
        "gated_added_or_changed": gated_changes,
        "gated_deleted": gated_deleted,
        "not_mounted_changes": sorted(set(unmounted)),
        "rejects": rejects,
        "warnings": warns,
        "client": {"present": bool(client_files), "files": client_files, "passed": client_ok,
                   "nonblocking_rejects": client_rejects},
    }

    # ---- human-readable report ----
    print(f"[gate] verdict: {verdict}")
    print(f"[gate] gated changes: {len(gated_changes)} added/changed, {len(gated_deleted)} deleted "
          f"(submission/ + repo/ + client/)")
    for rel in gated_changes[: a.max_findings]:
        print(f"[gate]   {'A' if rel in added else 'M'} {rel}")
    for rel in gated_deleted[: a.max_findings]:
        print(f"[gate]   D {rel}")
    if report["not_mounted_changes"]:
        print(f"[gate] changes outside submission/, repo/, client/ (not mounted, not checked): "
              f"{len(report['not_mounted_changes'])}")
    for r in rejects[: a.max_findings]:
        loc = f"{r['path']}:{r['line']}" if r["line"] else r["path"]
        print(f"[gate] REJECT {loc}: {r['rule']}" + (f" | {r['text']}" if r["text"] else ""))
    if len(rejects) > a.max_findings:
        print(f"[gate] ... {len(rejects) - a.max_findings} more rejects")
    for w in warns[: a.max_findings]:
        print(f"[gate] warn   {w['path']}:{w['line']}: {w['rule']} | {w['text']}")
    if len(warns) > a.max_findings:
        print(f"[gate] ... {len(warns) - a.max_findings} more warnings")
    for r in client_rejects[: a.max_findings]:
        loc = f"{r['path']}:{r['line']}" if r["line"] else r["path"]
        print(f"[gate] client reject (non-blocking) {loc}: {r['rule']}" + (f" | {r['text']}" if r["text"] else ""))

    if a.json:
        with open(a.json, "w") as fh:
            json.dump(report, fh, indent=1, ensure_ascii=False)

    if a.patch:
        with open(a.patch, "w") as fh:
            for rel in sorted(set(gated_changes) | set(gated_deleted)):
                old = as_text(read_bytes(P[rel][1])) if rel in P and P[rel][0] == "file" else ""
                new = as_text(read_bytes(C[rel][1])) if rel in C and C[rel][0] == "file" else ""
                if old is None or new is None:
                    fh.write(f"Binary or non-UTF-8 file differs: {rel}\n")
                    continue
                fh.writelines(difflib.unified_diff(
                    old.splitlines(True), new.splitlines(True),
                    fromfile=("a/" + rel) if rel in P else "/dev/null",
                    tofile=("b/" + rel) if rel in C else "/dev/null"))

    if a.stage and verdict == "PASS":
        st = a.stage
        if os.path.exists(st):
            shutil.rmtree(st)
        os.makedirs(os.path.join(st, "repo_overlay"))
        sub_src = os.path.join(a.candidate, "submission")
        sub_dst = os.path.join(st, "submission")
        os.makedirs(sub_dst)
        for rel, (kind, path) in C.items():
            if rel.startswith("submission/") and kind == "file":
                dst = os.path.join(st, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(path, dst)
        if not os.path.isdir(sub_src):
            print("[gate] note: workspace has no submission/ directory", file=sys.stderr)
        for rel in gated_changes:
            if rel.startswith("repo/"):
                dst = os.path.join(st, "repo_overlay", rel[len("repo/"):])
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(C[rel][1], dst)
        with open(os.path.join(st, "repo_deleted.txt"), "w") as fh:
            for rel in gated_deleted:
                if rel.startswith("repo/"):
                    fh.write(rel[len("repo/"):] + "\n")
        if client_ok and not a.client_nonblocking:  # final scoring never stages / mounts the client
            for rel in client_files:
                if C[rel][0] == "file":
                    dst = os.path.join(st, rel)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(C[rel][1], dst)

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"[gate] INTERNAL ERROR: {e!r}", file=sys.stderr)
        sys.exit(2)
