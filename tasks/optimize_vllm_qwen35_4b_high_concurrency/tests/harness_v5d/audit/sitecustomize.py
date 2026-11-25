"""Runtime audit hook, v5 (ported from our earlier harness and hardened after a review).

Loaded as sitecustomize in every Python process of a measured vLLM server (API server, engine core, helpers), for
the baseline and the candidate alike, BEFORE any vllm import. It never blocks anything. It is a tripwire, not a
wall: the candidate runs as the same Unix user (no sudo on the pod), so the real protections are the narrow
returns, the sealed secrets and the output witness. Hardening against the v4 attacks:
  - no module-level state: names, lists and the output fd live in the closure of _install(), and the module
    deletes its own globals, so `sitecustomize._emit = ...` and friends do nothing;
  - events go to a FIFO whose reader is the scorer (P3_AUDIT_FIFO): what was written cannot be truncated or
    rewritten afterwards (v4: os.ftruncate on the jsonl erased the record);
  - every process writes a `start` record first; the scorer voids a session that has fewer than two server
    processes with a start record (hook not loaded, or removed before it could report = fail closed).
Events (each a JSON line, at most 50 per kind and process):
  start           : this process loaded the hook (pid, ppid, argv0)
  write_protected : opened for writing / truncated / removed / renamed under P3_AUDIT_PROTECT_W
  write_outside   : the same anywhere else outside this server's own dirs (P3_AUDIT_ALLOW_W) and /dev/null,
                    /proc, /sys, /dev/nvidia*. /dev/shm counts as outside except the names multiprocessing and
                    vLLM create (sem.mp-*, psm_*, mp-*)
  code_outside    : Python source / extension loaded from outside the venv, the stdlib, PYTHONPATH and
                    P3_AUDIT_CODE_ROOTS
  exec            : a process started that is not python (multiprocessing spawn), a compiler or a system tool
  fork            : os.fork() called from code outside the venv / stdlib (multiprocessing spawn does not fork)
  hook            : sys.addaudithook / settrace / setprofile from code outside the venv / stdlib
  read_protected  : a file opened for reading, or a directory listed, under P3_AUDIT_PROTECT_R (the home directory, the
                    run scratch tree, /tmp, /var/tmp, /dev/shm, /run/lock, /root) outside
                    P3_AUDIT_ALLOW_R / this server's own dirs / the venv / the overlay / the model / the warm JIT
                    cache master; and /proc/<pid>/{cmdline,environ,fd,maps,mem,...} of other processes.
                    Opens with no responsible Python frame are exempt (interpreter).
Paths (v5d): every event path and every configured root (P3_AUDIT_* lists, PYTHONPATH, sys.prefix / base_prefix,
sys.path) is compared in os.path.realpath form, so a file is classified the same whichever spelling reaches it
(v5c used abspath: a symlinked venv interpreter made stdlib / venv reads look protected). A symlink in an allowed
dir that points into a protected dir is now classified by its target, i.e. flagged. remove / rmdir / rename act on
the link itself, so only their parent dir is resolved. /proc/<pid>/... secrets are matched on both spellings, and
an event whose two spellings differ carries the abspath one as "raw".
If P3_COUNTERS_DIR is set, the counter module is loaded after the hook (counter runs only, never scored).
"""


def _install():
    import json
    import os
    import re
    import sys

    fifo = os.environ.get("P3_AUDIT_FIFO")
    if not fifo:
        return
    try:
        fd = os.open(fifo, os.O_WRONLY)
    except OSError:
        return  # no reader: the scorer then sees no start record from this process and voids the session
    from functools import lru_cache

    rp = os.path.realpath  # pure Python on 3.12 (lstat / readlink): raises no audit event, so no hook recursion
    here = os.path.dirname(rp(__file__))
    me = os.path.abspath(__file__)  # compared with raw co_filename spellings in who()

    def norm(p):
        return rp(p).rstrip(os.sep) + os.sep

    def roots(var):
        return tuple(norm(p) for p in os.environ.get(var, "").split(":") if p)

    pp = tuple(norm(p) for p in os.environ.get("PYTHONPATH", "").split(":") if p)
    libroots = (norm(sys.prefix), norm(sys.base_prefix))
    shm = rp("/dev/shm") + os.sep  # "/dev/shm/" unless the host links it elsewhere (e.g. /run/shm)
    shm_ok = re.compile("^" + re.escape(shm) + r"(sem\.mp-|psm_|mp-|wrk_|sem\.)")
    allow_w = roots("P3_AUDIT_ALLOW_W") + ("/dev/null", "/proc/", "/sys/", "/dev/nvidia", "/dev/char/", "/dev/pts/",
                                           "/dev/tty")
    prot_w = roots("P3_AUDIT_PROTECT_W")
    code = roots("P3_AUDIT_CODE_ROOTS") + pp + libroots + (norm(here),)
    prot_r = roots("P3_AUDIT_PROTECT_R")
    allow_r = roots("P3_AUDIT_ALLOW_R") + allow_w + pp + libroots + (norm(here),)
    exe_ok = frozenset(("python", "python3", "python3.12", "gcc", "cc", "g++", "c++", "ld", "ld.bfd", "nvcc", "ptxas", "ptxas-blackwell",
                        "cuobjdump", "nvdisasm", "fatbinary", "nvlink", "nvidia-smi", "ldconfig", "ldconfig.real",
                        "uname", "lscpu", "objdump", "taskset", "file"))

    @lru_cache(maxsize=8)
    def _syspath_real(entries, cwd):  # cwd is part of the key: relative entries resolve against it
        return frozenset(rp(x or ".") for x in entries)

    def syspath_real():
        e = tuple(x for x in sys.path if isinstance(x, str))
        cwd = ""
        if any(not x.startswith(os.sep) for x in e):
            try:
                cwd = os.getcwd()
            except OSError:
                pass
        return _syspath_real(e, cwd)

    proc_secret = re.compile(r"^/proc/(\d+)/(cmdline|environ|fd|fdinfo|maps|smaps|mem|cwd|root|exe|map_files|task)")
    cnt = {}
    wflags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
    write = os.write
    dumps = json.dumps
    getpid = os.getpid
    getframe = sys._getframe

    def emit(kind, **kw):
        n = cnt.get(kind, 0) + 1
        cnt[kind] = n
        if n <= 50:
            try:
                write(fd, (dumps(dict(kind=kind, pid=getpid(), **kw), default=str)[:3800] + "\n").encode())
            except Exception:
                pass

    def who(skip_lib):
        """Source file of the first frame outside this hook (and, if skip_lib, outside the stdlib / venv)."""
        try:
            f = getframe(2)
        except ValueError:
            return ""
        while f is not None:
            fn = f.f_code.co_filename
            if fn != me and not fn.startswith("<") and "/importlib/" not in fn:
                a = rp(fn)  # not cached: a cached spelling could be swapped for a symlink later
                if not skip_lib or not a.startswith(libroots):
                    return a
            f = f.f_back
        return ""

    def foreign_proc(s):
        m = proc_secret.match(s)
        return bool(m) and int(m.group(1)) != getpid()

    def extra(p, raw):
        return {"raw": raw} if raw is not None and raw != p else {}

    def check_read(p, kind, raw=None):
        """p: realpath form; raw: the abspath spelling (only for the /proc secret test and the record)."""
        if foreign_proc(p) or (raw is not None and foreign_proc(raw)):
            w = who(True)
            if w:
                emit("read_protected", path=p, by=w, op=kind, **extra(p, raw))
            return
        q = p + os.sep
        if not prot_r or not q.startswith(prot_r) or q.startswith(allow_r):
            return
        if kind == "list" and p in syspath_real():
            return
        w = who(True)
        if w:
            emit("read_protected", path=p, by=w, op=kind, **extra(p, raw))

    def check_write(p, op, raw=None):
        q = p + os.sep
        if q.startswith(allow_w) or p.startswith(allow_w):
            return
        if p.startswith(shm) and shm_ok.match(p):
            return
        emit("write_protected" if q.startswith(prot_w) else "write_outside", path=p, op=op, by=who(False),
             **extra(p, raw))

    def link_entry(raw):
        """remove / rmdir / rename act on the directory entry itself (a symlink is not followed): resolve the
        parent only."""
        d, b = os.path.split(raw)
        return os.path.join(rp(d), b) if b else rp(raw)

    def fdpath(x):
        try:
            return os.readlink(f"/proc/self/fd/{int(x)}")
        except Exception:
            return None

    def hook(ev, args):
        try:
            if ev == "open":
                path, mode, flags = args
                if isinstance(path, int):
                    return
                raw = os.path.abspath(os.fsdecode(path))
                p = rp(raw)
                w = (mode is not None and any(c in mode for c in "wax+")) or (
                    mode is None and isinstance(flags, int) and bool(flags & wflags))
                if w:
                    check_write(p, "open", raw)
                else:
                    if (p.endswith((".py", ".so", ".pyc")) or raw.endswith((".py", ".so", ".pyc"))) \
                            and not p.startswith(code):
                        emit("code_outside", path=p, **extra(p, raw))
                    check_read(p, "read", raw)
            elif ev in ("os.listdir", "os.scandir", "glob.glob"):
                a0 = args[0] if args else "."
                if isinstance(a0, int) or a0 is None:
                    return
                raw = os.path.abspath(os.fsdecode(a0)).split("*")[0].rstrip(os.sep) or "/"
                check_read(rp(raw), "list", raw)
            elif ev == "os.truncate":
                raw = fdpath(args[0]) if isinstance(args[0], int) else os.path.abspath(os.fsdecode(args[0]))
                if raw:  # an fd's link target can be "pipe:[..]" etc.: only absolute targets are resolved
                    check_write(rp(raw) if raw.startswith(os.sep) else raw, "truncate", raw)
            elif ev in ("os.remove", "os.rmdir"):
                if not isinstance(args[0], int):
                    raw = os.path.abspath(os.fsdecode(args[0]))
                    check_write(link_entry(raw), ev, raw)
            elif ev == "os.rename":
                for a in args[:2]:
                    if not isinstance(a, int):
                        raw = os.path.abspath(os.fsdecode(a))
                        check_write(link_entry(raw), ev, raw)
            elif ev in ("subprocess.Popen", "os.exec", "os.posix_spawn", "os.spawn"):
                exe = os.path.basename(os.fsdecode(args[0])) if args and args[0] else ""
                argv = args[1] if len(args) > 1 else None
                a = [os.fsdecode(x) for x in argv] if isinstance(argv, (list, tuple)) else []
                if exe.startswith("python"):
                    if "-c" in a and not any("multiprocessing" in x for x in a):
                        emit("exec", exe=exe, argv=a[:6])
                elif exe not in exe_ok:
                    emit("exec", exe=exe, argv=a[:6])
            elif ev == "os.system":
                emit("exec", cmd=str(args[0])[:200])
            elif ev == "os.fork":
                w = who(True)
                if w:
                    emit("fork", by=w)
            elif ev in ("sys.addaudithook", "sys.settrace", "sys.setprofile"):
                w = who(False)
                if not w.startswith(libroots):
                    emit("hook", event=ev, by=w)
        except Exception:
            pass

    emit("start", ppid=os.getppid(), argv0=(sys.argv[0] if sys.argv else "")[-120:])
    sys.addaudithook(hook)


_install()
del _install

import os as _os  # noqa: E402

if _os.environ.get("P3_COUNTERS_DIR"):
    import p3_counters  # noqa: E402,F401  (counter runs only; harness/counters is then on PYTHONPATH)
del _os
