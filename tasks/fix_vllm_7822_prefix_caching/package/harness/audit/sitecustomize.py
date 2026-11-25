"""P2 audit hook (review F4.2). Loaded as sitecustomize in every Python process of a measured job (offline runner,
server, engine subprocesses) for parent and candidate alike, BEFORE any vllm import. It never blocks anything; it records
events a candidate's code could use to tamper with a measurement or to carry state across runs:
  write_protected : a file opened for writing under the harness / task / venv / model trees (P2_AUDIT_PROTECT_W), or
                    under this run's OUT dir by code other than the trusted drivers (P2_AUDIT_TRUSTED, full paths)
  write_outside   : a file opened for writing outside the run's own scratch, its cache dir (P2_AUDIT_ALLOW_W), /dev/shm,
                    /dev/null, /proc, /sys (= a cross-run persistence channel)
  code_outside    : Python source / extension read from outside the venv, the stdlib, PYTHONPATH (overlay + this dir)
                    and P2_AUDIT_CODE_ROOTS (task dir, bench dir, this run's cache + scratch: compiler output)
  exec            : subprocess / os.system / exec / posix_spawn of something other than python (multiprocessing spawn
                    only), a compiler or an NVIDIA / system tool
  hook            : sys.addaudithook / settrace / setprofile after this hook (attempts to interfere with auditing)
  read_protected  : (2026-11-23, leakage) a file opened for READING, or a directory listed, under P2_AUDIT_PROTECT_R
                    (harness, task data, human fix, reference outputs, other runs' stages / overlays / caches), outside
                    P2_AUDIT_ALLOW_R / this run's own dirs / the interpreter, venv and overlay, by code other than the
                    trusted drivers. Exempt: the task's own scorer modules (*.py directly in P2_AUDIT_TASKDIR, read e.g. by
                    linecache for tracebacks) and opens with no Python frame (the interpreter loading its main script).
One JSON line per event (at most 50 per kind and process) goes to $P2_AUDIT_OUT/audit.<pid>.jsonl through an fd opened
here (os.write, no further open events); the main process adds an "end" record at exit. The harness (audit_check.py)
flags every event, and a missing "end" when P2_AUDIT_NEED_END=1 (offline drivers)."""
import os, sys
_OUT = os.environ.get("P2_AUDIT_OUT")
if _OUT:
    import atexit, json
    _fd = os.open(os.path.join(_OUT, f"audit.{os.getpid()}.jsonl"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    _here = os.path.dirname(os.path.abspath(__file__))
    def _norm(p):
        return os.path.abspath(p).rstrip(os.sep) + os.sep
    def _roots(var):
        return tuple(_norm(p) for p in os.environ.get(var, "").split(":") if p)
    _ALLOW_W = _roots("P2_AUDIT_ALLOW_W") + ("/dev/shm/", "/dev/null", "/proc/", "/sys/", "/dev/nvidia", "/dev/char/")
    _PROT_W = _roots("P2_AUDIT_PROTECT_W")
    _RUNOUT = _norm(os.environ.get("R4_OUT", _OUT))
    _CODE = (_roots("P2_AUDIT_CODE_ROOTS") + tuple(_norm(p) for p in os.environ.get("PYTHONPATH", "").split(":") if p)
             + (_norm(sys.prefix), _norm(sys.base_prefix), _norm(_here)))
    _TRUSTED = frozenset(os.path.abspath(p) for p in os.environ.get("P2_AUDIT_TRUSTED", "").split(":") if p)
    _PROT_R = _roots("P2_AUDIT_PROTECT_R")
    _ALLOW_R = (_roots("P2_AUDIT_ALLOW_R") + _ALLOW_W + (_RUNOUT, _norm(sys.prefix), _norm(sys.base_prefix), _norm(_here))
                + tuple(_norm(p) for p in os.environ.get("PYTHONPATH", "").split(":") if p))
    _TASKDIR = os.path.abspath(os.environ["P2_AUDIT_TASKDIR"]) if os.environ.get("P2_AUDIT_TASKDIR") else None
    _EXE_OK = frozenset(("python", "python3", "python3.10", "gcc", "cc", "g++", "c++", "ld", "nvcc", "ptxas", "cuobjdump",
                         "nvdisasm", "fatbinary", "nvlink", "nvidia-smi", "ldconfig", "ldconfig.real", "uname", "lscpu"))
    _cnt = {}
    _WFLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
    _SELF = os.path.abspath(__file__)
    def _emit(kind, **kw):
        n = _cnt.get(kind, 0) + 1; _cnt[kind] = n
        if n <= 50:
            try:
                os.write(_fd, (json.dumps(dict(kind=kind, pid=os.getpid(), **kw), default=str) + "\n").encode())
            except Exception:
                pass
    def _writer():  # source file of the first frame outside this hook, the import machinery and the io/os wrappers
        f = sys._getframe(2)
        while f is not None:
            fn = f.f_code.co_filename
            if fn != _SELF and not fn.startswith("<") and "/importlib/" not in fn and not fn.endswith(("/os.py", "/io.py", "/codecs.py", "/_pyio.py", "/pathlib.py", "/shutil.py", "/tempfile.py")):
                return os.path.abspath(fn)
            f = f.f_back
        return ""
    def _reader():  # like _writer, but also skips stdlib frames (linecache, warnings, json, ...): the responsible caller
        try:
            f = sys._getframe(2)
        except ValueError:
            return ""
        lib = _norm(os.path.join(sys.base_prefix, "lib"))
        while f is not None:
            fn = f.f_code.co_filename
            if fn != _SELF and not fn.startswith("<") and not os.path.abspath(fn).startswith(lib) and "/site-packages/" not in fn:
                return os.path.abspath(fn)
            f = f.f_back
        return ""
    def _check_read(p, kind):
        if not _PROT_R or not (p + os.sep).startswith(_PROT_R) or (p + os.sep).startswith(_ALLOW_R):
            return
        if kind == "list" and p in {os.path.abspath(x or ".") for x in sys.path if isinstance(x, str)}:
            return   # the import system scanning a sys.path entry (e.g. the driver's own directory), not a lookup
        if _TASKDIR and os.path.dirname(p) == _TASKDIR and p.endswith(".py"):
            return
        who = _reader()
        if who and who not in _TRUSTED:
            _emit("read_protected", path=p, by=who, op=kind)
    def _hook(ev, args):
        if ev in ("os.listdir", "os.scandir", "glob.glob"):
            try:
                a0 = args[0] if args else "."
                if isinstance(a0, int) or a0 is None:
                    return
                _check_read(os.path.abspath(os.fsdecode(a0)).split("*")[0].rstrip(os.sep) or "/", "list")
            except Exception:
                pass
            return
        if ev == "open":
            path, mode, flags = args
            if isinstance(path, int):
                return
            try:
                p = os.path.abspath(os.fsdecode(path))
            except Exception:
                return
            w = (mode is not None and any(c in mode for c in "wax+")) or (mode is None and isinstance(flags, int) and bool(flags & _WFLAGS))
            if w:
                if (p + os.sep).startswith(_RUNOUT) or p.startswith(_RUNOUT):
                    who = _writer()
                    if who not in _TRUSTED:
                        _emit("write_protected", path=p, by=who)
                elif p.startswith(_ALLOW_W):
                    return
                elif p.startswith(_PROT_W):
                    _emit("write_protected", path=p, by=_writer())
                else:
                    _emit("write_outside", path=p, by=_writer())
            else:
                if p.endswith((".py", ".so", ".pyc")) and not p.startswith(_CODE):
                    _emit("code_outside", path=p)
                _check_read(p, "read")
        elif ev in ("subprocess.Popen", "os.exec", "os.posix_spawn", "os.spawn"):
            try:
                exe = os.path.basename(os.fsdecode(args[0])) if args and args[0] else ""
                argv = args[1] if len(args) > 1 else None
                a = [os.fsdecode(x) for x in argv] if isinstance(argv, (list, tuple)) else []
            except Exception:
                exe, a = "?", []
            if exe.startswith("python"):
                if "-c" in a and not any("multiprocessing" in x for x in a):
                    _emit("exec", exe=exe, argv=a[:6])
            elif exe not in _EXE_OK:
                _emit("exec", exe=exe, argv=a[:6])
        elif ev == "os.system":
            _emit("exec", cmd=str(args[0])[:200])
        elif ev in ("sys.addaudithook", "sys.settrace", "sys.setprofile"):
            _emit("hook", event=ev, by=_writer())
    sys.addaudithook(_hook)
    _MAIN = os.getpid()
    def _end():
        if os.getpid() == _MAIN:
            try:
                os.write(_fd, (json.dumps({"kind": "end", "pid": _MAIN, "counts": _cnt}) + "\n").encode())
            except Exception:
                pass
    atexit.register(_end)
