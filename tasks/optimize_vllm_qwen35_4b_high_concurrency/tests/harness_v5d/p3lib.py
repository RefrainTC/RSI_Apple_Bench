"""P3 shared helpers: start/stop a vLLM server from a stage dir, run one bench, scrape counters.

Run with the pristine venv python and -I (or -P with PYTHONPATH unset): nothing here imports
agent code. Only the *server* process gets the stage's vllm overlay on PYTHONPATH.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

# v5d host portability. Unset (calibration host) = the v5c paths. P3_ROOT replaces the project root; P3_STAGE replaces the local
# staging root that task.json values use (/tmp/taskc_stage/...; rewritten in load_task).
DEFAULT_ROOT = "/opt/rsi/taskc"  # placeholder; set P3_ROOT
DEFAULT_STAGE = "/tmp/taskc_stage"


def _root_env(var: str, default: str) -> str:
    v = os.environ.get(var)
    if not v:
        return default
    v = os.path.normpath(v)
    if not os.path.isabs(v):
        raise ValueError(f"{var} must be an absolute path, got {v!r}")
    return v


ROOT = _root_env("P3_ROOT", DEFAULT_ROOT)
STAGE = _root_env("P3_STAGE", DEFAULT_STAGE)
ROOT_OVERRIDE = ROOT if ROOT != DEFAULT_ROOT else None     # None on the calibration host
STAGE_OVERRIDE = STAGE if STAGE != DEFAULT_STAGE else None  # None on the calibration host
P = Path(ROOT)
HERE = Path(__file__).resolve().parent  # this harness copy
VENV = P / "venv"  # NAS venv (orchestration only when a task is staged to /tmp)
PY = VENV / "bin" / "python"
SITE_VLLM = VENV / "lib" / "python3.12" / "site-packages" / "vllm"


def venv_of(task: dict) -> Path:
    return Path(task.get("venv") or VENV)


def py_of(task: dict) -> Path:
    return venv_of(task) / "bin" / "python"


def site_vllm_of(task: dict) -> Path:
    return venv_of(task) / "lib" / "python3.12" / "site-packages" / "vllm"


def scratch_of(task: dict, label: str, fallback: Path) -> Path:
    """Per-run scratch (overlays, compile caches, TMPDIR): local NVMe when the task is staged."""
    # keyed by task (+ GPU slot) and label: two GPUs running the same label for different tasks must not
    # share overlays or caches (2026-11-23 15:48: gpt-oss aa1 and Qwen-hc aa1 collided in run/aa1)
    sub = task["name"] + (f"_g{task['gpu_slot_override']}" if task.get("gpu_slot_override") else "")
    return Path(task["scratch"]) / sub / label if task.get("scratch") else fallback


def ensure_staged(task: dict) -> None:
    """/tmp is wiped on pod restart: re-run the staging script if the staged inputs are missing."""
    if task.get("stage_script") and not (py_of(task).exists() and Path(task["model_dir"]).is_dir()):
        subprocess.run(["bash", task["stage_script"]], check=True)


def storage_meta(task: dict) -> dict:
    return {"storage": task.get("storage", "nas"), "venv": str(venv_of(task)), "model_dir": task["model_dir"],
            "scratch": task.get("scratch"), "gpus": task.get("gpus"), "cpus": task.get("cpus"),
            "client_cpus": task.get("client_cpus"),
            "caches": "triton+inductor shared on scratch, vllm compile cache per arm" if task.get("scratch")
            else "all caches per arm under results/ (NAS)"}

# Flags the scorer always passes; a submission may not set or override them.
LOCKED_FLAG_NAMES = {
    "--model", "--tokenizer", "--served-model-name", "--host", "--port", "--max-model-len",
    "--gpu-memory-utilization", "--enable-prefix-caching", "--no-enable-prefix-caching",
    "--kv-cache-dtype", "--quantization", "-q", "--dtype", "--tensor-parallel-size", "-tp",
    "--pipeline-parallel-size", "-pp", "--data-parallel-size", "-dp", "--seed",
    "--load-format", "--download-dir", "--hf-overrides", "--override-generation-config",
    "--generation-config", "--revision", "--tokenizer-revision", "--trust-remote-code",
    "--kv-transfer-config", "--kv-events-config",
    "--num-gpu-blocks-override", "--kv-cache-memory", "--kv-cache-memory-bytes",
    "--cpu-offload-gb", "--swap-space", "--calculate-kv-scales", "--logprobs-mode",
    "--max-logprobs", "--api-key", "--chat-template", "--config", "--allowed-local-media-path",
    "--enable-log-requests", "--uds", "--ssl-keyfile", "--ssl-certfile",
}
# 2026-11-23 (coordinator, option A): the Mamba/GDN state dtype flags are no longer locked. An fp16 GDN state
# is invisible to every numerics measurement we have (logprob gate, drift bound, 1024-token decode probe,
# output witness), so a rule against it could only be enforced by a log line the candidate prints itself.
# Orchestrator-only escape hatch for labelled exploration runs (never set on the agent path):
# P3_EXPLORE_ALLOW="--kv-cache-dtype,--mamba-ssm-cache-dtype". Locked args given later win in argparse,
# so an allowed flag is appended after the locked ones.
ENV_ALLOWED_PREFIX = ("VLLM_", "TORCHINDUCTOR_", "NCCL_", "OMP_NUM_THREADS", "CUDA_DEVICE_MAX_CONNECTIONS")
ENV_DENIED = {
    "VLLM_CACHE_ROOT", "VLLM_CONFIG_ROOT", "VLLM_ASSETS_CACHE", "VLLM_HOST_IP", "VLLM_PORT",
    "VLLM_RPC_BASE_PATH", "VLLM_ALLOW_LONG_MAX_MODEL_LEN", "VLLM_USE_MODELSCOPE",
    "VLLM_PLUGINS", "VLLM_ALLOW_RUNTIME_LORA_UPDATING", "VLLM_SERVER_DEV_MODE",
    "VLLM_KV_CACHE_LAYOUT", "VLLM_LOGGING_CONFIG_PATH", "VLLM_DEBUG_DUMP_PATH",
    "VLLM_MARLIN_INPUT_DTYPE",  # W4A8 activation quantization = a numerics change, like --kv-cache-dtype
}


# One-GPU tasks can run on either card (orchestrator runs only: P3_GPU_SLOT=0|1). Each card has its own
# NUMA-free CPU slice for the server and for the bench client; a score is always paired on one card.
DEFAULT_GPU_SLOTS = {"0": {"gpus": "0", "cpus": "0-23", "client_cpus": "24-31"},
                "1": {"gpus": "1", "cpus": "32-55", "client_cpus": "56-63"}}


def _gpu_slots() -> dict:
    """v5d: P3_GPU_SLOTS = JSON in the shape of DEFAULT_GPU_SLOTS (slot name -> {"gpus": one GPU index, "cpus",
    "client_cpus"}; an empty cpus / client_cpus string means no taskset) replaces the the calibration host table; unset = the calibration host."""
    raw = os.environ.get("P3_GPU_SLOTS")
    if not raw:
        return DEFAULT_GPU_SLOTS
    tab = json.loads(raw)
    if not isinstance(tab, dict) or not tab:
        raise ValueError("P3_GPU_SLOTS must be a non-empty JSON object")
    out = {}
    for k, v in tab.items():
        if not (re.fullmatch(r"[A-Za-z0-9_.-]+", str(k)) and isinstance(v, dict)
                and set(v) == {"gpus", "cpus", "client_cpus"} and all(isinstance(x, str) for x in v.values())
                and re.fullmatch(r"\d+", v["gpus"])):
            raise ValueError(f"P3_GPU_SLOTS[{k!r}] = {v!r}: need exactly gpus (one GPU index), cpus, client_cpus "
                             "(strings)")
        out[str(k)] = dict(v)
    return out


GPU_SLOTS = _gpu_slots()


def _rebase(x):
    """v5d: task.json strings under the default staging root (/tmp/taskc_stage) or project root move under P3_STAGE / P3_ROOT
    (only when set and different; nested lists / dicts included, keys untouched)."""
    if isinstance(x, str):
        for old, new in ((DEFAULT_STAGE, STAGE_OVERRIDE), (DEFAULT_ROOT, ROOT_OVERRIDE)):
            if new and (x == old or x.startswith(old + "/")):
                return new + x[len(old):]
        return x
    if isinstance(x, list):
        return [_rebase(y) for y in x]
    if isinstance(x, dict):
        return {k: _rebase(y) for k, y in x.items()}
    return x


def load_task(task_dir: Path) -> dict:
    t = json.loads((task_dir / "task.json").read_text())
    if STAGE_OVERRIDE or ROOT_OVERRIDE:
        t = _rebase(t)
    t["_dir"] = str(task_dir)
    slot = os.environ.get("P3_GPU_SLOT")
    if slot and "," not in t["gpus"]:
        if slot not in GPU_SLOTS:
            raise KeyError(f"P3_GPU_SLOT={slot!r} is not a slot of the GPU slot table {sorted(GPU_SLOTS)} "
                           "(see P3_GPU_SLOTS)")
        t.update(GPU_SLOTS[slot])
        t["gpu_slot_override"] = slot
    return t


def parse_submission(stage: Path) -> tuple[list[str], dict[str, str], list[str]]:
    """Return (extra_args, extra_env, problems). Blank lines and # comments ignored."""
    problems: list[str] = []
    args: list[str] = []
    env: dict[str, str] = {}
    fa = stage / "submission" / "extra_args.txt"
    fe = stage / "submission" / "env.txt"
    if fa.exists():
        for ln in fa.read_text().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            toks = ln.split(None, 1)
            flag = toks[0]
            name = flag.split("=", 1)[0]
            if not name.startswith("-"):
                problems.append(f"extra_args: not a flag: {ln!r}")
                continue
            allow = {x for x in os.environ.get("P3_EXPLORE_ALLOW", "").split(",") if x}
            if name not in allow and (name in LOCKED_FLAG_NAMES or any(name.startswith(x + "=") for x in LOCKED_FLAG_NAMES)):
                problems.append(f"extra_args: locked flag {name}")
                continue
            args.append(flag)
            if len(toks) > 1:
                args.append(toks[1].strip())
    if fe.exists():
        for ln in fe.read_text().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            if "=" not in ln:
                problems.append(f"env: bad line {ln!r}")
                continue
            k, v = ln.split("=", 1)
            k = k.strip()
            allow = {x for x in os.environ.get("P3_EXPLORE_ALLOW", "").split(",") if x}
            if not k.startswith(ENV_ALLOWED_PREFIX) or (k in ENV_DENIED and k not in allow):
                problems.append(f"env: variable not allowed: {k}")
                continue
            env[k] = v.strip()
    return args, env, problems


def rmtree_rw(p: Path):
    def onerr(func, path, exc):
        os.chmod(os.path.dirname(path), 0o755)
        func(path)
    shutil.rmtree(p, onerror=onerr)


def build_overlay(stage: Path, out: Path, site_vllm: Path = SITE_VLLM) -> Path:
    """overlay/vllm = stage repo/vllm (.py etc.) + symlinks to the pristine compiled .so files."""
    if out.exists():
        rmtree_rw(out)
    src = stage / "repo" / "vllm"
    dst = out / "vllm"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                    copy_function=shutil.copyfile)
    for root, dirs, _ in os.walk(dst):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o755)
    os.chmod(dst, 0o755)
    n = 0
    for so in site_vllm.rglob("*.so"):
        rel = so.relative_to(site_vllm)
        tgt = dst / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        if tgt.exists() or tgt.is_symlink():
            tgt.unlink()
        tgt.symlink_to(so)
        n += 1
    (out / "OVERLAY_INFO.txt").write_text(f"from {src}\nso_links {n}\n")
    compile_tree(dst)
    return out


def compile_tree(root: Path) -> None:
    """Write the .pyc files of a tree as the harness user, so a server never needs to write bytecode into its
    overlay or hooks (compileall only compiles; it does not run the code)."""
    subprocess.run([str(PY), "-I", "-m", "compileall", "-q", "-j", "8", str(root)], capture_output=True, timeout=600)


def hooks_dir(task: dict) -> Path:
    """Per-harness copy of audit/ and counters/ for the server's PYTHONPATH, outside the harness tree: the server
    must be able to read it under the S1 sandbox, where $P is not readable."""
    h = hashlib.sha256()
    for f in sorted((HERE / "audit").glob("*.py")) + sorted((HERE / "counters").glob("*.py")):
        h.update(f.name.encode() + b"\0" + f.read_bytes())
    base = Path(task["scratch"]) if task.get("scratch") else HERE.parent / "run"
    d = base / "hooks" / h.hexdigest()[:16]
    if not (d / "READY").exists():
        tmp = d.with_name(d.name + f".tmp{os.getpid()}")
        if tmp.exists():
            rmtree_rw(tmp)
        for sub in ("audit", "counters"):
            shutil.copytree(HERE / sub, tmp / sub, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            compile_tree(tmp / sub)
        (tmp / "READY").write_text("ok\n")
        for dp, dns, fns in os.walk(tmp):
            os.chmod(dp, 0o755)
            for n in fns:
                os.chmod(os.path.join(dp, n), 0o644)
        try:
            os.rename(tmp, d)
        except OSError:  # another scorer made it first
            rmtree_rw(tmp)
    return d


class AuditFifo:
    """The audit hook writes to a FIFO; this reader (the scorer) appends what arrives to <audit_dir>/audit.jsonl.
    A writer cannot truncate or rewrite a FIFO, so recorded events survive the candidate (v4 review M3)."""

    def __init__(self, fifo: Path, audit_dir: Path):
        import select
        self._select = select.select
        audit_dir.mkdir(parents=True, exist_ok=True)
        fifo.parent.mkdir(parents=True, exist_ok=True)
        if fifo.exists():
            fifo.unlink()
        os.mkfifo(fifo, 0o600)
        os.chmod(fifo, 0o622)  # S1 sandbox: other users may write, nobody but the scorer reads
        self.path = fifo
        self.fd = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)  # RDWR: never EOF while servers come and go
        self.out = open(audit_dir / "audit.jsonl", "ab")
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _drain(self):
        while True:
            try:
                b = os.read(self.fd, 1 << 16)
            except BlockingIOError:
                return
            if not b:
                return
            self.out.write(b)
            self.out.flush()

    def _run(self):
        while not self._stop.is_set():
            r, _, _ = self._select([self.fd], [], [], 0.5)
            if r:
                self._drain()

    def close(self):
        self._stop.set()
        self._t.join(5)
        self._drain()
        self.out.close()
        os.close(self.fd)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


# ---- S1 sandbox: candidate and baseline servers under their own uid (REVIEW_v4 S1). Needs passwordless sudo for
# `setpriv`; OFF unless task.json has "sandbox": true. Not used on the calibration host as of 2026-11-23 (the operator has not
# approved sudo); kept behind this one switch so it can be enabled and validated later.
SBX_UID_BASE = 62000


def sandbox_uid(task: dict) -> int | None:
    if not task.get("sandbox"):
        return None
    slot = task.get("gpu_slot_override") or task["gpus"].split(",")[0]
    return SBX_UID_BASE + int(slot)


def as_uid(uid: int, argv: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["sudo", "-n", "/usr/bin/setpriv", f"--reuid={uid}", f"--regid={uid}", "--clear-groups",
                           "--no-new-privs"] + argv, **kw)


def uid_pids(uid: int) -> list[int]:
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/status") as f:
                for ln in f:
                    if ln.startswith("Uid:"):
                        if int(ln.split()[1]) == uid:
                            out.append(int(d))
                        break
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return out


def pid_cmd(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ")[:200].decode(errors="replace")
    except Exception:
        return ""


SWEEP_ROOTS = ("/tmp", "/var/tmp", "/dev/shm", "/run/lock")


def uid_sweep(uid: int) -> list[str]:
    """Delete every file of the sandbox uid in the world-writable places (and top-level ~, which is
    world-writable on the calibration host): no state survives a session. Returns what was there (outside the sandbox work root)."""
    r = as_uid(uid, ["/usr/bin/find", *SWEEP_ROOTS, "-xdev", "-user", str(uid), "-print"],
               capture_output=True, text=True, timeout=300)
    r2 = as_uid(uid, ["/usr/bin/find", os.path.expanduser("~"), "-maxdepth", "1", "-user", str(uid), "-print"],
                capture_output=True, text=True, timeout=120)
    found = [x for x in (r.stdout + r2.stdout).splitlines() if x]
    if found:
        as_uid(uid, ["/usr/bin/find", *SWEEP_ROOTS, "-xdev", "-depth", "-user", str(uid), "-delete"],
               capture_output=True, timeout=300)
        tops = [x for x in r2.stdout.splitlines() if x and x != os.path.expanduser("~")]
        if tops:
            as_uid(uid, ["/bin/rm", "-rf", "--", *tops], capture_output=True, timeout=300)
    return found


def uid_kill_all(uid: int, sig: str = "KILL") -> None:
    as_uid(uid, ["/bin/kill", f"-{sig}", "-1"], capture_output=True, timeout=30)


def coder_procs() -> dict[int, dict]:
    """pid -> {ppid, start (clock ticks), cmd} for this user's processes."""
    me, out = os.getuid(), {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            if os.stat(f"/proc/{d}").st_uid != me:
                continue
            with open(f"/proc/{d}/stat") as f:
                st = f.read()
            rest = st[st.rfind(")") + 2:].split()
            if rest[0] in ("Z", "X"):  # zombies / dead: already gone
                continue
            out[int(d)] = {"ppid": int(rest[1]), "start": int(rest[19]), "cmd": pid_cmd(int(d))}
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, IndexError):
            continue
    return out


def environ_has(pid: int, needles: tuple[bytes, ...]) -> bool:
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            e = f.read()
        return any(n in e for n in needles)
    except Exception:
        return False


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def gpu_mem_used(gpu: str) -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=20).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return -1


def wait_gpu_free(gpus: str, limit_mib: int = 1500, timeout: float = 600) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if all(0 <= gpu_mem_used(g) < limit_mib for g in gpus.split(",")):
            return True
        time.sleep(3)
    return False


class Server:
    def __init__(self, task: dict, stage: Path, overlay: Path, log_path: Path,
                 cache_root: Path, label: str, audit_dir: Path | None = None, run_tag: str | None = None,
                 private_cache: bool = False):
        self.task, self.stage, self.label = task, stage, label
        self.log_path = log_path
        self.port = free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.proc: subprocess.Popen | None = None
        self.peak_mib: dict[str, int] = {}
        self.killed_reason: str | None = None
        self._stop = threading.Event()
        args, env_extra, problems = parse_submission(stage)
        if problems:
            raise RuntimeError("submission rejected: " + "; ".join(problems))
        self.extra_args, self.env_extra = args, env_extra
        self.overlay = overlay
        self.cache_root = cache_root
        self.counters_dir = None
        self.audit_dir = audit_dir      # runtime audit hook on (scoring / dev runs) when set
        self.run_tag = run_tag or f"p3_{label}_{self.port}_{int(time.time())}"
        # private_cache: this server gets its own copy of the warm Triton / inductor cache (copied from the
        # task's shared master on first use of cache_root), so nothing it writes can reach another arm or run
        self.private_cache = private_cache
        self.other_gpu: list[dict] = []
        self.lingering: list[dict] = []
        self.sbx_uid = sandbox_uid(task)
        self.sbx_files: list[str] = []
        self._fifo: AuditFifo | None = None
        self._procs0: dict[int, dict] = {}

    def cmd(self) -> list[str]:
        t = self.task
        c = [str(py_of(t)), "-P", "-m", "vllm.entrypoints.cli.main", "serve", t["model_dir"],
             "--served-model-name", t["served_name"], "--host", "127.0.0.1", "--port", str(self.port)]
        c += list(t["locked_args"]) + list(t.get("base_args", [])) + self.extra_args
        cpus = t.get("cpus")
        if cpus:
            c = ["taskset", "-c", cpus] + c
        return c

    def _master(self) -> Path | None:
        t = self.task
        if not t.get("scratch"):
            return None
        return Path(t.get("jit_master") or (Path(t["scratch"]) / "shared_cache"))

    def env(self) -> dict:
        t = self.task
        if self.sbx_uid is not None:  # S1: everything the server writes lives under one sandbox-owned dir
            self.tmpdir = Path(t["scratch"]) / "sbx" / f"{self.sbx_uid}_{self.port}" / "t"
        else:
            self.tmpdir = (Path(t["scratch"]) if t.get("scratch") else P) / "t" / str(self.port)
        hooks = hooks_dir(t)
        e = {
            "PATH": f"{venv_of(t)}/bin:/usr/local/bin:/usr/bin:/bin",
            "HOME": str(self.cache_root / "home"),
            "USER": "p3srv", "LOGNAME": "p3srv",
            "TMPDIR": str(self.tmpdir),
            "CUDA_VISIBLE_DEVICES": t["gpus"],
            "PYTHONPATH": ":".join(
                ([str(hooks / "audit")] if self.audit_dir else [])
                + ([str(hooks / "counters")] if self.counters_dir else [])
                + [str(self.overlay)]),
            "VLLM_CACHE_ROOT": str(self.cache_root / "vllm"),
            # Triton / inductor caches: a private copy of the warm master per server when private_cache
            "TRITON_CACHE_DIR": str(self._jit_cache() / "triton"),
            "TORCHINDUCTOR_CACHE_DIR": str(self._jit_cache() / "inductor"),
            "P3_RUN_TAG": self.run_tag,
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
            "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
            "LD_LIBRARY_PATH": os.environ.get("P3_LD_LIBRARY_PATH", ""),
        }
        if self.sbx_uid is None:
            # zmq ipc paths must stay < 107 chars, so TMPDIR is short: <scratch>/t/<port>
            self.tmpdir.mkdir(parents=True, exist_ok=True)
            for k in ("home", "vllm"):
                (self.cache_root / k).mkdir(parents=True, exist_ok=True)
            for k in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
                Path(e[k]).mkdir(parents=True, exist_ok=True)
        if self.counters_dir:
            e["P3_COUNTERS_DIR"] = str(self.counters_dir)
        if self.audit_dir:
            own = [self.cache_root, self.tmpdir, self.overlay] + ([self.counters_dir] if self.counters_dir else [])
            master = [self._master()] if self._master() else []
            scratch = Path(t["scratch"]) if t.get("scratch") else P
            e.update({
                "P3_AUDIT_FIFO": str(self._fifo_path()),
                "P3_AUDIT_ALLOW_W": ":".join(map(str, own)),
                "P3_AUDIT_PROTECT_W": ":".join(map(str, [P, venv_of(t), Path(t["model_dir"]).parent, scratch])),
                "P3_AUDIT_PROTECT_R": ":".join(map(str, [os.path.expanduser("~"), scratch, "/tmp", "/var/tmp", "/dev/shm",
                                                          "/run/lock", "/root"])),
                "P3_AUDIT_ALLOW_R": ":".join(map(str, own + [Path(t["model_dir"]), venv_of(t), hooks] + master)),
                "P3_AUDIT_CODE_ROOTS": ":".join(map(str, own + master + [hooks])),
            })
        e.update(t.get("base_env", {}))
        e.update(self.env_extra)
        for k in [k for k in self.env_extra if k.startswith("P3_")]:  # never from a submission (gate also rejects)
            e.pop(k, None)
        return e

    def _fifo_path(self) -> Path:
        base = Path(self.task["scratch"]) if self.task.get("scratch") else P / "run"
        return base / "fifo" / f"audit_{self.port}"

    def _jit_cache(self) -> Path:
        shared = self._master() or self.cache_root
        return self.cache_root / "jit" if self.private_cache else shared

    def _prepare_private_cache(self):
        if self.sbx_uid is not None:
            return  # copied by the sandbox uid itself in _sandbox_dirs
        if not self.private_cache or (self.cache_root / "jit").exists():
            return
        master = self._master()
        if master and master.is_dir():
            shutil.copytree(master, self.cache_root / "jit", symlinks=False)
        (self.cache_root / "jit").mkdir(parents=True, exist_ok=True)

    def _sandbox_dirs(self, env: dict):
        """S1: the sandbox uid creates its own dirs (0700) under <scratch>/sbx (1777) and copies the JIT master."""
        sbx = Path(self.task["scratch"]) / "sbx"
        sbx.mkdir(parents=True, exist_ok=True)
        os.chmod(sbx, 0o1777)
        dirs = [self.tmpdir, Path(env["HOME"]), Path(env["VLLM_CACHE_ROOT"])]
        sh = "umask 077; mkdir -p " + " ".join(f"'{d}'" for d in dirs)
        if self.private_cache and self._master() and self._master().is_dir():
            sh += f" && cp -a '{self._master()}' '{self.cache_root / 'jit'}'"
        sh += f" && mkdir -p '{env['TRITON_CACHE_DIR']}' '{env['TORCHINDUCTOR_CACHE_DIR']}'"
        r = as_uid(self.sbx_uid, ["/bin/sh", "-c", sh], capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"sandbox setup failed: {r.stderr[-300:]}")

    def start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.log_path, "w")
        self._fp.write("# cmd: " + " ".join(self.cmd()) + "\n# env_extra: " + json.dumps(self.env_extra) + "\n")
        self._fp.flush()
        if self.sbx_uid is not None:
            # cache_root must be sandbox-writable: move it under the sandbox root (callers read it back only via
            # the paths in env, never directly)
            self.cache_root = Path(self.task["scratch"]) / "sbx" / f"{self.sbx_uid}_{self.port}"
            stale = uid_pids(self.sbx_uid)
            if stale:  # field not clean: a previous session left processes
                uid_kill_all(self.sbx_uid)
                self.lingering += [{"pid": p, "cmd": pid_cmd(p), "when": "before_start"} for p in stale]
            self.sbx_files += uid_sweep(self.sbx_uid)
        self._prepare_private_cache()
        if self.audit_dir:
            self._fifo = AuditFifo(self._fifo_path(), self.audit_dir)
        env = self.env()
        argv = self.cmd()
        if self.sbx_uid is not None:
            self._sandbox_dirs(env)
            argv = ["sudo", "-n", "/usr/bin/setpriv", f"--reuid={self.sbx_uid}", f"--regid={self.sbx_uid}",
                    "--clear-groups", "--no-new-privs", "/usr/bin/env", "-i"] + [f"{k}={v}" for k, v in env.items()] \
                + ["/bin/sh", "-c", 'umask 077; cd "$0" && exec "$@"', str(self.tmpdir)] + argv
            env = {"PATH": "/usr/bin:/bin"}
        self._procs0 = coder_procs()
        _LIVE.add(self)  # v5d: from here until the end of stop(), a SIGINT / SIGTERM handler can kill this server
        self.proc = subprocess.Popen(argv, stdout=self._fp, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True,
                                     cwd=str(self.tmpdir) if self.sbx_uid is None else "/")
        self._wd = threading.Thread(target=self._watchdog, daemon=True)
        self._wd.start()

    def _watchdog(self):
        lim = int(self.task.get("gpu_mem_kill_mib", 22300))
        k = 0
        while not self._stop.is_set():
            for g in self.task["gpus"].split(","):
                m = gpu_mem_used(g)
                self.peak_mib[g] = max(self.peak_mib.get(g, 0), m)
                if m > lim:
                    self.killed_reason = f"gpu{g} {m} MiB > {lim}"
                    self.kill()
                    return
            k += 1
            if k % 5 == 0:
                self._check_other_gpus()
            self._stop.wait(1.0)

    def _mine(self, pid: int) -> bool:
        """A process of this server: the sandbox uid, or (same-user mode) this run's tag in its environment."""
        if self.sbx_uid is not None:
            return pid in set(uid_pids(self.sbx_uid))
        return environ_has(pid, (f"P3_RUN_TAG={self.run_tag}".encode(),))

    def _check_other_gpus(self):
        """Processes of this server holding a context on a GPU that is not the task's. v5d: every GPU nvidia-smi
        lists (v5c asked for cards 0 and 1 only, which is all the calibration host has, so the calibration host sees the same lists)."""
        try:
            idx = {}
            for ln in subprocess.run(["nvidia-smi", "--query-gpu=uuid,index", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=20).stdout.strip().splitlines():
                f = [x.strip() for x in ln.split(",")]
                if len(f) == 2 and f[0] and f[1].isdigit():  # skip error / N/A lines of a GPU nvidia-smi cannot read
                    idx[f[0]] = f[1]
            apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=20).stdout
        except Exception:
            return
        mine = set(self.task["gpus"].split(","))
        for ln in apps.strip().splitlines():
            try:
                pid, uuid = [x.strip() for x in ln.split(",")]
                if idx.get(uuid) in mine:
                    continue
                if self._mine(int(pid)):
                    self.other_gpu.append({"pid": int(pid), "gpu": idx.get(uuid)})
            except Exception:
                continue

    def wait_ready(self, timeout: float = 1200) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(self.base_url + "/v1/models", timeout=5) as r:
                    if r.status == 200:
                        self.ready_s = time.time() - t0
                        return True
            except Exception:
                pass
            time.sleep(2)
        return False

    def kill(self):
        if self.proc and self.proc.poll() is None:
            if self.sbx_uid is not None:
                uid_kill_all(self.sbx_uid, "TERM")
            else:
                try:
                    os.killpg(self.proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                if self.sbx_uid is not None:
                    uid_kill_all(self.sbx_uid, "KILL")
                else:
                    try:
                        os.killpg(self.proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                self.proc.wait(30)

    def stop(self):
        self._stop.set()
        self.kill()
        try:
            self._fp.close()
        except Exception:
            pass
        if self.sbx_uid is not None:
            left = uid_pids(self.sbx_uid)
            self.lingering += [{"pid": p, "cmd": pid_cmd(p)} for p in left]
            uid_kill_all(self.sbx_uid)
            for _ in range(20):
                if not uid_pids(self.sbx_uid):
                    break
                time.sleep(0.5)
            if self.counters_dir:
                src = Path(self.counters_dir)
                r = as_uid(self.sbx_uid, ["/bin/tar", "-C", str(src.parent), "-cf", "-", src.name],
                           capture_output=True, timeout=120)
                if r.returncode == 0:
                    subprocess.run(["tar", "-C", str(src.parent), "-xf", "-"], input=r.stdout, timeout=120)
            self.sbx_files += uid_sweep(self.sbx_uid)
        else:
            self.lingering += kill_tagged(self.run_tag)
            time.sleep(1.0)  # let killed helpers be reaped before the orphan census
            self.lingering += self._orphans()
        if self._fifo:
            self._fifo.close()
        _LIVE.discard(self)

    def _emergency_kill(self, grace: float = 10.0) -> None:
        """v5d, signal path only: the server's process group gets SIGTERM, then SIGKILL if its leader is still up
        after `grace` s (as kill() does), and every process with this run's tag is killed (as stop() does). Leaves
        the log, the FIFO and the lingering census to stop(), which the callers' finally blocks still run."""
        self._stop.set()
        if self.sbx_uid is not None:
            uid_kill_all(self.sbx_uid, "TERM")
            t0 = time.time()
            while uid_pids(self.sbx_uid) and time.time() - t0 < grace:
                time.sleep(0.2)
            uid_kill_all(self.sbx_uid, "KILL")
            return
        pr = self.proc
        if pr is not None and pr.poll() is None:
            try:
                os.killpg(pr.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                pr.wait(grace)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(pr.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    pr.wait(10)
                except subprocess.TimeoutExpired:
                    pass
        kill_tagged(self.run_tag)

    def _orphans(self) -> list[dict]:
        """Same-user mode: processes started during this session that are now orphans (parent = init) or carry
        this run's overlay / audit FIFO in their environment, i.e. something the server detached and that
        cleared its tag (REVIEW_v4 4.1 (c)). Killed when they carry a marker of this run, reported always."""
        found = []
        needles = (str(self.overlay).encode(), str(self._fifo_path()).encode(), str(self.tmpdir).encode())
        for pid, p in coder_procs().items():
            if pid in self._procs0 and self._procs0[pid]["start"] == p["start"]:
                continue
            marked = environ_has(pid, needles)
            if not (marked or p["ppid"] == 1):
                continue
            if marked:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            found.append({"pid": pid, "cmd": p["cmd"], "orphan": p["ppid"] == 1, "marked": marked})
        return found


# ---- v5d: live-server registry and SIGINT / SIGTERM cleanup. Each server runs in its own session
# (start_new_session=True), so a signal to the scorer never reached it: v5c left vLLM running after a SIGTERM
# (default action: die without finally blocks) or a SIGINT between start() and the caller's try.
_LIVE: set = set()  # Servers between start() (just before Popen) and the end of stop()
_SIG = {"busy": False}


def _kill_live(grace: float) -> None:
    for s in list(_LIVE):
        try:
            s._emergency_kill(grace)
        except Exception:
            pass


def _on_signal(signum, frame):
    if _SIG["busy"]:  # a second signal while cleaning up: kill hard and leave now
        _kill_live(0.0)
        os._exit(128 + signum)
    _SIG["busy"] = True
    print(f"[p3lib] signal {signum}: stopping {len(_LIVE)} live server(s), exit {128 + signum}", file=sys.stderr,
          flush=True)
    _kill_live(10.0)
    raise SystemExit(128 + signum)  # finally blocks (Server.stop, logs) still run; exit code 130 / 143


def install_signal_handlers() -> None:
    """Entry scripts call this once (main thread). A signal that is ignored at start (SIGINT of a background job)
    stays ignored."""
    for sig in (signal.SIGINT, signal.SIGTERM):
        if signal.getsignal(sig) is not signal.SIG_IGN:
            signal.signal(sig, _on_signal)


def kill_tagged(tag: str) -> list[dict]:
    """Processes still alive after the server's process group was stopped that carry this run's tag in their
    environment (a candidate that detached a helper: setsid, double fork). Killed and reported."""
    found = []
    me = os.getuid()
    needle = f"P3_RUN_TAG={tag}".encode()
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            if os.stat(f"/proc/{d}").st_uid != me:
                continue
            with open(f"/proc/{d}/environ", "rb") as f:
                if needle not in f.read().split(b"\0"):
                    continue
            with open(f"/proc/{d}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ")[:200].decode(errors="replace")
            os.kill(int(d), signal.SIGKILL)
            found.append({"pid": int(d), "cmd": cmd})
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found


def tree_digest(roots: list[Path]) -> dict[str, str]:
    """Per root: sha256 over (relative path, size, mtime_ns, ctime_ns) of every file and link. ctime cannot be
    set back, so any write, chmod or replace shows. ~0.3 s for the 52k-file venv on NVMe."""
    out = {}
    for r in roots:
        h = hashlib.sha256()
        r = Path(r)
        if not r.exists():
            out[str(r)] = "missing"
            continue
        for dp, dns, fns in os.walk(r, followlinks=False):
            # __pycache__ is skipped: trusted processes write it on first import. A planted .pyc is a write
            # under a protected tree, which the audit hook reports.
            dns[:] = sorted(d for d in dns if d != "__pycache__")
            for n in sorted(fns) + sorted(x for x in dns if os.path.islink(os.path.join(dp, x))):
                p = os.path.join(dp, n)
                try:
                    st = os.lstat(p)
                except FileNotFoundError:
                    continue
                h.update(f"{os.path.relpath(p, r)}\0{st.st_size}\0{st.st_mtime_ns}\0{st.st_ctime_ns}\n".encode())
        out[str(r)] = h.hexdigest()
    return out


def integrity_roots(task: dict) -> list[Path]:
    """What a candidate server must never change: the venv (incl. the bench client), the model, the harness,
    the task's hidden scoring files and the warm JIT cache master."""
    roots = [venv_of(task), Path(task["model_dir"]), HERE, Path(task["_dir"]) / "scoring"]
    if task.get("scratch"):
        roots.append(Path(task.get("jit_master") or (Path(task["scratch"]) / "shared_cache")))
        roots.append(hooks_dir(task))
    return roots


def audit_events(audit_dir: Path) -> list[dict]:
    """All records of a session (v5: one audit.jsonl written by the FIFO reader; v4: audit.<pid>.jsonl)."""
    ev = []
    if audit_dir and Path(audit_dir).is_dir():
        for f in sorted(Path(audit_dir).glob("audit*.jsonl")):
            for ln in f.read_text(errors="replace").splitlines():
                try:
                    ev.append(json.loads(ln))
                except Exception:
                    ev.append({"kind": "unparsable", "line": ln[:200]})
    return ev


def audit_verdict(events: list[dict], min_procs: int = 2) -> tuple[list[dict], dict]:
    """(flags, summary). Fail closed: fewer than min_procs processes with a start record = audit_missing (hook
    not loaded or disabled before it could report). Every non-start record is an audit event."""
    starts = {e.get("pid") for e in events if e.get("kind") == "start"}
    bad = [e for e in events if e.get("kind") != "start"]
    counts: dict = {}
    for e in bad:
        counts[e.get("kind")] = counts.get(e.get("kind"), 0) + 1
    flags = []
    if len(starts) < min_procs:
        flags.append({"kind": "audit_missing", "detail": {"processes_with_start": len(starts), "need": min_procs}})
    if bad:
        flags.append({"kind": "audit", "detail": {"counts": counts, "first": bad[:3]}})
    return flags, {"processes": len(starts), "events": counts}


def scrape_metrics(base_url: str) -> dict:
    """Counters that show which mechanism ran (engine step sizes, preemptions, token totals)."""
    out: dict = {}
    try:
        with urllib.request.urlopen(base_url + "/metrics", timeout=10) as r:
            text = r.read().decode()
    except Exception as e:
        return {"error": str(e)}
    for ln in text.splitlines():
        if ln.startswith("#"):
            continue
        m = re.match(r'^(vllm:[a-z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)$', ln)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", float(m.group(3))
        if name.endswith("_created"):
            continue
        keep = ("iteration_tokens_total", "num_preemptions", "prompt_tokens_total",
                "generation_tokens_total", "request_success_total", "spec_decode",
                "num_requests_running", "kv_cache_usage", "gpu_cache_usage")
        if not any(k in name for k in keep):
            continue
        le = re.search(r'le="([^"]+)"', labels)
        key = name + (f"[le={le.group(1)}]" if le else "")
        if "finished_reason" in labels:
            fr = re.search(r'finished_reason="([^"]+)"', labels)
            key = name + f"[{fr.group(1)}]"
        out[key] = out.get(key, 0.0) + val
    return out


def metrics_delta(a: dict, b: dict) -> dict:
    return {k: round(b[k] - a.get(k, 0.0), 3) for k in b if isinstance(b.get(k), float)
            and "running" not in k and "usage" not in k and b[k] - a.get(k, 0.0) != 0}


def bench_argv(task: dict, base_url: str, wl: dict, prof: dict | None, seed: int, num_prompts: int,
               output_len: int | None = None, conc: int | None = None) -> list[str]:
    """`vllm bench serve` arguments for one run (unique prompts per seed, ignore_eos). Random-token prompts by
    default; a workload with "dataset": {"name": "sharegpt", "path": ...} uses natural-text prompts drawn from
    that file (the client shuffles it with the seed) with a fixed output length."""
    if conc is None:
        conc = int((prof or {}).get("concurrency", wl.get("max_concurrency", 1)))
    ds = wl.get("dataset")
    if ds:
        assert ds.get("name") == "sharegpt", ds
        data = ["--dataset-name", "sharegpt", "--dataset-path", str(ds["path"]),
                "--sharegpt-output-len", str(output_len or wl["output_len"])]
    else:
        data = ["--dataset-name", "random", "--random-input-len", str(wl["input_len"]),
                "--random-output-len", str(output_len or wl["output_len"]),
                "--random-range-ratio", str(wl.get("range_ratio", 0.0))]
    a = ["--backend", "vllm", "--model", task["served_name"], "--tokenizer", task["model_dir"],
         "--base-url", base_url, "--endpoint", "/v1/completions"] + data + [
         "--num-prompts", str(num_prompts), "--max-concurrency", str(conc), "--request-rate", "inf",
         "--ignore-eos", "--seed", str(seed), "--temperature", "1.0", "--top-p", "1.0", "--top-k", "0",
         "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,99", "--disable-tqdm"]
    if wl.get("prefix_len"):
        a += ["--random-prefix-len", str(wl["prefix_len"])]
    return a


def run_driver(task: dict, base_url: str, jobs: list[dict], spec_path: Path, log_path: Path,
               timeout: int = 2400) -> list[dict | None]:
    """Run all jobs in one bench_driver.py process; return per job the result line the driver printed on
    its stdout (a pipe owned by the scorer; None = failed run). The spec goes in on stdin; spec_path and
    log_path are only records."""
    spec = json.dumps({"base_url": base_url, "jobs": jobs})
    spec_path.write_text(spec)
    cmd = [str(py_of(task)), "-I", str(HERE / "bench_driver.py")]
    if task.get("client_cpus"):
        cmd = ["taskset", "-c", task["client_cpus"]] + cmd
    env = {"PATH": f"{venv_of(task)}/bin:/usr/bin:/bin", "HOME": str(spec_path.parent), "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost",
           "VLLM_NO_USAGE_STATS": "1"}
    try:
        cp = subprocess.run(cmd, env=env, input=spec, capture_output=True, text=True, timeout=timeout,
                            cwd=str(spec_path.parent))
        txt = cp.stdout + cp.stderr
    except subprocess.TimeoutExpired as e:
        txt = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        txt += "\n[run_driver] TIMEOUT\n"
    log_path.write_text(txt)
    got = {}
    for ln in txt.splitlines():
        if ln.startswith("[driver-result] "):
            try:
                d = json.loads(ln[len("[driver-result] "):])
                got[d["out"]] = d
            except Exception:
                pass
    res = []
    for j in jobs:
        d = got.get(j["out"])
        if d is not None and (d.get("_err") or not d.get("_has_result")):
            d = d if d.get("_has_result") else None
        res.append(d)
    return res


def result_tamper(results: list[dict | None]) -> list[str]:
    """Result files whose content changed after the driver wrote them (the scorer never reads them)."""
    bad = []
    for d in results:
        if not d:
            continue
        p = Path(d["out"])
        try:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
        except FileNotFoundError:
            h = "missing"
        if h != d.get("_sha256"):
            bad.append(p.name)
    return bad


PRIMARY = {"ttft_ms": ("median_ttft_ms", True), "tpot_ms": ("median_tpot_ms", True),
           "request_throughput": ("request_throughput", False), "output_throughput": ("output_throughput", False)}


def primary_score(d: dict, metric: str) -> float | None:
    key, lower = PRIMARY[metric]
    v = d.get(key)
    if not v or v <= 0:
        return None
    if d.get("completed", 0) < d.get("num_prompts", d.get("completed", 0)):
        return None
    return 1000.0 / v if lower else float(v)
