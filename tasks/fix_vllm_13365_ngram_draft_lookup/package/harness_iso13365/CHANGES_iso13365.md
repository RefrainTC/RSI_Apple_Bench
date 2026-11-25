# harness_iso13365: what differs from harness/ (copied 2026-11-24 08:0x UTC; harness/ itself is untouched)

- Every `H=/home/coder/work/p2-port/harness` points to `harness_iso13365` (calibrate, score, score_phase2, pairs, snapshot_gate, ...).
- Local NVMe root is `$P2T` (task.env: main `/tmp/p2`, n1 `/tmp/ag13365`) instead of a hard-coded `/tmp/p2`: compile caches,
  staged-venv markers, per-run venv copies, audit protect/allow lists.
- CPU pinning: `P2_CPUS_OVERRIDE` (set by task.env on n1 = CCD3 24-31,56-63) wins over the main pod's table in calibrate.sh
  and integrity.sh (p2_job.sh already honoured it).
- GPU lock: host-local `$P2T/locks/iso13365_gpu<N>.lock` (the shared NFS `locks/gpu1.lock` would serialise main GPU1 and n1 GPU1).
- `p2_job.sh` is a verbatim copy of scripts/p2_job.sh (run_variant.sh calls this copy); `jobvar.py` copied from scripts/.
- integrity.sh: scorer manifest hashes harness_iso13365/ (+ scripts/p2_cpus.sh, the pinning table, the task files);
  the venv `home =` lines are compared with `$P2T` normalised to `<P2T>`, so main and n1 share one manifest.
- kernel_gate.py: `numba` is no longer rejected as such (the human fix is a Numba CPU JIT). Rejected instead: numba GPU
  targets (`numba.cuda`, `cuda.jit`, `target='cuda'`, roc), native interop (`cfunc`, `carray`, `intrinsic`, `numba.core`,
  `numba.extending`, `voidptr`, `CPointer`); ctypes / cffi / Triton / CUDA rules unchanged.
- run_variant.sh: exits 8 before starting a run when /home/coder/work/p2-port/STOP_iso13365 exists.
