# tasks/iso13365 (operator notes; not part of the agent's view)

ISO-Bench vLLM #13365, "[V1][Spec Decode] Optimize N-gram matching with Numba". Parent c8d70e24 (0.7.3.dev215),
fix 4c822298 (dev216). The fix moves `NgramProposer`'s KMP search from a Python loop over numpy scalars to two
`@numba.jit(nopython=True)` functions and JIT-compiles them once in `GPUModelRunner.__init__` (np.zeros(1024, int32)).

- Venv: `13365_base` = the parent venv + numba 0.60.0 / llvmlite 0.43.0 copied from the fix venv (the parent venv had no
  numba, so the fix could not run as an overlay on it). Parent role = base venv, no overlay (numba installed, unused).
- Harness: `harness_iso13365/` (copy of harness/, see its CHANGES_iso13365.md). Ops scripts: `ops_iso13365/`.
- Hosts: main GPU1 (CCD1 8-15,40-47, /tmp/p2) and n1 GPU1 (CCD3 24-31,56-63, /tmp/ag13365); task.env picks by hostname.
- Commands: `bash ops_iso13365/calib.sh 1`, `bash ops_iso13365/score_fix.sh <label> 1`, `bash ops_iso13365/transfer.sh <label> 1`,
  `bash ops_iso13365/make_canaries.sh` + `bash ops_iso13365/score_ws.sh <label> tasks/iso13365/canaries/<C> 1`,
  `python3 -I ops_iso13365/jobs_summary.py tasks/iso13365 <labels...>`.
