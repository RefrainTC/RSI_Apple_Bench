# Agent kit

The task packages can reproduce the human fix's speedup. This kit adds what you need to score an AI agent's own code
with them on your GPU: a view of the task that hides the fix, a safe way to build the agent's code into a candidate,
a REF measured on your GPU, and one scoring entry point that writes a reward file.

The kit lives in `<package>/kit/`. Its scripts find the package one directory up (or use `ISO_ROOT`) and read its
`env.sh` / `site.env` (STAGE, GPU, CPUS, PORT).

## What each script does

| Script | Purpose | GPU | Time |
|---|---|---|---|
| `make_agent_view.sh <dir>` | Builds the directory the agent gets: a private copy of the unmodified venv, the interpreter, the model and the CUDA tree, the benchmark and the scored loads (comments stripped), and `dev_bench.sh`. It checks that the view's engine is the parent's byte for byte, then scans the view for leaks. | no | ~1 min, ~15 GB |
| `dev_bench.sh` (in the view) | Times the scored load on the agent's tree. It uses the same job wrapper (`harness/job.sh`), environment, CPU pinning and compile-cache layout as the scorer's timed runs. It prints the agent's own metric only: no parent, no fix, no REF. | yes | one timed run (~1 min for sgl16521) |
| `make_candidate.sh <name> <tree or diff>` | Builds `$STAGE/venvs/<pr>_<name>` from the agent's edited engine package (a directory, a whole venv, or a unified diff). It checks the parent first. Only `*.py` may differ. The candidate is a real copy, and it is byte-compiled and marked staged. | no | ~30 s, ~11 GB |
| `set_ref.py <label...>` | Reads finished calibration and fix-scoring jobs from this machine and sets REF and the band in `harness/gate.json`. It also writes `gpu_name` and the date there, and keeps the shipped values under `reference_rtx4090`. | no | seconds |
| `test.sh <tree or diff> [name]` | The verifier. It checks the GPU, the parent, the package and the seal, builds the candidate, runs `harness/run.sh <fresh label> score <name>` while a monitor watches, checks everything again, and writes the reward file. | yes | one scoring job (~9 min for sgl16521) + ~1 min of checks |

`harness/check.py` is unchanged. It ignores the extra keys in `gate.json`.

## Order on a new GPU type

1. **Setup** (RUN.md steps 1-4: venvs, model, CUDA pieces, staging; `export TZ=UTC`). CPU and network, 5-25 min.
2. **Calibrate**: `bash harness/run.sh cal_g0 cal` (~25 min), plus at least one scoring job of the fix:
   `bash harness/run.sh s1_g0 score` (~9 min). Three jobs in total (one calibration and two scorings) give a usable
   spread.
3. **Set REF**: `python3 -I kit/set_ref.py cal_g0 s1_g0 s2_g0`. It refuses jobs from another host, jobs where the fix
   failed a check, mixed GPU types, and fewer than 2 jobs. Use `--dry-run` to print without writing.
4. **Make the view**: `bash kit/make_agent_view.sh /path/outside/the/package`. Give the agent this directory only,
   never the package. The scan report goes to `kit_state/view_scan_<dir>.txt`, outside the view, because it lists
   the terms it searched for.
5. **Score**: `KIT_LOCK=/path/gpu.lock REWARD_FILE=/logs/verifier/reward.json bash kit/test.sh <agent engine dir or diff>`.
   The first `test.sh` run also *seals* the parent (~1-2 min): it byte-compiles the parent venv and the interpreter,
   then records the sha256 of every file of the trusted trees in `kit_state/seal.sha256`.

`test.sh` exits 0 when it scored the job, whatever the reward. It exits 3 on a refusal and 1 on an error. The
reward is `check.py`'s reward when every check passes, and 0 otherwise. The reward file looks like this:
`{"reward", "speedup", "verdict", "checks": {name: {ok, detail}}, "label", "pairs", "harness_fails"}`.

## What `test.sh` checks

- **GPU**: `gate.json` must have a `gpu_name`, and it must equal `nvidia-smi`'s name for `$GPU`. Otherwise it refuses
  before doing anything else.
- **Parent engine**: every engine file of the staged parent venv must hash as the setup recorded
  (`setup_out/parent_<engine>.sha256`), with no file added. Checked before the lock, again under the lock, and again
  after the job.
- **Package files**: the files must match `MANIFEST.sha256`. `gate.json` is checked against the hash `set_ref.py`
  recorded.
- **Seal**: every file of the parent venv, the staged interpreter, the model and the CUDA tree must keep its sealed
  sha256. Checked before and after the job.
- **Between runs** (`lib/monitor.py`): after each run of the job finishes, the monitor re-walks the trusted trees and
  compares size, mtime and ctime. ctime changes on every write and a user process cannot set it back. This catches a
  candidate run that edits the parent even when a later candidate run restores it. The monitor also freezes each
  finished run's result files: a later run must not rewrite, for example, a parent run's `latency.json`. After each
  candidate run it also looks for processes that run started and that are still alive.
- **Harness gate**: `check.py`'s verdict must not be `fail:*` or `incomplete`, and the job must be a scoring job of
  this candidate.

## What the kit does NOT protect against

- **Timing runs inside the candidate's process.** The benchmark measures wall time with `time.perf_counter()` in the
  same Python process that imports the agent's engine, and that process writes its own `latency.json`. A candidate
  can patch the clock, sleep less in its own runs, or write made-up numbers into its own result files. The output
  checks (token counts, settings, correctness pass, teacher-forced re-read by the trusted parent) limit what a fake
  can get away with. They do not prove the timing is honest. Closing this gap needs timing from outside the process:
  a server-mode benchmark with the client in a trusted process, or the timer in the parent harness. That means
  changing the benchmark, which the kit does not do.
- **Same user, no sandbox.** Candidate runs have the operator's uid and can write anywhere the operator can. The kit
  *detects* writes to the trusted trees, the package and finished results, after each run and at the end. It does not
  *prevent* them. It cannot see a change made and undone inside one candidate run if no parent run happens in
  between, but such a change cannot affect any timing. Unprivileged sandboxes (bubblewrap) are blocked on Ubuntu
  hosts with `apparmor_restrict_unprivileged_userns=1`. A container or a second uid would turn detection into
  prevention.
- **Other paths.** Files outside the watched trees can be changed without detection: `$STAGE/cache` (compile
  caches; the parent's cache is per role), `/tmp`, `$HOME`, the harness's own `results/<label>/*.joblog`, and the
  GPU's state such as clocks. A leftover process that clears its environment, never touches the candidate venv, and
  was forked before the run's start time would not be flagged.
- **The view is only as clean as the scan.** Hard terms are the fix commit, the changed file names, the REF values,
  and the names of gate/variants/RUN.md/PROVENANCE. They must not occur in the view. Soft terms (`leakterms/<pr>.txt`
  and the mechanism phrase in RUN.md's title) are listed for a human to review. The engine source is the parent's,
  byte for byte. The public task text and task directory names are outside the kit's control.
- **`KIT_LINK_DEPS=1`** hard-links the non-engine packages into the candidate. A write through those links reaches
  the parent. The seal and the monitor would flag it, but only after the fact.
- **NVIDIA only**, like the packages (`nvidia-smi`, CUDA wheels).

## Files

`make_agent_view.sh`, `make_candidate.sh`, `set_ref.py`, `test.sh`, `view/dev_bench.sh`, `view/README.md` (the
agent's README), `lib/common.sh` (package and family detection, parent and package checks), `lib/integrity.sh`
(seal), `lib/monitor.py`, `lib/finish.py` (reward file), `lib/strip.py` (comment and docstring removal, AST-checked),
`leakterms/<pr>.txt` (soft terms, operator-side only). State goes to `<package>/kit_state/`, which is never part of
the view.

Families: `c5x` (sgl16521, sgl20972), `c5y-sglang` (sgl9460, sgl14956, sgl17005, sgl20503, sgl27868) and `c5y-vllm`
(vllm27418, vllm30056, vllm35442, vllm37338). The kit detects the family from `bench/`. It reads the timed runs' job
environment from `harness/run.sh` and refuses with "needs an adapter" if the layout is unknown.
