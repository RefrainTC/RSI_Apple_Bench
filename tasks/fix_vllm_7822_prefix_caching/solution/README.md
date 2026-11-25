# Reference solution: vLLM #7822

- Upstream PR: https://github.com/vllm-project/vllm/pull/7822 ("[Performance][BlockManagerV2] Mark prefix cache block
  as computed after schedule"), merged 2024-08-26.
- Fix (merge commit): `2deb029d115dadd012ce5ea70487a207cb025493`.
- Parent (first parent, the task's baseline): `029c71de11bc3bcf84a1b3cf9d91e79ab6949799`. Both report vLLM 0.5.5 (V0).

## What it changes

Two files, 33 lines: `vllm/core/block/prefix_caching_block.py` and `vllm/core/block_manager_v2.py`. After each
scheduling step, the fix marks the blocks that requests found in the prefix cache as computed, so later requests in the
batch reuse the cached prefix instead of prefilling it again. On this load the first pass drops from 9.10 s to 6.15 s.

## How the package builds the fixed arm

- Both arms are vLLM's per-commit wheels from wheels.vllm.ai (URLs and sha256 in `package/ENV.md`), installed into two
  venvs (`7822_parent`, `7822_fix`) from hashed locks with `--no-deps`. The locks include `vllm-flash-attn==2.6.1`;
  without it vLLM 0.5.5 silently uses another attention path. `package/setup/setup_venvs.sh` checks the freeze and every
  installed `vllm` file against the authors' records.
- Calibration derives the **oracle stage** (`tasks/iso7822/calib/oracle_stage`) by comparing the `.py` files of the two
  venvs (`harness/mk_oracle_stage.py`). The fix arm runs the parent venv with that stage as an overlay, exactly as a
  candidate would; a venv-equivalence pair (fix venv vs parent venv, no overlay) checks that the overlay is faithful.
- `harness/mk_fix_ws.sh tasks/iso7822 <dir>` copies the pristine workspace and applies the oracle stage to `repo/`. That
  workspace goes through the normal gate and `harness/score.sh` like any candidate.

## Result

REF **1.479** on RTX 4090 (6 jobs, 1.477-1.483, between-job SD 0.17 %). 72 of 100 outputs diverge from the parent at
near-ties (<= 0.69 nats), which the calibrated output check tolerates. See `../README.md` and
`../results/summary.json`.
