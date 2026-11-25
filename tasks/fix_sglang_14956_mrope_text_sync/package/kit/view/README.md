# Agent view

Everything you need to work on this task, and nothing the scorer uses to grade you.

- `venv/` : the unmodified engine and its dependencies (a private copy; edit it freely). The engine's Python source is
  under `venv/lib/python3.10/site-packages/`.
- `bench/`, `loads/` : the benchmark and the load you are scored on (`loads/cal.env`), plus the correctness pass's
  prompts (`loads/cor.env`).
- `dev_bench.sh` : times the scored load on your tree in the same job environment the scorer uses. `--help` lists the
  options. One run takes about as long as one timed run of the scorer.
- `st/` : interpreter, model weights and CUDA pieces the runs use. Do not edit.

To hand in: give the operator your edited engine package directory (or a unified diff against the unmodified one).
The operator builds and scores it with the task's scorer, which times your tree against the unmodified tree in
alternating runs and checks the outputs. Only Python files may differ from the unmodified tree.
