# env.sh : the only place with machine-specific settings for this package (iso7822).
# Every script sources it (if ISO_ROOT/P2T are not already exported); you can also `source env.sh` in your shell.
# Precedence: variables already exported > site.env (yours, next to this file; see site.env.example) > defaults below.
if [ -z "${ISO_ENV_DONE:-}" ]; then
  _iso_pkg=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
  # site.env must not override what the caller set (e.g. `GPU=1 CPUS=8-15,40-47 PORT=47200 bash ...score.sh`):
  # remember every variable that is already set, read site.env, then put the remembered values back.
  _iso_keep=""
  for _iso_v in ISO_ROOT STAGE P2T GPU PORT CPUS UV HOST_PY_BIN; do
    [ -n "${!_iso_v+x}" ] && _iso_keep+="$_iso_v=$(printf %q "${!_iso_v}"); "
  done
  [ -f "$_iso_pkg/site.env" ] && . "$_iso_pkg/site.env"
  eval "$_iso_keep"
  # ISO_ROOT: this package's directory (persistent disk). Venvs, uv cache, the uv CPython, the HF model cache, results,
  #           locks and the STOP file live under it. Default: the directory that holds this env.sh.
  export ISO_ROOT=${ISO_ROOT:-$_iso_pkg}
  # STAGE: fast local disk (NVMe) for the staged venvs, interpreter, model and compile caches; exported to the scripts
  #        as P2T (the name the authors' harness uses; "STAGE" is a local variable inside run_variant.sh).
  export P2T=${P2T:-${STAGE:-/tmp/iso7822}}
  export -n STAGE 2>/dev/null; STAGE=$P2T
  # GPU: index of the one GPU a job uses (nvidia-smi numbering). PORT: base port for vLLM (VLLM_PORT); one per GPU.
  export GPU=${GPU:-0}
  export PORT=${PORT:-$((47000 + 200 * GPU))}
  # CPUS: taskset list for the job (the authors pinned each GPU to one 8-core CCD + SMT siblings, e.g. GPU0 0-7,32-39).
  #       Default: all online CPUs (no pinning). Exported to the harness as P2_CPUS_OVERRIDE.
  export CPUS=${CPUS:-0-$(( $(nproc --all) - 1 ))}
  export P2_CPUS_OVERRIDE=$CPUS
  # UV: the uv binary (tested: uv 0.11.14).
  export UV=${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}
  # host python3 for the harness's own stdlib-only scripts; falls back to the uv CPython built by setup/setup_venvs.sh
  if command -v python3 > /dev/null 2>&1; then HOST_PY_BIN=${HOST_PY_BIN:-$(dirname "$(command -v python3)")}
  else HOST_PY_BIN=${HOST_PY_BIN:-$ISO_ROOT/.uv-python/cpython-3.10.20-linux-x86_64-gnu/bin}; export PATH="$HOST_PY_BIN:$PATH"; fi
  export HOST_PY_BIN
  export ISO_ENV_DONE=1
  unset _iso_pkg _iso_keep _iso_v
fi
