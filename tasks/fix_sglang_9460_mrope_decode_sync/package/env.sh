# env.sh : the only place with machine-specific settings for this package. Every script sources it.
# Precedence: variables already set by the caller > site.env (yours, next to this file; see site.env.example) > defaults.
if [ -z "${C5Y_ENV_DONE:-}" ]; then
  _p=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
  # site.env must not override what the caller set (e.g. `GPU=1 CPUS=24-31,56-63 bash harness/run.sh ...`):
  # remember every variable that is already set, read site.env, then put the remembered values back.
  _keep=""
  for _v in ISO_ROOT STAGE GPU PORT CPUS UV SYSPY STOP_FILES; do
    [ -n "${!_v+x}" ] && _keep+="$_v=$(printf %q "${!_v}"); "
  done
  [ -f "$_p/site.env" ] && . "$_p/site.env"
  eval "$_keep"
  # ISO_ROOT: persistent work root (default: this directory): venvs, uv cache, interpreter, model, results, STOP file.
  export ISO_ROOT=${ISO_ROOT:-$_p}
  # STAGE: fast local disk (NVMe) for the staged interpreter, venvs, model, CUDA headers, compile caches, scratch.
  #        Keep it short (< ~20 characters): SGLang's IPC socket paths are limited to 107 characters.
  export STAGE=${STAGE:-/tmp/c5y/st}
  # GPU: nvidia-smi index of the one GPU a job uses. PORT: base of the engine's ports (one per GPU; SGLang uses PORT+100..).
  export GPU=${GPU:-0}
  export PORT=${PORT:-$((36000 + 500 * GPU))}
  # CPUS: taskset list for the job (the authors pinned each GPU to one 8-core CCD + SMT siblings). Default: no pinning.
  export CPUS=${CPUS:-0-$(( $(nproc --all) - 1 ))}
  export UV=${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}
  export SYSPY=${SYSPY:-python3}
  # STOP_FILES: extra stop files (space separated) checked before every GPU job, in addition to $ISO_ROOT/STOP.
  export STOP_FILES=${STOP_FILES:-}
  export C5Y_ENV_DONE=1
  unset _p _keep _v
fi
