# env.sh : machine-specific settings for package iso14857, sourced by every script. Override any of them in the
# environment before running a script; the defaults are shown. Nothing else in the package names a machine path.
#   ISO_ROOT  persistent work root (this package's own directory by default): venvs, model, LoRA, uv cache, results.
#   STAGE     fast local disk (local NVMe) for the staged interpreter, venv copies, model copy and compile caches.
#   UV        uv binary (0.11.14 was used).          SYSPY  any host python3 >= 3.8 (stdlib only; used with -I).
#   CPUS      taskset CPU list for the GPU job (default: the authors' CCD table on their pods, else this shell's affinity).
#   PORT      base port for the engine's internal ports (default 31000 + 200*GPU, +1000 on the authors' n1 pod).
#   GPU       is the first argument of b3_caldrv.sh (nvidia-smi index), not an environment variable.
: "${ISO_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
: "${STAGE:=/tmp/iso14857}"
: "${UV:=uv}"
: "${SYSPY:=python3}"
export ISO_ROOT STAGE UV SYSPY
