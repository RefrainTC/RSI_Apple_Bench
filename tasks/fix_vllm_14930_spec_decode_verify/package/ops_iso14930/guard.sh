# guard.sh (sourced): gpu_free <gpu> ; stop_check
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
P2=$ISO_ROOT
stop_check() { if [ -e $P2/STOP_iso14930 ]; then echo "$(date -Is) STOP file present, stopping"; return 1; fi; return 0; }
gpu_free() { local u; u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$1")
  if nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | grep -q "$u"; then echo "$(date -Is) GPU $1 ($u) busy: refuse"; return 1; fi; return 0; }
