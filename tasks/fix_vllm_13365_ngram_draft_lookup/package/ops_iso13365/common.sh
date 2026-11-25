# common.sh : sourced by the iso13365 ops scripts. STOP files, GPU-empty check, logging.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
P2=$ISO_ROOT; H=$P2/harness_iso13365; TD=$P2/tasks/iso13365; OPS=$P2/ops_iso13365
LOGD=$OPS/logs; mkdir -p $LOGD
log() { echo "$(date -u '+%F %T') [$(hostname | cut -c1-12)] $*" | tee -a $LOGD/ops.log; }
stop_check() {  # exit cleanly when a STOP file exists
  if [ -e $P2/STOP_iso13365 ]; then log "STOP_iso13365 present: stopping before $1"; exit 8; fi
}
gpu_free() {  # <gpu index> : 0 if nothing runs on that GPU (by UUID)
  local u; u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$1") || return 1
  if nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | grep -q "$u"; then
    log "GPU $1 ($u) has a compute process: refusing"; nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader | grep "$u" | tee -a $LOGD/ops.log; return 1; fi
  return 0
}
hostctx() {  # load + other GPUs' utilisation, for the record
  echo "load=$(cut -d' ' -f1-3 /proc/loadavg) gpus=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ';')"
}
