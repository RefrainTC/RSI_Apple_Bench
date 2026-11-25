# common.sh (sourced) : iso12380 operator helpers. GPU 0 of THIS pod only.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
P2=$ISO_ROOT; H=$P2/harness_iso12380; TD=$P2/tasks/iso12380; OPS=$P2/ops_iso12380
HT=$(bash $H/hosttag.sh); LOGD=$OPS/logs; mkdir -p $LOGD
stop_check() { [ -e $P2/STOP_iso12380 ] && { echo "$(date '+%F %T') STOP_iso12380 present: stopping" | tee -a $LOGD/driver_$HT.log; exit 8; }; return 0; }
gpu0_free() {  # refuse when anything runs on this pod's GPU 0
  local u; u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0)
  [ -z "$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | awk -F', ' -v u="$u" '$2==u')" ] || {
    echo "$(date '+%F %T') GPU0 ($u) busy: refusing" | tee -a $LOGD/driver_$HT.log; return 1; }
}
mktmpd() { mkdir -p $P2T/tmp && chmod 700 $P2T/tmp; mktemp -d -p $P2T/tmp iso12380.XXXXXX; }
