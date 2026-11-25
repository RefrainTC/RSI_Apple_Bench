#!/bin/bash
# gate_selftest.sh <taskdir> : negative + positive cases for snapshot_gate.sh on copies of the pristine workspace.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
TD=$(readlink -f "$1"); H=$ISO_ROOT/harness_iso14930; W=$ISO_ROOT/smoke/selftest_$(basename $TD)
rm -rf $W; mkdir -p $W
mk() { rm -rf $W/ws; cp -a $TD/pristine $W/ws; chmod -R u+w $W/ws; }
f=$(cd $TD/pristine/repo && find vllm -name 'sampler.py' -path '*sample*' | head -1); [ -n "$f" ] || f=$(cd $TD/pristine/repo && find vllm -name 'scheduler.py' | head -1)
run() { bash $H/snapshot_gate.sh $TD $W/ws $W/g_$1 > /dev/null 2>&1; rc=$?; echo "$1 expect=$2 rc=$rc $( [ $rc = $2 ] && echo OK || echo MISMATCH ) :: $(grep -m1 -E 'REJECT|PASS' $W/g_$1/gate.txt)"; }
mk; run pristine 0
mk; printf '\n# harmless comment\nX_P2_TEST = 1\n' >> $W/ws/repo/$f; run plain_edit 0
mk; printf '\nimport triton\n' >> $W/ws/repo/$f; run import_triton 1
mk; printf '\n@triton.jit\ndef k(x): pass\n' >> $W/ws/repo/$f; run triton_jit 1
mk; printf '\nfrom torch.utils.cpp_extension import load_inline\n' >> $W/ws/repo/$f; run load_inline 1
mk; printf '\nimport ctypes\n' >> $W/ws/repo/$f; run ctypes 1
mk; printf '\n# tamper\nimport os; os.environ.get("P2_OVERLAY")\n' >> $W/ws/repo/$f; run harness_tamper 1
mk; printf '\nX = 1\n' >> $W/ws/repo/vllm/envs.py; run locked_envs 1
mk; printf '\x00\x01binary' > $W/ws/repo/vllm/blob.bin; run binary 1
mk; ln -s /etc/passwd $W/ws/repo/vllm/link.py; run symlink 1
mk; echo 'echo hi' >> $W/ws/dev_run.sh; run dev_run_changed 1
mk; rm $W/ws/repo/$f; run delete_file 0
