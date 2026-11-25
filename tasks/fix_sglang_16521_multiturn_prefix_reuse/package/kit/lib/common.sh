# common.sh : sourced by every kit script. Finds the package, reads its settings and detects the family.
# The kit sits in <package>/kit/ (the export copies it there); ISO_ROOT set by the caller wins.
# Sets: KIT ISO_ROOT STAGE GPU CPUS PORT SYSPY PR ENGINE (sglang|vllm) ENG_TAG (sgl|vllm) FAMILY (c5x|c5y-sglang|c5y-vllm)
#       BENCH (bench/<file>.py) PV (staged parent venv) SPREL (lib/python3.X/site-packages) HREC (parent engine hash
#       record) KSTATE ($ISO_ROOT/kit_state)
KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
kit_die() { echo "KIT ERROR: $*" >&2; exit 2; }
kit_log() { echo "[kit $(date -u '+%F %T')] $*" >&2; }
if [ -z "${ISO_ROOT:-}" ]; then ISO_ROOT=$(dirname "$KIT"); fi
export ISO_ROOT
[ -f "$ISO_ROOT/env.sh" ] && [ -f "$ISO_ROOT/env/build.env" ] && [ -f "$ISO_ROOT/harness/run.sh" ] \
  || kit_die "no package at ISO_ROOT=$ISO_ROOT (need env.sh, env/build.env, harness/run.sh)"
. "$ISO_ROOT/env.sh"; . "$ISO_ROOT/env/build.env"
[ -n "${PR:-}" ] || kit_die "env/build.env has no PR"
if [ -f "$ISO_ROOT/env/parent_sgl.sha256" ]; then ENGINE=sglang; ENG_TAG=sgl
elif [ -f "$ISO_ROOT/env/parent_vllm.sha256" ]; then ENGINE=vllm; ENG_TAG=vllm
else kit_die "cannot tell the engine: no env/parent_sgl.sha256 or env/parent_vllm.sha256"; fi
if [ -f "$ISO_ROOT/bench/sgl_c5x.py" ]; then FAMILY=c5x; BENCH=bench/sgl_c5x.py
elif [ -f "$ISO_ROOT/bench/sgl_c5y.py" ]; then FAMILY=c5y-sglang; BENCH=bench/sgl_c5y.py
elif [ -f "$ISO_ROOT/bench/offline_c5y.py" ]; then FAMILY=c5y-vllm; BENCH=bench/offline_c5y.py
else kit_die "unknown harness family (no bench/sgl_c5x.py, sgl_c5y.py or offline_c5y.py): needs an adapter"; fi
PV=$STAGE/venvs/${PR}_parent
SPREL=lib/python3.10/site-packages
# the hash record of the parent's engine files: the one this machine's setup wrote (setup_out/), else the authors' (env/)
HREC=$ISO_ROOT/setup_out/parent_${ENG_TAG}.sha256; [ -f "$HREC" ] || HREC=$ISO_ROOT/env/parent_${ENG_TAG}.sha256
KSTATE=$ISO_ROOT/kit_state
export KIT PR ENGINE ENG_TAG FAMILY BENCH PV SPREL HREC KSTATE

# verify_parent_engine : every engine file of the staged parent venv hashes as the setup recorded, and there is no extra
# file (bytecode aside). Prints one line; returns 1 on any mismatch.
verify_parent_engine() {
  local sp=$PV/$SPREL n1 n2 bad
  [ -f "$STAGE/.staged/venv_${PR}_parent" ] && [ -d "$sp/$ENGINE" ] || { echo "parent venv not staged: $PV"; return 1; }
  bad=$(cd "$sp" && sha256sum --quiet -c "$HREC" 2>&1 | head -5)
  [ -z "$bad" ] || { echo "PARENT ENGINE FILES DIFFER from $HREC: $(echo "$bad" | tr '\n' ' ')"; return 1; }
  n1=$(wc -l < "$HREC")
  n2=$(cd "$sp" && find "$ENGINE" \( -type f -o -type l \) ! -name '*.pyc' ! -path '*/__pycache__/*' | wc -l)
  [ "$n1" = "$n2" ] || { echo "PARENT ENGINE FILE COUNT $n2 != recorded $n1 (file added or removed)"; return 1; }
  echo "parent engine ok: $n1 $ENGINE files hash as recorded in $(basename "$(dirname "$HREC")")/$(basename "$HREC")"
}

# verify_package_files : package files (harness, bench, loads, env...) still match MANIFEST.sha256; gate.json is checked
# against the kit's own record when set_ref.py has rewritten it.
verify_package_files() {
  local bad
  [ -f "$ISO_ROOT/MANIFEST.sha256" ] || { echo "no MANIFEST.sha256"; return 1; }
  if [ -f "$KSTATE/gate.sha256" ]; then
    bad=$(cd "$ISO_ROOT" && grep -v '  harness/gate.json$' MANIFEST.sha256 | sha256sum --quiet -c - 2>&1 | head -5)
    (cd "$ISO_ROOT" && sha256sum --quiet -c "$KSTATE/gate.sha256" >/dev/null 2>&1) || bad="$bad harness/gate.json differs from the set_ref.py record"
  else
    bad=$(cd "$ISO_ROOT" && sha256sum --quiet -c MANIFEST.sha256 2>&1 | head -5)
  fi
  [ -z "$bad" ] || { echo "PACKAGE FILES DIFFER: $(echo "$bad" | tr '\n' ' ')"; return 1; }
  echo "package files ok (MANIFEST.sha256$([ -f "$KSTATE/gate.sha256" ] && echo ', gate.json as set_ref.py wrote it'))"
}

# trees whose bytes the scoring trusts (snapshot / seal targets): parent venv, staged interpreter, model, CUDA tree
trusted_trees() {
  local d; for d in "$PV" "$STAGE/py" "$STAGE/hf" "$STAGE/cuda"; do [ -d "$d" ] && echo "$d"; done
}
