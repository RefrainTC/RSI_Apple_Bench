#!/bin/bash
# package: CPUS from env.sh (exported as P2_CPUS_OVERRIDE) wins; the authors' table below is used only when it is unset.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
[ -n "${P2_CPUS_OVERRIDE:-}" ] && { echo "$P2_CPUS_OVERRIDE"; exit 0; }
# p2_cpus.sh <gpu> (iso12380 copy) : taskset CPU list for GPU <gpu> on THIS pod, from cpu_pinning_iso12380.txt (rows:
# host gpu cpus). Fails loudly if there is no row.
T=$ISO_ROOT/harness_iso12380/cpu_pinning_iso12380.txt; HT=$(bash $ISO_ROOT/harness_iso12380/hosttag.sh)
c=$(awk -v h="$HT" -v g="$1" '$1 !~ /^#/ && $1 == h && $2 == g {print $3}' "$T" 2>/dev/null)
[ -n "$c" ] || { echo "p2_cpus: no row for $HT gpu $1 in $T" >&2; exit 1; }
echo "$c"
