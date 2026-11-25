#!/bin/bash
# package: CPUS from env.sh (exported as P2_CPUS_OVERRIDE) wins; the authors' table below is used only when it is unset.
[ -n "${ISO_ENV_DONE:-}" ] || . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../env.sh"   # package: machine settings
[ -n "${P2_CPUS_OVERRIDE:-}" ] && { echo "$P2_CPUS_OVERRIDE"; exit 0; }
# p2_cpus.sh <gpu> : the taskset CPU list for GPU <gpu> from the pod's pinning table (one CCD = 8 physical cores + their
# SMT siblings per GPU; written from lscpu -e). Fails loudly if the table is missing or has no row for that GPU.
T=$ISO_ROOT/cpu_pinning_main.txt
c=$(awk -v g="$1" '$1 == g && $1 !~ /^#/ {print $2}' "$T" 2>/dev/null)
[ -n "$c" ] || { echo "p2_cpus: no row for gpu $1 in $T" >&2; exit 1; }
echo "$c"
