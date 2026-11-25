#!/bin/bash
# b3_cpus.sh <gpu> : taskset CPU list for GPU <gpu> on THIS pod (batch 3). One CCD per GPU (5975WX on both pods; SMT
# sibling = k+32). main: as /home/coder/work/cpu_pinning_main.txt. n1: GPU0 = CCD0 (as harness_iso12380), GPU1 = CCD3 (as
# harness_iso13365 used on n1 GPU1).
[ -n "${CPUS:-}" ] && { echo "$CPUS"; exit 0; }   # package: an explicit CPUS wins
HT=$(bash "$(dirname "$(readlink -f "$0")")/hosttag.sh")
case "$HT:$1" in
  main:0) echo 0-7,32-39;; main:1) echo 8-15,40-47;; main:2) echo 16-23,48-55;; main:3) echo 24-31,56-63;;
  n1:0) echo 0-7,32-39;; n1:1) echo 24-31,56-63;;
  *) taskset -pc $$ | sed 's/.*: //';;   # package: any other machine: this shell's CPU affinity (no pinning)
esac
