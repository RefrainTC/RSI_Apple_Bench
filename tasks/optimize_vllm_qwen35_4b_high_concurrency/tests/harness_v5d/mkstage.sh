#!/usr/bin/env bash
# mkstage.sh <task> <name> [flag lines...]  -> tasks/<task>/sweep/<name> (repo = pristine, config only)
# env lines are given as ENV:NAME=value
set -euo pipefail
P=${P3_ROOT:?set P3_ROOT}
T=$P/tasks/$1; N=$2; shift 2
S=$T/sweep/$N
rm -rf $S; mkdir -p $S/submission
ln -s $T/pristine/repo $S/repo
: > $S/submission/extra_args.txt; : > $S/submission/env.txt
for x in "$@"; do
  case "$x" in ENV:*) echo "${x#ENV:}" >> $S/submission/env.txt;; *) echo "$x" >> $S/submission/extra_args.txt;; esac
done
echo "$S"; cat $S/submission/extra_args.txt $S/submission/env.txt
