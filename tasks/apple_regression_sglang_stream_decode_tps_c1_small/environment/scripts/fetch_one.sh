#!/bin/bash
# fetch_one.sh <pin> : one short-lived `pip download --no-deps` (fresh HTTPS connections) for one lock pin.
# env from the caller: vpy (venv python), wd (wheel dir), IDXARGS (index args), LOGD (log dir). Up to 3 tries.
p=$1; lg=$LOGD/$(echo "$p" | tr "=<>/ " "_____").log
for t in 1 2 3; do
  timeout 900 $vpy -m pip download --no-deps -d "$wd" $IDXARGS "$p" > "$lg" 2>&1 && exit 0
  sleep 5
done
echo "$p" >> "$wd.fail"
exit 1
