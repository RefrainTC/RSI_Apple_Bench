#!/bin/bash
# hosttag.sh : which 4090-server pod this is (main = 4-GPU pod, n1 = 2-GPU pod). Copied from harness_iso12380.
case "$(hostname)" in gpu-pod-n1-*) echo n1;; gpu-pod-main-*) echo main;; *) echo "unknown-$(hostname)";; esac
