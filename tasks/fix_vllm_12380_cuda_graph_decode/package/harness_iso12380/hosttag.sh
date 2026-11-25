#!/bin/bash
# hosttag.sh : which 4090-server pod this is (main = pod-main 4-GPU pod, n1 = pod-n1 2-GPU pod). Both share the NFS home,
# so per-host names keep locks and labels apart.
case "$(hostname)" in gpu-pod-n1-*) echo n1;; gpu-pod-main-*) echo main;; *) echo "unknown-$(hostname)";; esac
