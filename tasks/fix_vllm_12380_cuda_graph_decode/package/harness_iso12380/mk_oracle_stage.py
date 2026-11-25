#!/usr/bin/env python3
"""mk_oracle_stage.py <parent site-packages> <fix site-packages> <pkg> <out stage dir>
Writes the human fix as a stage (same layout as kernel_gate --stage): .py files that differ or are new in the fix
package go to repo_overlay/<pkg>/..., .py files missing in the fix go to repo_deleted.txt. Non-.py files are compared
and reported only (they cannot be part of an agent's submission either)."""
import filecmp, json, os, shutil, sys
P, F, pkg, out = sys.argv[1:5]
shutil.rmtree(out, ignore_errors=True); os.makedirs(os.path.join(out, "repo_overlay"))
def files(root):
    r = {}
    for dp, dns, fns in os.walk(os.path.join(root, pkg)):
        dns[:] = [d for d in dns if d != "__pycache__"]
        for f in fns:
            if f.endswith(".pyc"): continue
            full = os.path.join(dp, f); r[os.path.relpath(full, root)] = full
    return r
p, f = files(P), files(F)
changed, deleted, nonpy = [], [], []
for rel, full in f.items():
    if rel not in p or not filecmp.cmp(full, p[rel], shallow=False):
        if rel.endswith(".py"):
            t = os.path.join(out, "repo_overlay", rel); os.makedirs(os.path.dirname(t), exist_ok=True); shutil.copy2(full, t)
            changed.append(rel)
        else:
            nonpy.append(rel)
for rel in p:
    if rel not in f:
        (deleted if rel.endswith(".py") else nonpy).append(rel)
open(os.path.join(out, "repo_deleted.txt"), "w").write("".join(d + "\n" for d in deleted))
json.dump({"changed_py": changed, "deleted_py": deleted, "nonpy_differences": nonpy}, open(os.path.join(out, "oracle_stage.json"), "w"), indent=1)
print(json.dumps({"changed_py": changed, "deleted_py": deleted, "nonpy_differences": nonpy}))
