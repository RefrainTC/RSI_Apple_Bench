"""download_model.py <out dir> : download Qwen/Qwen2.5-1.5B-Instruct at the pinned revision and record what arrived.

Same file filter as the task Dockerfile (`--exclude "original/*"`). Writes <out dir>/model_revision.json (repo, revision,
file list with sizes, sha256 of weights and JSON files). Our runs used revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306;
the script we ran downloaded the repo HEAD, which was that revision on both days we checked. This copy pins it.
Run with a venv that has huggingface_hub>=0.24,<1:  python3 -I download_model.py <out dir>
"""
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

OUT = Path(sys.argv[1])
REPO = "Qwen/Qwen2.5-1.5B-Instruct"
REV = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
dest = OUT / "Qwen2.5-1.5B-Instruct"
info = HfApi().model_info(REPO, revision=REV)
snapshot_download(REPO, revision=REV, local_dir=str(dest), ignore_patterns=["original/*"])
files = []
for p in sorted(dest.rglob("*")):
    if p.is_file() and ".cache" not in p.parts:
        rec = {"path": str(p.relative_to(dest)), "bytes": p.stat().st_size}
        if p.suffix in (".safetensors", ".json"):
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            rec["sha256"] = h.hexdigest()
        files.append(rec)
out = {"repo": REPO, "revision": info.sha, "last_modified": str(info.last_modified), "exclude": ["original/*"], "files": files}
(OUT / "model_revision.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({"revision": info.sha, "n_files": len(files)}))
