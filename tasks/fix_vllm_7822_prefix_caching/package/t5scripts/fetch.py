"""fetch.py <pr> <role> <commit> <outdir> : download the per-commit vLLM wheel from wheels.vllm.ai (index page first,
nightly alias as fallback; the alias file is renamed to its real METADATA version so pip/uv accept it). Prints the path.
Downloaded content is only unzipped for METADATA; nothing is executed. Run with python -I."""
import hashlib, html, os, re, sys, urllib.parse, urllib.request, zipfile
pr, role, c, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
UA = {"User-Agent": "wheel-fetch"}
def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60)
url = None
try:
    body = get(f"https://wheels.vllm.ai/{c}/vllm/").read(200000).decode("utf-8", "replace")
    names = sorted(set(re.findall(r'href="([^"]+\.whl)', body)))
    x86 = [n for n in names if "x86_64" in n]
    if x86:
        url = urllib.parse.urljoin(f"https://wheels.vllm.ai/{c}/vllm/", html.unescape(x86[0])).replace("+", "%2B")
except Exception as e:
    print("index:", type(e).__name__, file=sys.stderr)
if url is None:
    url = f"https://wheels.vllm.ai/{c}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl"
tmp = os.path.join(out, "download.part")
h = hashlib.sha256()
with get(url) as r, open(tmp, "wb") as f:
    while True:
        b = r.read(1 << 20)
        if not b: break
        f.write(b); h.update(b)
with zipfile.ZipFile(tmp) as z:
    meta = [n for n in z.namelist() if n.endswith(".dist-info/METADATA")][0]
    ver = re.search(r"^Version: (.+)$", z.read(meta).decode(), re.M).group(1).strip()
name = f"vllm-{ver}-cp38-abi3-manylinux1_x86_64.whl"
os.replace(tmp, os.path.join(out, name))
with open(os.path.join(out, "SOURCE.txt"), "w") as f:
    f.write(f"pr={pr} role={role} commit={c}\nurl={url}\nsha256={h.hexdigest()}\nversion={ver}\nfile={name}\n")
print(os.path.join(out, name))
