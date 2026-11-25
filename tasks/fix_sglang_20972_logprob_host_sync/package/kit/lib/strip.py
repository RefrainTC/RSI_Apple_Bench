"""strip.py py|sh <src> <dst> : copy a file without its comments (and, for Python, docstrings), so that the agent's copy
of the benchmark and loads runs the same code but carries none of the authors' notes (which name tasks and mechanisms).
Python: the stripped file must parse to the same AST as the original with docstrings removed (checked; exit 1 if not).
Shell: full-line comments and trailing ' # ...' outside quotes are removed; the result must pass bash -n."""
import ast, io, re, subprocess, sys, tokenize

kind, src, dst = sys.argv[1:4]
text = open(src).read()


def py_strip(text):
    tree = ast.parse(text)
    drop = {}  # line -> True for docstring lines
    def docs(node):
        for n in ast.walk(node):
            if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body:
                b = n.body[0]
                if isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant) and isinstance(b.value.value, str):
                    yield n, b
    lines = text.split("\n")
    for n, b in list(docs(tree)):
        for i in range(b.lineno - 1, b.end_lineno):
            drop[i] = True
        if len(n.body) == 1:  # keep the block valid
            ind = re.match(r"\s*", lines[b.lineno - 1]).group(0)
            lines[b.lineno - 1] = ind + "pass"; drop.pop(b.lineno - 1)
    cut = {}
    for t in tokenize.generate_tokens(io.StringIO(text).readline):
        if t.type == tokenize.COMMENT: cut[t.start[0] - 1] = t.start[1]
    out = []
    for i, l in enumerate(lines):
        if drop.get(i): continue
        if i in cut: l = l[:cut[i]].rstrip()
        if l.strip() == "" and out and out[-1].strip() == "": continue
        out.append(l)
    res = "\n".join(out).lstrip("\n")
    # verify: same AST as the original without docstrings
    t0 = ast.parse(text)
    for n, b in list(docs(t0)):
        if len(n.body) > 1: n.body = n.body[1:]
        else: n.body = [ast.Pass()]
    if ast.dump(t0) != ast.dump(ast.parse(res)): sys.exit(f"strip.py: AST differs after stripping {src}")
    return res + "\n"


def sh_strip(text):
    out = []
    for i, l in enumerate(text.split("\n")):
        if i == 0 and l.startswith("#!"): out.append(l); continue
        q = None; cutat = None
        for j, c in enumerate(l):
            if q:
                if c == q: q = None
            elif c in "'\"": q = c
            elif c == "#" and (j == 0 or l[j - 1] in " \t;"): cutat = j; break
        if cutat is not None: l = l[:cutat].rstrip()
        if l.strip(): out.append(l)
    res = "\n".join(out) + "\n"
    open(dst + ".chk", "w").write(res)
    r = subprocess.run(["bash", "-n", dst + ".chk"]); subprocess.run(["rm", "-f", dst + ".chk"])
    if r.returncode: sys.exit(f"strip.py: bash -n fails after stripping {src}")
    return res


open(dst, "w").write(py_strip(text) if kind == "py" else sh_strip(text))
