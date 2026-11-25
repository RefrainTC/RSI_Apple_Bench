"""Generate the config-only sweep candidates: <sweep dir>/variants/<name>.sh + <sweep dir>/variants/index.json.

usage: python3 -I make_variants.py <sweep dir> [--extra name=JSON ...]

Each candidate is a launch_server.sh with literal arguments only (stock vLLM 0.10.1.1 launch flags / env vars). At
scoring time it is copied unchanged to /app/submission/launch_server.sh. Comments must not contain words that the
source-scan regex in test.sh matches.

Three families:
  a01..a13  the task authors' 13 variants swept on H20 (solution/oracle_sweep_launch_server.sh, v1..v13), keeping the
            BASE_ARGS + EXTRA order (variant flags last; argparse takes the last occurrence), with 0.88 changed to
            0.85. BASE_ARGS is the old baseline: --max-num-seqs 128.
  g_*       our grid: max-num-seqs in {128,192,256,384} x async on/off, tok 8192.
  x_*       extra points chosen from the grid results (tok 16384, `vllm serve` so that --api-server-count really takes
            effect, ...), passed with --extra; their definitions are also written to index.json.
            JSON keys: source, extra (list), effective, seqs, tok, and optionally "entry": "serve".

Deduplication: `python3 -m vllm.entrypoints.openai.api_server` parses --api-server-count but never uses it
(vllm/entrypoints/cli/serve.py is the only place that reads it), so it is dropped from these variants' effective flags.
"""
import json
import sys
from pathlib import Path

SW = Path(sys.argv[1])
OUT = SW / "variants"
OUT.mkdir(parents=True, exist_ok=True)

NGRAM5 = '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":8,"prompt_lookup_min":3}'
NGRAM3 = '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":5,"prompt_lookup_min":2}'
CG = ["--cuda-graph-sizes", "1", "2", "4", "8", "16", "24", "32", "48", "64", "96", "128", "160", "192", "256"]


def base(seqs=128, tok=8192):
    return [
        "--model", '"$MODEL_PATH"',
        "--served-model-name", "default",
        "--host", "0.0.0.0",
        "--port", '"$PORT"',
        "--tensor-parallel-size", "1",
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "8192",
        "--max-num-seqs", str(seqs),
        "--max-num-batched-tokens", str(tok),
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--disable-log-requests",
        "--trust-remote-code",
    ]


def q(tok):
    """bash array element: JSON goes in single quotes, everything else as is."""
    return f"'{tok}'" if tok.startswith("{") else tok


V = []


def add(name, family, source, extra, effective, screened=True, dedupe=None, seqs=128, tok=8192,
        entry="api_server"):
    V.append(dict(name=name, family=family, source=source, base_seqs=seqs, base_tok=tok,
                  extra=extra, effective=effective, screened=screened, dedupe=dedupe, entry=entry))


# ---- the authors' 13 (H20 sweep of 2026-07-27, v1..v13 of oracle_sweep_launch_server.sh) ----
A = "author sweep (oracle_sweep_launch_server.sh)"
add("a01_async", "author", A + " v1_async", ["--async-scheduling"],
    "seqs128 tok8192 async")
add("a02_async_api4", "author", A + " v2_async_api4", ["--async-scheduling", "--api-server-count", "4"],
    "seqs128 tok8192 async (api4 has no effect)", screened=False,
    dedupe="effective flags = a01, and identical to solution/oracle_v2_launch_server.sh, already scored 5 times (median 1.3633); not rerun")
add("a03_async_fullcg", "author", A + " v3_async_fullcg",
    ["--async-scheduling", "--compilation-config", '{"full_cuda_graph": true}'],
    "seqs128 tok8192 async full_cuda_graph")
add("a04_async_bigtok", "author", A + " v4_async_bigtok",
    ["--async-scheduling", "--max-num-batched-tokens", "16384", "--disable-log-stats"],
    "seqs128 tok16384 async dls")
add("a05_api4_only", "author", A + " v5_api4_only", ["--api-server-count", "4", "--disable-log-stats"],
    "seqs128 tok8192 dls (api4 has no effect) ~ grid point s128/off + dls")
add("a06_combo", "author", A + " v6_combo",
    ["--async-scheduling", "--api-server-count", "4", "--disable-log-stats",
     "--max-num-batched-tokens", "16384"] + CG,
    "seqs128 tok16384 async dls cg≤256 (api4 has no effect)")
add("a07_async_cg128", "author", A + " v7_async_cg128", ["--async-scheduling", "--disable-log-stats"] + CG,
    "seqs128 tok8192 async dls cg≤256")
add("a08_ngram", "author", A + " v8_ngram", ["--disable-log-stats", "--speculative-config", NGRAM5],
    "seqs128 tok8192 dls ngram(5,8,3)")
add("a09_api4_ngram", "author", A + " v9_api4_ngram",
    ["--api-server-count", "4", "--disable-log-stats", "--speculative-config", NGRAM5],
    "= a08 (api4 has no effect)", screened=False, dedupe="effective flags = a08")
add("a10_async_api4_ngram", "author", A + " v10_async_api4_ngram",
    ["--async-scheduling", "--api-server-count", "4", "--disable-log-stats", "--speculative-config", NGRAM5],
    "seqs128 tok8192 async dls ngram(5,8,3) (source: async and speculative decoding are mutually exclusive; expected not to start)")
add("a11_api4_seqs256", "author", A + " v11_api4_seqs256",
    ["--api-server-count", "4", "--disable-log-stats", "--max-num-seqs", "256"],
    "seqs256 tok8192 dls (api4 has no effect) ~ grid point s256/off + dls")
add("a12_api4_ngram3", "author", A + " v12_api4_ngram3",
    ["--api-server-count", "4", "--disable-log-stats", "--speculative-config", NGRAM3],
    "seqs128 tok8192 dls ngram(3,5,2) (api4 has no effect)")
add("a13_api8_ngram", "author", A + " v13_api8_ngram",
    ["--api-server-count", "8", "--disable-log-stats", "--speculative-config", NGRAM5],
    "= a08 (api8 has no effect)", screened=False, dedupe="effective flags = a08")

# ---- grid on this card: seqs x async (tok 8192) ----
G = "5090 grid"
for s in (128, 192, 256, 384):
    for a in (True, False):
        name = f"g_s{s}_{'a' if a else 'n'}"
        eff = f"seqs{s} tok8192" + (" async" if a else "")
        screened, dedupe = True, None
        if s == 128 and a:
            screened, dedupe = False, "effective flags = a01 (authors' v1); only a01 is run"
        if s == 128 and not a:
            screened, dedupe = False, "~ a05 (one more --disable-log-stats, saves only CPU); only a05 is run"
        if s == 256 and not a:
            screened, dedupe = False, ("~ a11 (plus --disable-log-stats); identical to an earlier agent submission, "
                                       "already scored twice (1.354 / 1.373)")
        add(name, "grid", G, ["--async-scheduling"] if a else [], eff, screened=screened, dedupe=dedupe,
            seqs=s, tok=8192)

# ---- extra points (chosen from the grid results, --extra name=JSON) ----
args = sys.argv[2:]
while args:
    if args[0] != "--extra":
        sys.exit(f"bad arg {args[0]}")
    name, js = args[1].split("=", 1)
    d = json.loads(js)
    add(name, "ext", d["source"], d["extra"], d["effective"], seqs=d["seqs"], tok=d["tok"],
        entry=d.get("entry", "api_server"))
    args = args[2:]

HDR = """#!/usr/bin/env bash
# config-only sweep candidate: {name}
# source: {source}
# effective: {effective}
# Stock vLLM 0.10.1.1, launch flags only. gpu-memory-utilization 0.85 (hardware cap on 24-32 GB cards).
set -euo pipefail
PORT="${{PORT:-30001}}"
MODEL_PATH="${{MODEL_PATH:-/app/model}}"
export VLLM_LOGGING_LEVEL="${{VLLM_LOGGING_LEVEL:-WARNING}}"
export VLLM_USE_V1="${{VLLM_USE_V1:-1}}"
ARGS=(
{args}
)
echo "[sweep] variant={name} argv: ${{ARGS[*]}}" >&2
"""
for v in V:
    argv = base(v["base_seqs"], v["base_tok"]) + v["extra"]
    if v["entry"] == "serve":
        # `vllm serve <model>`: the model is positional, so drop --model
        i = argv.index("--model")
        argv = argv[:i] + argv[i + 2:]
        launch = 'exec python3 -m vllm.entrypoints.cli.main serve "$MODEL_PATH" "${ARGS[@]}"\n'
    else:
        launch = 'exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"\n'
    lines, i = [], 0
    while i < len(argv):  # one flag and its values per line
        j = i + 1
        while j < len(argv) and not argv[j].startswith("--"):
            j += 1
        lines.append("    " + " ".join(q(t) for t in argv[i:j]))
        i = j
    text = HDR.format(name=v["name"], source=v["source"], effective=v["effective"],
                      args="\n".join(lines)) + launch
    p = OUT / f"{v['name']}.sh"
    p.write_text(text)
    p.chmod(0o755)
    v["argv"] = argv

(OUT / "index.json").write_text(json.dumps(V, indent=1, ensure_ascii=False) + "\n")
print(f"wrote {len(V)} variants ({sum(v['screened'] for v in V)} to screen) -> {OUT}")
