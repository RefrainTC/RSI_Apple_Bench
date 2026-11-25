"""sgl_c5x.py <out dir> : workload, correctness pass and teacher-forced output check for the SGLang #16521 / #20972
tasks (trusted, ours; extends batch 4's bench/sgl_b4.py, whose timed part is unchanged). sglang.Engine, greedy,
ignore_eos. Every timed rep has its own seeded prompt set (seed 1 + rep), identical in both arms; warm-up seed 1000.

Timed modes (env):
  default      N prompts of IN random token ids -> OL tokens. Metric: median rep time.
  LOGPROB=k    every request return_logprob=True, top_logprobs_num=k (#20972); the rep digest covers ids + logprobs.
  LP_ONLY=1    every request return_logprob=True without top-k (#20972 transfer load / negative control).
  MT=1         two turns per rep (#16521): turn 1 = N x IN random ids -> OL tokens; turn 2 = turn-1 ids + turn-1 output
               ids + NEW fresh random ids -> OL2 tokens. Metric: turn-2 time (turn-1 time kept as turn1_per_rep_s).
               Writes <out>/seqs.json (prompt and output ids of both turns of every rep) for the teacher-forced check.
Other modes:
  CORRECT=1    untimed correctness pass on fixed text prompts -> <out>/correct.json.
               LOGPROB task: 24 prompts (tokenized here, sent as ids) in 3 generate() calls (batch 16, 1, 7), 64 greedy tokens, top-5 logprobs and
               token-id logprobs for 4 fixed ids. MT task: 32 two-turn conversations (turn 1: 48 tokens; turn 2 = turn-1
               ids + output + a follow-up -> 32 tokens with top-5 logprobs); seqs.json as for timed MT runs.
  TF=1         teacher-forced check (run in the PARENT venv only): for every run dir in TF_DIRS (":"-separated) reads
               seqs.json and re-reads each prompt + output in one prefill (radix cache and CUDA graphs off), and records
               for every output token the gap (nats) between the top-1 logprob and the logprob of the token actually
               produced. -> <out>/tf.json.
Every engine run also writes <out>/settings.json (the live engine's ServerArgs) for the settings check.
Env: M N IN OL REPS MEMF SGL_KW (JSON engine kwargs) LOGPROB LP_ONLY MT NEW OL2 CORRECT TF TF_DIRS SGLANG_PORT_BASE."""
import dataclasses, hashlib, json, os, random, statistics, sys, time

WORDS = ("the of and to in is was for on that with as by at from his her it an were are which this be or has had "
         "river city music garden winter science market history paper light").split()


def main():
    import sglang as sgl
    out = sys.argv[1]; E = os.environ.get
    M = E("M"); N = int(E("N", "64")); IN = int(E("IN", "256")); OL = int(E("OL", "256")); REPS = int(E("REPS", "3"))
    LOGPROB = int(E("LOGPROB", "0")); MT = E("MT", "0") == "1"; NEW = int(E("NEW", "32")); OL2 = int(E("OL2", "8"))
    CORRECT = E("CORRECT", "0") == "1"; TF = E("TF", "0") == "1"
    if not os.path.isdir(M):
        from huggingface_hub import snapshot_download
        M = snapshot_download(M, local_files_only=True)
    kw = dict(model_path=M, mem_fraction_static=float(E("MEMF", "0.80")), random_seed=0, log_level="error")
    if E("SGLANG_PORT_BASE"): kw["port"] = int(E("SGLANG_PORT_BASE"))
    if E("SGL_KW"): kw.update(json.loads(E("SGL_KW")))
    if TF: kw.update(disable_radix_cache=True, disable_cuda_graph=True)
    ids = lambda r, n: [r.randrange(100, 30000) for _ in range(n)]
    pset = lambda seed: (lambda r: [ids(r, IN) for _ in range(N)])(random.Random(seed))
    t0 = time.perf_counter(); llm = sgl.Engine(**kw); startup = time.perf_counter() - t0
    try:
        sa = dataclasses.asdict(llm.server_args)
        json.dump({k: v for k, v in sa.items() if isinstance(v, (str, int, float, bool, type(None)))},
                  open(os.path.join(out, "settings.json"), "w"), sort_keys=True)
    except Exception as e:  # recorded; the settings check then fails
        json.dump({"error": repr(e)}, open(os.path.join(out, "settings.json"), "w"))
    sp = lambda ol: {"temperature": 0, "max_new_tokens": ol, "ignore_eos": True}

    if TF:
        tf_check(llm, out, [d for d in E("TF_DIRS", "").split(":") if d]); llm.shutdown(); return
    if CORRECT:
        correct(llm, out, MT, sp, M); llm.shutdown(); return

    def gen(n, ol, **a):
        if LOGPROB: a.update(return_logprob=True, top_logprobs_num=LOGPROB)
        elif E("LP_ONLY") == "1": a.update(return_logprob=True)
        return llm.generate(sampling_params=sp(ol), **a)

    def digest(res):
        h = []
        for r in res:
            x = {"ids": r.get("output_ids"), "text": r.get("text")}
            if LOGPROB:
                mi = r["meta_info"]
                x["lp"] = [round(t[0], 3) for t in mi["output_token_logprobs"]]
                x["top"] = [[(round(t[0], 3), t[1]) for t in d] for d in mi["output_top_logprobs"]]
            h.append(x)
        return hashlib.sha256(json.dumps(h).encode()).hexdigest()[:16]

    run = lambda seed, n, ol: gen(n, ol, input_ids=pset(seed)[:n])
    t = time.perf_counter(); run(1000, 8, 8); warm = time.perf_counter() - t
    ts, t1s, ntoks, ntoks1, shas, shas1, shas2 = [], [], [], [], [], [], []
    seqs = []
    for rep in range(REPS):
        if MT:
            r = random.Random(1 + rep); p1 = [ids(r, IN) for _ in range(N)]
            t = time.perf_counter(); res1 = gen(N, OL, input_ids=p1); t1s.append(time.perf_counter() - t)
            o1 = [list(x["output_ids"]) for x in res1]
            p2 = [a + o + ids(r, NEW) for a, o in zip(p1, o1)]
            t = time.perf_counter(); res = gen(N, OL2, input_ids=p2); ts.append(time.perf_counter() - t)
            shas.append(digest(res1) + digest(res)); shas1.append(digest(res1)); shas2.append(digest(res))
            ntoks1.append(sum(x["meta_info"]["completion_tokens"] for x in res1))
            seqs.append({"t1": [[a, o] for a, o in zip(p1, o1)], "t2": [[a, list(x["output_ids"])] for a, x in zip(p2, res)],
                         "cached_t2": [x["meta_info"].get("cached_tokens") for x in res]})
        else:
            t = time.perf_counter(); res = run(1 + rep, N, OL); ts.append(time.perf_counter() - t); shas.append(digest(res))
        ntoks.append(sum(x["meta_info"]["completion_tokens"] for x in res))
    d = {"startup_s": startup, "warm_s": warm, "per_rep_s": ts, "median_s": statistics.median(ts), "metric_s": statistics.median(ts),
         "turn1_per_rep_s": t1s or None, "out_tokens_per_rep": ntoks, "out_tokens_turn1_per_rep": ntoks1 or None,
         "out_sha_per_rep": shas, "sha_turn1_per_rep": shas1 or None, "sha_turn2_per_rep": shas2 or None,
         "M": M, "N": N, "IN": IN, "OL": OL, "REPS": REPS, "LOGPROB": LOGPROB, "LP_ONLY": E("LP_ONLY", "0"), "MT": MT,
         "NEW": NEW, "OL2": OL2, "ENGINE_KW": {k: v for k, v in kw.items()}}
    if MT:
        d["cached_tokens_turn2_mean"] = statistics.mean(c or 0 for s in seqs for c in s["cached_t2"])
        json.dump(seqs, open(os.path.join(out, "seqs.json"), "w"))
    json.dump(d, open(os.path.join(out, "latency.json"), "w")); print(json.dumps({k: d[k] for k in ("per_rep_s", "median_s")}))
    llm.shutdown()


def correct(llm, out, MT, sp, M):
    r = random.Random(7)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(M)
    if not MT:
        # Text prompts are tokenized here and sent as token ids, so that all requests of a call reach the scheduler
        # together. Sent as text, they are tokenized one by one in the engine and arrive staggered; the human fix's
        # faster scheduler loop then groups the batch-7 call differently from the parent (deterministically), which
        # changes bf16 numerics (logprobs up to 0.25 nats, 3 of 7 outputs diverge at exact ties). See the card.
        texts = [" ".join(r.choice(WORDS) for _ in range(20 + 7 * i)) for i in range(24)]
        pids = [tok.encode(t, add_special_tokens=False) for t in texts]
        tid = [11, 13, 279, 315]
        res_all = []
        for a, b in ((0, 16), (16, 17), (17, 24)):
            res_all += llm.generate(input_ids=pids[a:b], sampling_params=sp(64), return_logprob=True, top_logprobs_num=5,
                                    token_ids_logprob=[tid] * (b - a))
        outs = []
        for x in res_all:
            mi = x["meta_info"]
            outs.append({"ids": [t[1] for t in mi["output_token_logprobs"]], "lp": [t[0] for t in mi["output_token_logprobs"]],
                         "top": [[[t[0], t[1]] for t in d] for d in mi["output_top_logprobs"]],
                         "tid": [[[t[0], t[1]] for t in d] for d in (mi.get("output_token_ids_logprobs") or [])],
                         "text": x["text"]})
        json.dump(outs, open(os.path.join(out, "correct.json"), "w")); print("CORRECT_DONE", len(outs)); return
    q1 = [" ".join(r.choice(WORDS) for _ in range(30 + 5 * i)) for i in range(32)]
    q2 = [" and then " + " ".join(r.choice(WORDS) for _ in range(12 + i)) for i in range(32)]
    p1 = [tok.encode(q, add_special_tokens=False) for q in q1]
    res1 = llm.generate(input_ids=p1, sampling_params=sp(48))
    o1 = [list(x["output_ids"]) for x in res1]
    p2 = [a + o + tok.encode(q, add_special_tokens=False) for a, o, q in zip(p1, o1, q2)]
    res2 = llm.generate(input_ids=p2, sampling_params=sp(32), return_logprob=True, top_logprobs_num=5)
    outs = []
    for a, x in zip(p2, res2):
        mi = x["meta_info"]
        outs.append({"ids": list(x["output_ids"]), "lp": [t[0] for t in mi["output_token_logprobs"]],
                     "top": [[[t[0], t[1]] for t in d] for d in mi["output_top_logprobs"]],
                     "cached": mi.get("cached_tokens"), "text": x["text"]})
    json.dump(outs, open(os.path.join(out, "correct.json"), "w"))
    json.dump([{"t1": [[a, o] for a, o in zip(p1, o1)], "t2": [[a, o["ids"]] for a, o in zip(p2, outs)]}],
              open(os.path.join(out, "seqs.json"), "w"))
    print("CORRECT_DONE", len(outs))


def tf_check(llm, out, dirs):
    """Teacher-forced re-read. gap = top-1 logprob - logprob of the produced token, per output token (0 = argmax)."""
    res = {}; t0 = time.perf_counter()
    for d in dirs:
        seqs = json.load(open(os.path.join(d, "seqs.json")))
        rd = {}
        for part in ("t1", "t2"):
            items = [(rep, i, p, o) for rep, s in enumerate(seqs) for i, (p, o) in enumerate(s[part])]
            nonarg = []; maxg = 0.0; npos = 0; bad = 0; worse = 0
            B = 16 if part == "t1" else 64   # bounds the fp32 logits of the scored positions (t1: 16 x 256 per call)
            for k in range(0, len(items), B):
                chunk = items[k:k + B]
                rr = llm.generate(input_ids=[p + o for _, _, p, o in chunk], sampling_params={"temperature": 0, "max_new_tokens": 1},
                                  return_logprob=True, logprob_start_len=[len(p) - 1 for _, _, p, _ in chunk], top_logprobs_num=1)
                for (rep, i, p, o), x in zip(chunk, rr):
                    mi = x["meta_info"]; itl = mi["input_token_logprobs"]; itop = mi["input_top_logprobs"]
                    # logprob_start_len = len(p) - 1: SGLang returns one entry per input token from that position on, the
                    # first one without a logprob (None); the last len(o) entries are the output tokens, each scored given
                    # everything before it. Checked by token id below.
                    itl, itop = itl[-len(o):], itop[-len(o):]
                    if (len(itl) != len(o) or len(itop) != len(o) or any(t[0] is None or t[1] != oj for t, oj in zip(itl, o))
                            or any(not tp or tp[0][0] is None for tp in itop)):
                        if bad == 0: print("TF_ALIGN_DEBUG", mi["input_token_logprobs"][:3], mi["input_top_logprobs"][:3], o[:3], flush=True)
                        bad += 1; continue
                    for j, (tl, top) in enumerate(zip(itl, itop)):
                        g = max(0.0, top[0][0] - tl[0]); npos += 1
                        if top[0][1] != o[j]:
                            nonarg.append([rep, i, j, round(g, 4)])
                            if g > 0: worse += 1
                        if g > maxg: maxg = g
            rd[part] = {"n_pos": npos, "n_nonargmax": len(nonarg), "n_worse": worse, "max_gap": round(maxg, 4), "len_mismatch": bad,
                        "nonargmax": sorted(nonarg, key=lambda z: -z[3])[:200]}
        res[d] = rd
        print("TF", d, {p: (v["n_pos"], v["n_nonargmax"], v["max_gap"], v["len_mismatch"]) for p, v in rd.items()}, flush=True)
    json.dump({"dirs": res, "tf_s": time.perf_counter() - t0}, open(os.path.join(out, "tf.json"), "w"))


if __name__ == "__main__":
    main()
