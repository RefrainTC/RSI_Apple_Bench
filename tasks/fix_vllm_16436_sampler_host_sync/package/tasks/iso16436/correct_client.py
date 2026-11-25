"""correct_client.py --port P --model M --out F : greedy outputs (tokens + top-5 logprobs) through the running server (stdlib only).
24 fixed prompts, temperature 0, max_tokens 64, no explicit penalties (so the server's default generation config,
incl. repetition_penalty 1.1 for this model, applies). Sent once sequentially and once all at once (batched path)."""
import argparse, json, threading, urllib.request
ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int); ap.add_argument("--model"); ap.add_argument("--out"); a = ap.parse_args()
TOPICS = ["the history of the printing press", "how a refrigerator works", "the rules of chess", "photosynthesis",
          "the causes of the French Revolution", "how vaccines train the immune system", "the water cycle", "binary search",
          "the difference between TCP and UDP", "why the sky is blue", "how to make bread at home", "the life of Marie Curie",
          "plate tectonics", "the Fibonacci sequence", "how a bill becomes a law", "black holes", "the basics of supply and demand",
          "how GPS works", "the Roman Empire", "recursion in programming", "the human heart", "jazz music", "volcanoes", "compound interest"]
PROMPTS = [f"Write a short, repetitive list of facts about {t}. Repeat key words often." for t in TOPICS]
def one(p):
    body = json.dumps({"model": a.model, "prompt": p, "max_tokens": 64, "temperature": 0, "logprobs": 5}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        ch = json.loads(r.read())["choices"][0]; lp = ch.get("logprobs") or {}
        return {"text": ch["text"], "ids": lp.get("tokens") or [], "top": lp.get("top_logprobs") or []}
seq = [one(p) for p in PROMPTS]
batch = [None] * len(PROMPTS)
def w(i): batch[i] = one(PROMPTS[i])
ts = [threading.Thread(target=w, args=(i,)) for i in range(len(PROMPTS))]
[t.start() for t in ts]; [t.join() for t in ts]
json.dump({"seq": seq, "batch": batch}, open(a.out, "w"), indent=1)
print("ok", len(seq), len(batch))
