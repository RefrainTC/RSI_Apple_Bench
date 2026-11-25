"""mk_alt.py <task workspace> <out dir> : V1 = a legitimate torch/Python-only rewrite of the parent's greedy rejection
path (what an agent under the no-kernel rule could write), to measure how much of the Triton fix's gain is reachable
without kernels. Same outputs by construction:
 (a) greedy: argmax over the flat [num_tokens, vocab] logits first, then split/pad the token ids (padding 0 = the argmax
     of the parent's all-zero padded rows), with split points computed on the CPU. Removes the device cumsum + .tolist()
     sync and the padded [batch, k+1, vocab] fp32 copy (~155 MB per step at batch 64).
 (b) model runner: valid sampled ids parsed from one .tolist() of the [batch, k+1] tensor instead of boolean-mask
     indexing + per-row .tolist()."""
import os, shutil, sys
ws, out = sys.argv[1], sys.argv[2]
RS = "repo/vllm/v1/sample/rejection_sampler.py"; MR = "repo/vllm/v1/worker/gpu_model_runner.py"
E = [(RS, """        sample_lens = [len(x) + 1 for x in draft_token_ids]
        target_probs = _convert_2d_probs(target_probs, sample_lens)
""", """        sample_lens = [len(x) + 1 for x in draft_token_ids]
        if sampling_metadata.all_greedy:
            split, acc = [], 0
            for n in sample_lens[:-1]:
                acc += n
                split.append(acc)
            flat_ids = target_probs.argmax(dim=-1)
            target_probs = pad_sequence(list(torch.tensor_split(flat_ids, split)), batch_first=True, padding_value=0)
        else:
            target_probs = _convert_2d_probs(target_probs, sample_lens)
"""),
     (RS, """            target_token_ids_tensor = target_probs.argmax(dim=-1)
""", """            target_token_ids_tensor = (target_probs if target_probs.dim() == 2 else target_probs.argmax(dim=-1))
"""),
     (MR, """            valid_mask = sampled_token_ids != INVALID_TOKEN_ID
            gen_lens = valid_mask.sum(dim=1).tolist()
            # TODO(woosuk): Optimize this.
            valid_sampled_token_ids = [
                seq.tolist()
                for seq in sampled_token_ids[valid_mask].split(gen_lens)
            ]""", """            valid_sampled_token_ids = [[t for t in row if t != INVALID_TOKEN_ID]
                                       for row in sampled_token_ids.tolist()]""")]
d = os.path.join(out, "V1"); shutil.rmtree(d, ignore_errors=True)
shutil.copytree(ws, d, symlinks=True, ignore=shutil.ignore_patterns("dev_runs", "__pycache__"))
for f, a, b in E:
    p = os.path.join(d, f); s = open(p).read(); assert s.count(a) == 1, (f, s.count(a), a[:60]); open(p, "w").write(s.replace(a, b))
print("V1 ok")
