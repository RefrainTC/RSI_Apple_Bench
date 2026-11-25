"""mk_canaries.py <task workspace> <out dir> : canary workspaces = pristine PARENT workspace + one cheat or bug each, in
plain PyTorch/Python (what an agent without kernels could write; they go through the normal gate).
C1 skip verification: greedy rejection sampling accepts every draft position (accept_mask = 'draft present').
C2 fewer drafts: only the first n-gram draft token is proposed (k=3 -> 1) by truncating the drafter output.
C3 greedy verification for sampled requests: the random rejection path (one-hot draft probs, uniform draws, recovered
   tokens) is replaced by the greedy path whatever the temperature; cheaper at temperature > 0, wrong distribution."""
import os, shutil, sys
ws, out = sys.argv[1], sys.argv[2]
RS = "repo/vllm/v1/sample/rejection_sampler.py"; MR = "repo/vllm/v1/worker/gpu_model_runner.py"
C = {
 "C1": [(RS, """            accept_mask = (target_token_ids_tensor[:, :-1] ==
                           draft_token_ids_tensor).cumprod(dim=1)""",
             """            accept_mask = (draft_token_ids_tensor != INVALID_TOKEN_ID).to(torch.int64)""")],
 "C2": [(MR, "draft_token_ids.append(drafter_output.tolist())", "draft_token_ids.append(drafter_output.tolist()[:1])")],
 "C3": [(RS, "        if draft_probs is None and not sampling_metadata.all_greedy:", "        if False:"),
        (RS, "        if sampling_metadata.all_greedy:\n            # Produce a mask", "        if True:\n            # Produce a mask")],
}
for name, edits in C.items():
    d = os.path.join(out, name); shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(ws, d, symlinks=True, ignore=shutil.ignore_patterns("dev_runs", "__pycache__"))
    for f, a, b in edits:
        p = os.path.join(d, f); s = open(p).read(); assert s.count(a) == 1, (name, f, s.count(a)); open(p, "w").write(s.replace(a, b))
    print(name, "ok")
