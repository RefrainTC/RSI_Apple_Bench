"""apply.py <pr> <name> <site-packages dir> : deliberately broken / cheating variants used to test the task's checks.
Python-only edits (no kernels) to an installed SGLang tree (a hard-linked copy of the parent or fix venv made by
harness/mkvariant.sh). Each edit replaces one exact snippet that must occur exactly once; patched files are written as new
files (unlink + write) so the hard-linked original venv is never modified. Prints the base role on --base.

#9460 (M-RoPE decode positions without per-request syncs; checks: correctness pass, teacher-forced gap, counts, settings):
  nodelta   fix + decode positions ignore the image's mrope_position_delta (text-only rule seq_len - 1 for every request)
  offby1    fix + decode positions one too far (delta + seq_len instead of delta + seq_len - 1)
  frozen    fix + each request keeps the positions of its first decode step (computed once, never advanced)
  fp8kv     fix + KV cache stored as fp8_e5m2, chosen inside the model runner (ServerArgs still say "auto")
  short     fix + requests asking for more than 200 new tokens get 200 (the timed reps stop early)
  lowres    fix + every image is downscaled to half its width and height before the vision tower (4x fewer image tokens)
"""
import os, sys

FBI = "sglang/srt/model_executor/forward_batch_info.py"
MR = "sglang/srt/model_executor/model_runner.py"
SCH = "sglang/srt/managers/scheduler.py"
QVL = "sglang/srt/multimodal/processors/qwen_vl.py"

DEC = """                    mrope_positions_list[batch_idx] = (
                        (mrope_position_deltas + self.seq_lens[batch_idx] - 1)
                        .unsqueeze(0)
                        .repeat(3, 1)
                    )
"""
BOOT = """            if recv_req.bootstrap_port is None:
                # Use default bootstrap port
                recv_req.bootstrap_port = self.server_args.disaggregation_bootstrap_port
"""
VARIANTS = {
    "9460": {
        "nodelta": ("fix", [(FBI, DEC, DEC.replace("(mrope_position_deltas + self.seq_lens[batch_idx] - 1)",
                                                   "(mrope_position_deltas * 0 + self.seq_lens[batch_idx] - 1)  # c5y variant nodelta"))]),
        "offby1": ("fix", [(FBI, DEC, DEC.replace("(mrope_position_deltas + self.seq_lens[batch_idx] - 1)",
                                                  "(mrope_position_deltas + self.seq_lens[batch_idx])  # c5y variant offby1"))]),
        "frozen": ("fix", [(FBI, DEC, """                    if getattr(mm_input, "_c5y_pos", None) is None:  # c5y variant frozen: first decode step only
                        mm_input._c5y_pos = (
                            (mrope_position_deltas + self.seq_lens[batch_idx] - 1)
                            .unsqueeze(0)
                            .repeat(3, 1)
                        )
                    mrope_positions_list[batch_idx] = mm_input._c5y_pos
""")]),
        "fp8kv": ("fix", [(MR, """        if self.server_args.kv_cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
""", """        if self.server_args.kv_cache_dtype == "auto":
            self.kv_cache_dtype = torch.float8_e5m2  # c5y variant fp8kv (was self.dtype)
""")]),
        "short": ("fix", [(SCH, BOOT, """            if recv_req.sampling_params.max_new_tokens is not None and recv_req.sampling_params.max_new_tokens > 200:  # c5y variant short
                recv_req.sampling_params.max_new_tokens = 200
""" + BOOT)]),
        "lowres": ("fix", [(QVL, """def resize_image(image, size_factor: int = IMAGE_FACTOR) -> Image.Image:
    width, height = image.size
""", """def resize_image(image, size_factor: int = IMAGE_FACTOR) -> Image.Image:
    width, height = image.size[0] // 2, image.size[1] // 2  # c5y variant lowres
""")]),
    },
}


def main():
    pr, name = sys.argv[1], sys.argv[2]
    base, edits = VARIANTS[pr][name]
    if sys.argv[3] == "--base":
        print(base); return
    sp = sys.argv[3]
    for rel, old, new in edits:
        p = os.path.join(sp, rel); s = open(p).read()
        n = s.count(old)
        if n != 1: sys.exit(f"anchor occurs {n} times in {rel} (variant {name})")
        s = s.replace(old, new)
        os.unlink(p)  # break the hard link to the base venv's file
        with open(p, "w") as f: f.write(s)
        mod = os.path.splitext(os.path.basename(p))[0]
        pc = os.path.join(os.path.dirname(p), "__pycache__")
        if os.path.isdir(pc):
            for f in os.listdir(pc):
                if f.startswith(mod + "."): os.unlink(os.path.join(pc, f))
        print(f"patched {rel}")


if __name__ == "__main__":
    main()
