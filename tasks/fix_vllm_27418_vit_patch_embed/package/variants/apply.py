"""apply.py <pr> <name> <site-packages dir> : deliberately broken / cheating variants used to test the task's checks.
Python-only edits (no kernels) to an installed vLLM tree (a hard-linked copy of the parent or fix venv made by
harness/mkvariant.sh). Each edit replaces one exact snippet that must occur exactly once; patched files are written as new
files (unlink + write) so the hard-linked original venv is never modified. Prints the base role on --base.

#27418 (Qwen2-VL patch embedding Conv3d -> Linear; checks: counts incl. image tokens, settings, teacher-forced re-read of
timed outputs and image pass, exact text pass):
  short       fix + requests asking for more than 4 new tokens get 4
  lowres      fix + the HF image processor is called with min_pixels = max_pixels = 451,584 (672 x 672: a quarter of the
              patches) whatever the engine arguments say (patched where vLLM merges the processor kwargs)
  skipimg     fix + the vision tower is not run; the image embeddings are zeros of the right shape (same token count)
  cheappatch  fix + the patch embedding uses only the first of the two temporal frames with the two frames' weights
              summed (half the GEMM; exact up to rounding for still images, whose two frames are copies; wrong for video)
  subpatch    fix + the patch embedding reads every second pixel of each 14 x 14 patch (7 x 7) with the weights summed
              over 2 x 2 blocks (a quarter of the GEMM; wrong features)
"""
import os, sys

SPP = "vllm/sampling_params.py"
PROC = "vllm/multimodal/processing.py"
Q2 = "vllm/model_executor/models/qwen2_vl.py"

MAXT = """        if self.max_tokens is not None and self.max_tokens < 1:
"""
MERGE = """        merged_kwargs = mm_config.merge_mm_processor_kwargs(kwargs)
"""
VIS = """                image_embeds = self.visual(pixel_values, grid_thw=grid_thw_list)
"""
PE = """    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x
"""
CHEAP = """    def forward(self, x: torch.Tensor) -> torch.Tensor:  # c5y variant cheappatch
        T, P2 = self.temporal_patch_size, self.patch_size * self.patch_size
        w = self.proj.weight.view(self.embed_dim, -1, T, P2).sum(2).reshape(self.embed_dim, -1)
        xs = x.view(x.shape[0], -1, T, P2)[:, :, 0].reshape(x.shape[0], -1)
        return xs @ w.t()
"""
SUB = """    def forward(self, x: torch.Tensor) -> torch.Tensor:  # c5y variant subpatch
        T, P, h = self.temporal_patch_size, self.patch_size, self.patch_size // 2
        w = self.proj.weight.view(self.embed_dim, -1, T, h, 2, h, 2).sum((4, 6)).reshape(self.embed_dim, -1)
        xs = x.view(x.shape[0], -1, T, P, P)[:, :, :, ::2, ::2].reshape(x.shape[0], -1)
        return xs @ w.t()
"""
VARIANTS = {
    "27418": {
        "short": ("fix", [(SPP, MAXT, """        if self.max_tokens is not None and self.max_tokens > 4:  # c5y variant short
            self.max_tokens = 4
""" + MAXT)]),
        "lowres": ("fix", [(PROC, MERGE, MERGE + """        if "max_pixels" in merged_kwargs:  # c5y variant lowres
            merged_kwargs = dict(merged_kwargs, min_pixels=451584, max_pixels=451584)
""")]),
        "skipimg": ("fix", [(Q2, VIS, """                image_embeds = torch.zeros(  # c5y variant skipimg (was self.visual(...))
                    pixel_values.shape[0] // (self.visual.spatial_merge_size ** 2), self.config.hidden_size,
                    dtype=self.visual.dtype, device=self.visual.device)
""")]),
        "cheappatch": ("fix", [(Q2, PE, CHEAP)]),
        "subpatch": ("fix", [(Q2, PE, SUB)]),
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
