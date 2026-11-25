"""make_lora.py <out dir> <hidden> <inter> <layers> <q_out> <kv_out> <rank> <seed> : write a random-weight PEFT LoRA
adapter (safetensors + adapter_config.json) for a Qwen2/Llama-style base, targets q,k,v,o,gate,up,down. Data only."""
import json, os, sys, torch
from safetensors.torch import save_file
o, H, I, L, Q, KV, r, seed = sys.argv[1], *map(int, sys.argv[2:9])
g = torch.Generator().manual_seed(seed); w = {}
dims = {"self_attn.q_proj": (H, Q), "self_attn.k_proj": (H, KV), "self_attn.v_proj": (H, KV), "self_attn.o_proj": (Q, H),
        "mlp.gate_proj": (H, I), "mlp.up_proj": (H, I), "mlp.down_proj": (I, H)}
for l in range(L):
    for m, (i, j) in dims.items():
        p = f"base_model.model.model.layers.{l}.{m}"
        w[p + ".lora_A.weight"] = (torch.randn(r, i, generator=g) / i ** 0.5).to(torch.bfloat16)
        w[p + ".lora_B.weight"] = (torch.randn(j, r, generator=g) * 0.002).to(torch.bfloat16)
os.makedirs(o, exist_ok=True); save_file(w, os.path.join(o, "adapter_model.safetensors"))
json.dump({"base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct", "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": r,
           "lora_alpha": 2 * r, "lora_dropout": 0.0, "bias": "none", "fan_in_fan_out": False, "inference_mode": True,
           "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
          open(os.path.join(o, "adapter_config.json"), "w"), indent=1)
print("wrote", o, len(w), "tensors")
