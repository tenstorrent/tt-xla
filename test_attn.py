import os
import sys
from python_package.tt_torch.codegen import codegen_py
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from pathlib import Path
import json

xr.set_device_type('TT')

# Add the DeepSeek model path to sys.path
deepseek_path = Path(__file__).parent / "DeepSeek-V3.2-Exp" / "inference"
sys.path.insert(0, str(deepseek_path))

from model import ModelArgs, MLA

with open("DeepSeek-V3.2-Exp/inference/config_671B_v3.2.json") as f:
    config = json.load(f)
    # Override dtype to bf16 since TT-XLA doesn't support fp8
    config["dtype"] = "bf16"
    args = ModelArgs(**config)

hidden_size = args.index_head_dim
import scipy
haddamard_matrix = (torch.tensor(scipy.linalg.hadamard(hidden_size), dtype=torch.bfloat16)*(hidden_size ** -0.5)).to("xla")
# Set default device to XLA before creating the model

att = MLA(args, haddamard_matrix).to("xla")
# Reset default device
torch.set_default_device("cpu")


# Load model with prefix removal
from safetensors import safe_open
import torch

# Load the safetensors file and remove prefix
safetensors_path = os.path.join("DeepSeek_params", f"model0-mp1-attn-bf16.safetensors")
prefix_to_remove = "layers.0.attn."

with safe_open(safetensors_path, framework="pt", device="cpu") as f:
    state_dict = {}
    for key in f.keys():
        # Remove the prefix if it exists
        new_key = key.replace(prefix_to_remove, "") if key.startswith(prefix_to_remove) else key
        state_dict[new_key] = f.get_tensor(key)

# # Apply weight resizing for the problematic weights
# weights_to_resize = {
#     'wq_b.weight': (0, 2),  # Resize first dimension by factor of 4
#     'wkv_b.weight': (0, 2), # Resize first dimension by factor of 4  
#     'wo.weight': (1, 2)     # Resize second dimension by factor of 4
# }

# for weight_name, (dim, factor) in weights_to_resize.items():
#     if weight_name in state_dict:
#         original_shape = state_dict[weight_name].shape
#         if dim == 0:
#             new_size = original_shape[0] // factor
#             state_dict[weight_name] = state_dict[weight_name][:new_size, :]
#         elif dim == 1:
#             new_size = original_shape[1] // factor
#             state_dict[weight_name] = state_dict[weight_name][:, :new_size]
#         print(f"Resized {weight_name}: {original_shape} -> {state_dict[weight_name].shape}")

# Load the modified state dict into the model (model is already on XLA device)
att.load_state_dict(state_dict, strict=False)
att.to("xla")
# Load freqs_cis and convert from complex to real format for TT-XLA compatibility
freqs_cis_complex = torch.load("DeepSeek_params/freqs_cis.pt")
# Convert complex tensor to real tensor with shape [..., 2] where last dim is [real, imag]
freqs_cis_full = torch.view_as_real(freqs_cis_complex)
# Create input tensor with correct shape
seq_len = 16
start_pos = 0
# torch.manual_seed(42)
# x = torch.randn(1, seq_len, args.dim).bfloat16().to("xla")
x = torch.load("DeepSeek_params/x.pt").to("xla")
mask = (torch.full((seq_len, seq_len), float("-inf"), device="cpu").triu_(1) if seq_len > 1 else None).to("xla")
# Slice freqs_cis for the current sequence
freqs_cis = freqs_cis_full[start_pos:start_pos+seq_len].to("xla")

compile_options = {
    "backend": "codegen_py",
    "export_path": "deepseek_attn_codegen",
    "export_tensors": True,
}
torch_xla.set_custom_compile_options(compile_options)

#output = att(x, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask)
#print(f"Output shape: {output}")

#codegen_py(att, x, export_path="deepseek_attn_codegen", kwargs={"start_pos": start_pos, "freqs_cis": freqs_cis, "mask": mask})

torch_xla.sync()

jitted_att = torch.compile(att, backend="tt")
output = jitted_att(x, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask)
#torch.save(output.cpu(), "DeepSeek_params/tt_final_output_jitted.pt")
print(f"Output: {output}")