import os

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from transformers import AutoModelForCausalLM

xr.set_device_type("TT")

MODEL_ID = "google/gemma-4-31B-it"
OPTIMIZATION_LEVEL = int(os.environ.get("OPT_LEVEL", "1"))
EXPORT_PATH = os.environ.get("EXPORT_PATH", "gemma4_31b_codegen")

os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
xr.use_spmd()

num_devices = xr.global_runtime_device_count()
mesh_shape = (1, num_devices)
device_ids = np.array(range(num_devices))
mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
print(f"Created device mesh: {mesh_shape} with {num_devices} devices")

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
model.eval()

compile_options = {
    "optimization_level": OPTIMIZATION_LEVEL,
    "backend": "codegen_py",
    "codegen_split_files": True,
    "export_path": EXPORT_PATH,
    "export_tensors": True,
}
torch_xla.set_custom_compile_options(compile_options)

device = xm.xla_device()
model.compile(backend="tt", options={"tt_legacy_compile": True})
model = model.to(device)

text_model = model.model.language_model
for layer in text_model.layers:
    xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", None))
    xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", None))
    xs.mark_sharding(layer.mlp.down_proj.weight, mesh, (None, "model"))

    xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
    xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
    if layer.self_attn.v_proj is not None:
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
    xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))

vocab_size = model.config.text_config.vocab_size
input_ids = torch.randint(0, vocab_size, (1, 128)).to(device)

output = model(input_ids)
xm.wait_device_ops()
print(f"Codegen complete. Check {EXPORT_PATH}/ for results.")
