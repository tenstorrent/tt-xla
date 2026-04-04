# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CPU sanity check for the ConvNeXt ForgeModel loader.
Run from the tt-xla repo root:
    source venv/activate
    python tests/torch/models/convnext/sanity_cpu.py
"""

import torch
from third_party.tt_forge_models.convnext.image_classification.pytorch import (
    ModelLoader,
    ModelVariant,
)

# --- float32 ---
loader = ModelLoader()
model = loader.load_model()
inputs = loader.load_inputs()
print(f"Variant:      {loader._variant}")
print(f"Model info:   {loader.get_model_info()}")
print(f"Input shape:  {inputs['pixel_values'].shape}, dtype={inputs['pixel_values'].dtype}")

with torch.no_grad():
    out = model(**inputs)
logits = out.logits
print(f"Output shape: {logits.shape}, dtype={logits.dtype}")
print(f"Top-1 class:  {logits.argmax(-1).item()}")

# --- bfloat16 ---
loader_bf = ModelLoader()
model_bf = loader_bf.load_model(dtype_override=torch.bfloat16)
inputs_bf = loader_bf.load_inputs(dtype_override=torch.bfloat16)
print(f"\nbfloat16 input dtype:  {inputs_bf['pixel_values'].dtype}")
with torch.no_grad():
    out_bf = model_bf(**inputs_bf)
print(f"bfloat16 output shape: {out_bf.logits.shape}, dtype={out_bf.logits.dtype}")
print("bfloat16 forward pass: OK")
