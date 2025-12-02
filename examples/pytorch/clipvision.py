# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Demonstrates codegen for CLIP Vision Model with Projection from HuggingFace

import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from transformers import AutoProcessor, CLIPVisionModelWithProjection
from transformers.image_utils import load_image

# Set up XLA runtime for TT backend
xr.set_device_type("TT")

processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = load_image(url)
inputs = processor(images=image, return_tensors="pt")

# Load CLIP Vision Model from HuggingFace
model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

with torch.inference_mode():
    outputs = model(**inputs)

print(outputs)

BATCH_SIZE = 2
# Set up codegen options
options = {
    "backend": "codegen_py",
    "export_path": f"clipvision_opt2_b{BATCH_SIZE}___",
    "optimization_level": 2,
    "export_tensors": True,
}

torch_xla.set_custom_compile_options(options)

# Compile for TT, then move the model and it's inputs to device.
device = xm.xla_device()
model.to(torch.bfloat16)
model.compile(backend="tt")
model = model.to(device)
x = torch.randn(BATCH_SIZE, 3, 224, 224, dtype=torch.bfloat16).to(device)

# Run the model. This triggers code generation.
output = model(pixel_values=x)