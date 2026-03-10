import torch
import torch.nn as nn
import torch_xla.runtime as xr
from tt_torch import codegen_py
from third_party.tt_forge_models.swin.image_classification.pytorch import ModelLoader, ModelVariant
from loguru import logger

# Set up XLA runtime for TT backend.
xr.set_device_type("TT")

loader = ModelLoader(ModelVariant.SWIN_S)
model = loader.load_model(dtype_override=torch.bfloat16)
inputs = loader.load_inputs(dtype_override=torch.bfloat16)

codegen_py(model, inputs, export_path="conv2d")