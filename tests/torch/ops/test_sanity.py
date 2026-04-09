# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import ComparisonConfig, Framework, Workload
from infra.testers.single_chip.op.op_tester import OpTester
from loguru import logger
from transformers import CLIPProcessor, CLIPModel

import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


def compute_pcc(x: torch.Tensor, y: torch.Tensor):
    x_float = x.to(torch.float64) if x.dtype != torch.float64 else x
    y_float = y.to(torch.float64) if y.dtype != torch.float64 else y

    x_flat, y_flat = x_float.flatten(), y_float.flatten()
    vx = x_flat - x_flat.mean()
    vy = y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()

    if denom == 0:
        return float("nan")
    else:
        return ((vx @ vy) / denom).item()


def test_quick_gelu():

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.act = model.vision_model.encoder.layers[11].mlp.activation_fn

        def forward(self, x):
            return self.act(x) 
    
    # prepare the model
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336", dtype=torch.bfloat16)
    model.eval()
    model = Wrapper(model)
    
    # Compile model
    compiled_model = torch.compile(model, backend="tt")
    device = xm.xla_device()
    compiled_model = compiled_model.to(device)
    
    # load inputs
    cpu_input = torch.load('gelu_ip_golden.pt',map_location="cpu")
    tt_input = torch.load('gelu_ip_device.pt',map_location="cpu")
    
    logger.info("cpu_input={}",cpu_input)
    logger.info("cpu_input.shape={}",cpu_input.shape)
    logger.info("cpu_input.dtype={}",cpu_input.dtype)
    
    logger.info("tt_input={}",tt_input)
    logger.info("tt_input.shape={}",tt_input.shape)
    logger.info("tt_input.dtype={}",tt_input.dtype)
    
    # pcc check for inputs 
    input_pcc = compute_pcc(cpu_input,tt_input)
    logger.info("input_pcc={}",input_pcc)
    
    tt_input = tt_input.to(device)
    
    # tt run
    with torch.no_grad():
        tt_output = compiled_model(tt_input)

    # cpu tun
    with torch.no_grad():
        cpu_output = model(cpu_input)
        
    # pcc check for outputs
    output_pcc = compute_pcc(cpu_output,tt_output.cpu())
    logger.info("output_pcc={}",output_pcc)
    
    