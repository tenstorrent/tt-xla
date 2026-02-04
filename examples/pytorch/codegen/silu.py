# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import torch
import torch_xla.runtime as xr
from tt_torch import codegen_py
from third_party.tt_forge_models.yolov9.pytorch.loader import ModelLoader, ModelVariant
from loguru import logger


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.act = model.model[22].cv3[1][0].act

    def forward(self, x):
        x = self.act(x)
        return x


def main():
    # Set up XLA runtime for TT backend.
    xr.set_device_type("TT")
    
    loader = ModelLoader(ModelVariant.S)

    model = loader.load_model(dtype_override=torch.bfloat16)
    model = Wrapper(model)
    model.eval()
    x = torch.load('act_ip.pt',map_location="cpu")

    logger.info("x={}",x)
    logger.info("x.shape={}",x.shape)
    logger.info("x.dtype={}",x.dtype)

    codegen_py(model, x, export_path="silu")


if __name__ == "__main__":
    main()
