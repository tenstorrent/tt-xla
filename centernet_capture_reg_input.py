#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch_xla
import torch_plugin_tt  # noqa: F401

from third_party.tt_forge_models.centernet.pytorch.loader import ModelLoader, ModelVariant


class BackboneAndReg01(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base = model.base
        self.dla_up = model.dla_up
        self.ida_up = model.ida_up
        self.reg0 = model.reg[0]
        self.reg1 = model.reg[1]
        self.first_level = model.first_level
        self.last_level = model.last_level

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x)
        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))
        x = self.reg0(y[-1])
        x = self.reg1(x)
        return x


loader = ModelLoader(variant=ModelVariant.DLA_1X_COCO)
model = loader.load_model().eval()
inputs = loader.load_inputs()

wrapper = BackboneAndReg01(model).eval()
with torch.no_grad():
    cpu_out = wrapper(inputs)
torch.save(cpu_out, "reg1_out_cpu.pt")

dev = torch_xla.device()
wrapper_tt = BackboneAndReg01(model).eval().to(dev)
compiled = torch.compile(wrapper_tt, backend="tt", options={})
with torch.no_grad():
    tt_out = compiled(inputs.to(dev))
torch_xla.sync()
torch.save(tt_out.cpu(), "reg1_out_tt.pt")
