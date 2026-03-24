#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from third_party.tt_forge_models.centernet.pytorch.loader import ModelLoader, ModelVariant

PCC_THRESHOLD = 0.97


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().cpu().flatten()
    b = b.detach().float().cpu().flatten()
    va, vb = a - a.mean(), b - b.mean()
    denom = va.norm() * vb.norm()
    return 1.0 if float(denom) == 0 else float((va @ vb) / denom)


class RegConv(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.reg2 = model.reg[2]

    def forward(self, x):
        return self.reg2(x)


loader = ModelLoader(variant=ModelVariant.DLA_1X_COCO)
model = loader.load_model().eval()

reg = RegConv(model).eval()

inp_cpu = torch.load("reg1_out_cpu.pt")
inp_tt = torch.load("reg1_out_tt.pt")

with torch.no_grad():
    ref = reg(inp_cpu)
    out = reg(inp_tt)

pcc = compute_pcc(ref, out)
assert pcc >= PCC_THRESHOLD, f"PCC={pcc:.6f} below threshold {PCC_THRESHOLD}"
