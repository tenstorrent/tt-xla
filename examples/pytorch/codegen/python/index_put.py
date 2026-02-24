# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch_xla.runtime as xr
from tt_torch import codegen_py

# Set up XLA runtime for TT backend
xr.set_device_type("TT")


class index_put(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self,  hidden_states,p0,p1, image_features_proj):
            
            hidden_states = hidden_states.index_put(
                (p0,p1), image_features_proj, accumulate=False
            )

            return hidden_states 


model = index_put()
model.eval()

hidden_states = torch.load('hidden_states.pt',map_location="cpu")
p0= torch.load('positions_0.pt',map_location="cpu")
p1 = torch.load('positions_1.pt',map_location="cpu")
image_features_proj = torch.load('image_features_proj.pt',map_location="cpu")

inputs = [hidden_states,p0,p1,image_features_proj]

codegen_py(model, *inputs, export_path="index_put")

