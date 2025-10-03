# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Code adapted from: https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2
License: https://github.com/X-PLUG/mPLUG-Owl/blob/main/LICENSE
"""

import torch
import torch.utils.checkpoint
from torch import nn


class MultiwayNetwork(nn.Module):
    def __init__(self, module_provider, num_multiway=2, out_features=None):
        super(MultiwayNetwork, self).__init__()

        self.multiway = torch.nn.ModuleList(
            [module_provider() for _ in range(num_multiway)]
        )
        self.out_features = out_features

    def forward(self, hidden_states, multiway_indices):

        if len(self.multiway) == 1:
            return self.multiway[0](hidden_states)
        if self.out_features:
            output_hidden_states = torch.empty(
                hidden_states.size(0),
                hidden_states.size(1),
                self.out_features,
                dtype=hidden_states.dtype,
            ).to(hidden_states.device)
        else:
            output_hidden_states = torch.empty_like(hidden_states)
        for idx, subway in enumerate(self.multiway):
            local_indices = multiway_indices.eq(idx).nonzero(as_tuple=True)
            hidden = hidden_states[local_indices].unsqueeze(1).contiguous()
            if hidden.numel():
                output = subway(hidden)
                if isinstance(output, tuple):
                    output = output[0]
                output = output.squeeze(1)
                output_hidden_states[local_indices] = output

        return output_hidden_states.contiguous()
