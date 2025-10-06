# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
class GraphModule(torch.nn.Module):
    def forward(self, L_input_ids_: "i64[1, 5]"):
        l_input_ids_ = L_input_ids_

        # File: /localdev/mstojkovic/tt-xla/repro_assert_metadata.py:22 in forward, code: mask = input_ids.ne(self.padding_idx).int()
        ne: "b8[1, 5]" = l_input_ids_.ne(1)
        l_input_ids_ = None
        mask: "i32[1, 5]" = ne.int()
        ne = None

        # File: /localdev/mstojkovic/tt-xla/repro_assert_metadata.py:26 in forward, code: cumsum_result = torch.cumsum(mask, dim=1)  # This creates Int64 tensor
        cumsum_result: "i64[1, 5]" = torch.cumsum(mask, dim=1, dtype=mask.dtype)

        # File: /localdev/mstojkovic/tt-xla/repro_assert_metadata.py:27 in forward, code: incremental_indices = cumsum_result.type_as(mask) * mask
        type_as: "i32[1, 5]" = cumsum_result.type_as(mask)
        cumsum_result = None
        incremental_indices: "i32[1, 5]" = type_as * mask
        type_as = mask = None

        # File: /localdev/mstojkovic/tt-xla/repro_assert_metadata.py:30 in forward, code: position_ids = incremental_indices.long() + self.padding_idx
        long: "i64[1, 5]" = incremental_indices.long()
        incremental_indices = None
        position_ids: "i64[1, 5]" = long + 1
        long = None
        return (position_ids,)
