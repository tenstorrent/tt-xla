# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
class GraphModule(torch.nn.Module):
    def forward(self, args_0):
        args_0: "i64[1, 5]"

        (args_0,) = fx_pytree.tree_flatten_spec(([args_0], {}), self._in_spec)
        # File: /localdev/mstojkovic/tt-xla/repro_assert_metadata.py:22 in forward, code: mask = input_ids.ne(self.padding_idx).int()
        ne: "b8[1, 5]" = torch.ops.aten.ne.Scalar(args_0, 1)
        args_0 = None
        convert_element_type: "i32[1, 5]" = (
            torch.ops.prims.convert_element_type.default(ne, torch.int32)
        )
        _assert_tensor_metadata = torch.ops.aten._assert_tensor_metadata.default(
            ne, None, None, torch.bool
        )
        ne = _assert_tensor_metadata = None

        # File: /localdev/mstojkovic/tt-xla/repro_assert_metadata.py:26 in forward, code: cumsum_result = torch.cumsum(mask, dim=1)  # This creates Int64 tensor
        cumsum: "i64[1, 5]" = torch.ops.aten.cumsum.default(convert_element_type, 1)

        # File: /localdev/mstojkovic/tt-xla/repro_assert_metadata.py:27 in forward, code: incremental_indices = cumsum_result.type_as(mask) * mask
        convert_element_type_1: "i32[1, 5]" = (
            torch.ops.prims.convert_element_type.default(cumsum, torch.int32)
        )
        _assert_tensor_metadata_1 = torch.ops.aten._assert_tensor_metadata.default(
            cumsum, None, None, torch.int64
        )
        cumsum = _assert_tensor_metadata_1 = None
        mul: "i32[1, 5]" = torch.ops.aten.mul.Tensor(
            convert_element_type_1, convert_element_type
        )
        convert_element_type_1 = convert_element_type = None

        # File: /localdev/mstojkovic/tt-xla/repro_assert_metadata.py:30 in forward, code: position_ids = incremental_indices.long() + self.padding_idx
        convert_element_type_2: "i64[1, 5]" = (
            torch.ops.prims.convert_element_type.default(mul, torch.int64)
        )
        _assert_tensor_metadata_2 = torch.ops.aten._assert_tensor_metadata.default(
            mul, None, None, torch.int32
        )
        mul = _assert_tensor_metadata_2 = None
        add: "i64[1, 5]" = torch.ops.aten.add.Tensor(convert_element_type_2, 1)
        convert_element_type_2 = None
        return pytree.tree_unflatten((add,), self._out_spec)
