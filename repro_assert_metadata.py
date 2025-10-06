# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


class PositionIdsModel(torch.nn.Module):
    def __init__(self, padding_idx=1):
        super().__init__()
        self.padding_idx = padding_idx

    def forward(self, input_ids):
        """
        Reproduces the create_position_ids_from_input_ids function from XLM-RoBERTa
        that causes the _assert_tensor_metadata issue.

        This is the exact code from modeling_xlm_roberta.py lines 1562-1564:
        """
        # Line 1562: mask = input_ids.ne(padding_idx).int()
        mask = input_ids.ne(self.padding_idx).int()

        # Line 1563: incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        # Note: past_key_values_length is 0 in our case
        cumsum_result = torch.cumsum(mask, dim=1)  # This creates Int64 tensor
        incremental_indices = cumsum_result.type_as(mask) * mask

        # Line 1564: return incremental_indices.long() + padding_idx
        position_ids = incremental_indices.long() + self.padding_idx

        return position_ids


# --------------------------------
# Test run
# --------------------------------
def test_position_ids():
    # Instantiate model.
    model = PositionIdsModel(padding_idx=1)

    # Put it in inference mode and compile it.
    model = model.eval()
    model.compile(backend="tt")  # openxla, tt

    # Generate typical transformer input (batch_size=1, seq_len=5)
    # Using token ids: [0, 5, 10, 15, 1] where 1 is padding
    input_ids = torch.tensor([[0, 5, 10, 15, 1]], dtype=torch.long)

    # Connect the device.
    device = "xla"  # xla

    # Move inputs and model to device.
    input_ids = input_ids.to(device)
    model = model.to(device)

    print(f"Input shape: {input_ids.shape}")
    print(f"Input: {input_ids}")

    # Run model (with no gradient calculation since we only need inference).
    with torch.no_grad():
        position_ids = model(input_ids)

    print(f"Position IDs: {position_ids}")


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("CPU")  # TT

    test_position_ids()
