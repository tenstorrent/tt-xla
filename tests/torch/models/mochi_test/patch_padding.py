# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Utility to patch CogVideoXCausalConv3d layers to use constant padding instead of replicate/reflect.

This avoids L1 memory errors caused by replicate padding's complex gather/embedding lowering.
"""

from typing import Optional

import torch
import torch.nn.functional as F


def replace_padding_to_constant(model):
    """
    Recursively find all CogVideoXCausalConv3d layers and replace their padding mode
    from replicate/reflect to constant (zeros).

    This modifies the model in-place by replacing the fake_context_parallel_forward method
    to use constant padding instead of replicate padding.

    Args:
        model: PyTorch model (e.g., VAE decoder)

    Returns:
        Number of layers patched
    """
    import types

    patched_count = 0

    for name, module in model.named_modules():
        # Check if this is a CogVideoXCausalConv3d layer
        if "CogVideoXCausalConv3d" in type(module).__name__:
            old_pad_mode = getattr(module, "pad_mode", "unknown")

            # Replace the fake_context_parallel_forward method to use constant padding
            def new_fake_context_parallel_forward(
                self, inputs: torch.Tensor, conv_cache: Optional[torch.Tensor] = None
            ) -> torch.Tensor:
                # Use constant padding instead of replicate
                inputs = F.pad(
                    inputs, self.time_causal_padding, mode="constant", value=0.0
                )
                return inputs

            # Bind the new method to this specific module instance
            module.fake_context_parallel_forward = types.MethodType(
                new_fake_context_parallel_forward, module
            )

            patched_count += 1
            print(
                f"Patched {name}: padding '{old_pad_mode}' → constant (via fake_context_parallel_forward)"
            )

    if patched_count == 0:
        print(f"No CogVideoXCausalConv3d layers found to patch")
    else:
        print(f"\nSuccessfully patched {patched_count} CogVideoXCausalConv3d layer(s)")

    return patched_count
