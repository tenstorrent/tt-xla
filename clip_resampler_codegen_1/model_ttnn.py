# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN model wrapper for CLIP Vision Encoder + IP-Adapter Resampler."""

import main
import ttnn
import utils
from models.common.lightweightmodule import LightweightModule


class CLIPVisionEncoderAndResamplerTTNN(LightweightModule):
    """CLIP Vision Encoder + IP-Adapter Resampler TTNN model."""

    # Index of pixel_values in the inputs list
    PIXEL_VALUES_INDEX = 390

    def __init__(self, device):
        """
        Initialize the model by loading weights.

        Args:
            device: TTNN device
        """
        self.device = device

        # Load all inputs (weights + original input tensor)
        self._inputs = main.load_inputs_for__main()

    def forward(self, pixel_values):
        """
        Run the CLIP + Resampler model.

        Args:
            pixel_values: Input image tensor [batch, channels, height, width]
                         Must be a host TTNN tensor (device=None)

        Returns:
            List containing the output tensor
        """
        # Move input to device
        assert pixel_values.device() is None, "pixel_values must be on host"
        pixel_values_device = ttnn.to_device(
            pixel_values,
            self.device,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )

        # Replace input at PIXEL_VALUES_INDEX with the provided pixel_values
        inputs = list(self._inputs)
        inputs[self.PIXEL_VALUES_INDEX] = pixel_values_device

        # Run the model
        return main._main(inputs)
