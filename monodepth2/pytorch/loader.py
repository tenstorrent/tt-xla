# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Monodepth2 model loader implementation
"""

from ...base import ForgeModel
from .src.utils import load_model, load_input


class ModelLoader(ForgeModel):
    """Loads Monodepth2 model and sample input."""

    # Shared configuration parameters
    model_name = "mono_640x192"

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load pretrained Monodepth2 model."""
        model, height, width = load_model(cls.model_name)
        model.eval()

        cls._height = height
        cls._width = width

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None):
        """Prepare sample input for Monodepth2 model"""

        inputs = load_input(cls._height, cls._width)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
