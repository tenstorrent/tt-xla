# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MNIST model loader implementation
"""

from ...base import ForgeModel
from .src.utils import load_model, load_input


class ModelLoader(ForgeModel):
    """Loads MNIST model and sample input."""

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load pretrained MNIST model."""
        model = load_model()
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None):
        """Prepare sample input for MNIST model"""

        inputs = load_input()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
