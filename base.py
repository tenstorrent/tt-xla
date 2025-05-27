# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Base class for all model implementations in tt-forge-models
"""
from abc import ABC, abstractmethod
import torch


class ForgeModel(ABC):
    """Base class for all model implementations that can be shared across Tenstorrent projects."""

    @classmethod
    @abstractmethod
    def load_model(cls, **kwargs):
        """Load and return the model instance.

        Returns:
            torch.nn.Module: The model instance
        """
        pass

    @classmethod
    @abstractmethod
    def load_inputs(cls, **kwargs):
        """Load and return sample inputs for the model.

        Returns:
            Any: Sample inputs that can be fed to the model
        """
        pass

    @classmethod
    @abstractmethod
    def decode_output(cls, **kwargs):
        """Load and return sample inputs for the model.

        Returns:
            Any: Output will be Decoded from the model
        """
        pass
