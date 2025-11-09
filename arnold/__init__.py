# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Arnold DQN model implementation."""
# Import from the PyTorch implementation by default
from .pytorch import ModelLoader, ModelVariant

__all__ = ["ModelLoader", "ModelVariant"]
