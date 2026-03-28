# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama 4 Maverick multimodal PyTorch model implementation for Tenstorrent projects.
"""

from .loader import ModelLoader, ModelVariant

__all__ = ["ModelLoader", "ModelVariant"]
