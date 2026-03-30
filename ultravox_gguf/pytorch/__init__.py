# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ultravox v0.5 Llama 3.1 8B GGUF PyTorch model loader implementation.
"""
from .loader import ModelLoader, ModelVariant

__all__ = ["ModelLoader", "ModelVariant"]
