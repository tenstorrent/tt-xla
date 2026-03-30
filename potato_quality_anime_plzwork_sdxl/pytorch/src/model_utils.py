# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for Potato Quality Anime SDXL model loading and processing.

This model is an SDXL fine-tune, so we reuse the Stable Diffusion XL utilities.
"""

from stable_diffusion_xl.pytorch.src.model_utils import (
    load_pipe,
    stable_diffusion_preprocessing_xl,
)

__all__ = ["load_pipe", "stable_diffusion_preprocessing_xl"]
