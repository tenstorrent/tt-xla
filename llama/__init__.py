# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Llama model implementation for Tenstorrent projects.
"""
# Import from the causal LM implementation by default for backward compatibility
from .causal_lm.pytorch import ModelLoader
