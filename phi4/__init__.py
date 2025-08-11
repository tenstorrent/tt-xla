# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PHI4 model implementation for Tenstorrent projects.

This module provides PHI4 model loaders for different tasks:
- causal_lm: Causal language modeling
- token_cls: Token classification
- seq_cls: Sequence classification

Example usage:
    from phi4.causal_lm.pytorch import ModelLoader as CausalLMLoader
    from phi4.token_cls.pytorch import ModelLoader as TokenClassificationLoader
    from phi4.seq_cls.pytorch import ModelLoader as SequenceClassificationLoader
"""

# Import from different task implementations
from . import causal_lm
from . import token_cls
from . import seq_cls

# For backward compatibility, import the causal LM as default
from .causal_lm.pytorch import ModelLoader
