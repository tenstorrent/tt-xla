# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PHI2 model implementation for Tenstorrent projects.

This module provides PHI2 model loaders for different tasks:
- causal_lm: Causal language modeling
- token_classification: Token classification
- sequence_classification: Sequence classification

Example usage:
    from phi2.causal_lm.pytorch import ModelLoader as CausalLMLoader
    from phi2.token_classification.pytorch import ModelLoader as TokenClassificationLoader
    from phi2.sequence_classification.pytorch import ModelLoader as SequenceClassificationLoader
"""

# Import from different task implementations
from . import causal_lm
from . import token_classification
from . import sequence_classification

# For backward compatibility, import the causal LM as default
# (since phi2 is primarily known for causal language modeling)
from .causal_lm.pytorch import ModelLoader
