# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PHI1_5 model implementation for Tenstorrent projects.

This module provides PHI1_5 model loaders for different tasks:
- causal_lm: Causal language modeling
- token_classification: Token classification
- sequence_classification: Sequence classification

Example usage:
    from phi1_5.causal_lm.pytorch import ModelLoader as CausalLMLoader
    from phi1_5.token_classification.pytorch import ModelLoader as TokenClassificationLoader
    from phi1_5.sequence_classification.pytorch import ModelLoader as SequenceClassificationLoader
"""

# Import from different task implementations
from . import causal_lm
from . import token_classification
from . import sequence_classification

# For backward compatibility, import the token classification as default
# (since the original loader.py was for token classification)
from .token_classification.pytorch import ModelLoader
