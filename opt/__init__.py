# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OPT model implementation for Tenstorrent projects.

This module provides OPT model loaders for different tasks:
- causal_lm: Causal language modeling
- qa: Question answering
- sequence_classification: Sequence classification

Example usage:
    from opt.causal_lm.pytorch import ModelLoader as CausalLMLoader
    from opt.qa.pytorch import ModelLoader as QALoader
    from opt.sequence_classification.pytorch import ModelLoader as SequenceClassificationLoader
"""

# Import from different task implementations
from . import causal_lm
from . import qa
from . import sequence_classification

# For backward compatibility, import the causal LM as default
from .causal_lm.pytorch import ModelLoader
