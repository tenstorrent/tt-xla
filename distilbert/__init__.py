# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DistilBERT model implementation for Tenstorrent projects.

This module provides DistilBERT model loaders for different tasks:
- masked_lm: Masked Language Modeling
- question_answering: Question Answering
- sequence_classification: Sequence Classification
- token_classification: Token Classification

Example usage:
    from distilbert.masked_lm.pytorch import ModelLoader as MaskedLMLoader
    from distilbert.question_answering.pytorch import ModelLoader as QuestionAnsweringLoader
    from distilbert.sequence_classification.pytorch import ModelLoader as SequenceClassificationLoader
    from distilbert.token_classification.pytorch import ModelLoader as TokenClassificationLoader
"""

# Import from different task implementations
from . import masked_lm
from . import question_answering
from . import sequence_classification
from . import token_classification
