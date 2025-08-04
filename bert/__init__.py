# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BERT model implementation for Tenstorrent projects.

This module provides BERT model loaders for different tasks:
- masked_lm: Masked Language Modeling
- question_answering: Question Answering
- sequence_classification: Sequence Classification
- token_classification: Token Classification
- sentence_embedding_generation: Sentence Embedding Generation

Example usage:
    from bert.masked_lm.pytorch import ModelLoader as MaskedLMLoader
    from bert.question_answering.pytorch import ModelLoader as QuestionAnsweringLoader
    from bert.sequence_classification.pytorch import ModelLoader as SequenceClassificationLoader
    from bert.token_classification.pytorch import ModelLoader as TokenClassificationLoader
    from bert.sentence_embedding_generation.pytorch import ModelLoader as SentenceEmbeddingGenerationLoader
"""

# Import from different task implementations
from . import masked_lm
from . import question_answering
from . import sequence_classification
from . import token_classification
from . import sentence_embedding_generation
