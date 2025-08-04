# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DPR model implementation for Tenstorrent projects.

This module provides DPR model loaders for different tasks:
- context_encoder: Context encoder
- question_encoder: Question encoder
- reader: Reader

Example usage:
    from dpr.context_encoder.pytorch import ModelLoader as ContextEncoderLoader
    from dpr.question_encoder.pytorch import ModelLoader as QuestionEncoderLoader
    from dpr.reader.pytorch import ModelLoader as ReaderLoader
"""

# Import from different task implementations
from . import context_encoder
from . import question_encoder
from . import reader
