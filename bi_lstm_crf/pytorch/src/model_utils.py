# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for BiLSTM-CRF model loading and processing.
"""

import torch
from bi_lstm_crf import BiRnnCrf


def get_vocab_mappings():
    """Get vocabulary and tag mappings for BiLSTM-CRF model.

    Returns:
        tuple: (word_to_ix, tag_to_ix) - Dictionary mappings for words and tags
    """
    word_to_ix = {
        "apple": 0,
        "corporation": 1,
        "is": 2,
        "in": 3,
        "georgia": 4,
        "<PAD>": 5,
    }
    tag_to_ix = {"B": 0, "I": 1, "O": 2}
    return word_to_ix, tag_to_ix


def create_bi_lstm_crf_model():
    """Create BiLSTM-CRF model with default configuration.

    Returns:
        BiRnnCrf: Configured BiLSTM-CRF model instance
    """
    word_to_ix, tag_to_ix = get_vocab_mappings()

    # Default model configuration
    embedding_dim = 5
    hidden_dim = 4
    num_rnn_layers = 1

    model = BiRnnCrf(
        vocab_size=len(word_to_ix),
        tagset_size=len(tag_to_ix),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_rnn_layers=num_rnn_layers,
        rnn="lstm",
    )

    return model


def create_sample_input(test_sentence=None):
    """Create sample input tensor from test sentence.

    Args:
        test_sentence: List of words to convert to tensor (optional)

    Returns:
        torch.Tensor: Input tensor with word indices
    """
    if test_sentence is None:
        test_sentence = ["apple", "corporation", "is", "in", "georgia"]

    word_to_ix, _ = get_vocab_mappings()

    # Validate words are in vocabulary
    for word in test_sentence:
        assert word in word_to_ix, f"Error: '{word}' is not in dictionary!"

    # Convert to tensor indices
    test_input = torch.tensor(
        [[word_to_ix[w] for w in test_sentence]], dtype=torch.long
    )

    return test_input
