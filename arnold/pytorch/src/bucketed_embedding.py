# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Bucketed embedding layer for game variables.
"""
import torch.nn as nn


class BucketedEmbedding(nn.Embedding):
    """Embedding layer that buckets input indices to reduce vocabulary size."""

    def __init__(self, bucket_size, num_embeddings, *args, **kwargs):
        self.bucket_size = bucket_size
        real_num_embeddings = (num_embeddings + bucket_size - 1) // bucket_size
        super(BucketedEmbedding, self).__init__(real_num_embeddings, *args, **kwargs)

    def forward(self, indices):
        # Use integer division to bucket indices
        # Convert to long to ensure integer dtype for embedding layer
        bucketed_indices = (indices // self.bucket_size).long()
        return super(BucketedEmbedding, self).forward(bucketed_indices)
