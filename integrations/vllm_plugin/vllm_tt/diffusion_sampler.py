# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC
"""DiffusionSampler for DiffusionGemma custom sampling logic."""

import torch
import torch.nn as nn
from vllm.v1.outputs import SamplerOutput

from .metadata import XLASupportedSamplingMetadata
from .sampler import Sampler


class DiffusionSampler(Sampler):
    """Sampler for DiffusionGemma discrete diffusion decoding.

    Inherits from Sampler and reuses its core sampling operations. The main
    difference is that DiffusionGemma's token selection is handled by
    diffusion_sample_step (which applies temperature schedule, Gumbel-max,
    confidence thresholding, and accept/renoise logic). This sampler is used for:
    1. Rejection sampling in spec-decode paths (gap 5)
    2. Logprobs computation for inference analysis

    The diffusion_sample_step (gap 2) already computes the final tokens, so
    this sampler is primarily used as a component of the rejection sampler
    for spec-decode integration and for logprobs gathering.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: XLASupportedSamplingMetadata,
    ) -> SamplerOutput:
        """Sample tokens using the parent Sampler's logic.

        For DiffusionGemma, logits passed here are either:
        1. Scaled logits from diffusion_sample_step (gap 2) for regular sampling
        2. Bonus logits for spec-decode rejection sampling (gap 5)

        Args:
            logits: [num_tokens, vocab] logits to sample from
            sampling_metadata: Sampling configuration

        Returns:
            SamplerOutput with sampled_token_ids [num_tokens, 1]
        """
        # Delegate to parent Sampler's forward implementation
        return super().forward(logits, sampling_metadata)
