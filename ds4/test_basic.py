# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Basic pytest tests for DeepSeek V4 Flash modules using run_graph_test infrastructure.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch_xla.runtime as xr

# Add ds4 directory to path for imports
ds4_path = Path(__file__).parent
if str(ds4_path) not in sys.path:
    sys.path.insert(0, str(ds4_path))

from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from model import ModelArgs, ParallelEmbedding, Linear, ParallelHead, RMSNorm


def init_weights(module):
    """Initialize weights with random values to avoid NaN/inf from uninitialized memory."""
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.zeros_(module.bias)
    # Initialize Compressor's APE (Absolute Positional Encoding) parameter
    if hasattr(module, 'ape') and module.ape is not None:
        nn.init.normal_(module.ape, mean=0.0, std=0.02)
    # Initialize Hyper-Connections parameters
    if hasattr(module, 'hc_fn') and module.hc_fn is not None:
        nn.init.normal_(module.hc_fn, mean=0.0, std=0.02)
    if hasattr(module, 'hc_scale') and module.hc_scale is not None:
        nn.init.ones_(module.hc_scale)
    if hasattr(module, 'hc_base') and module.hc_base is not None:
        nn.init.zeros_(module.hc_base)


# ============================================================================
# ParallelEmbedding Tests
# ============================================================================


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [16, 32])
def test_parallel_embedding_mini(batch_size, seq_len):
    """
    Test ParallelEmbedding module with reduced dimensions.

    ParallelEmbedding is a vocabulary-sharded embedding layer.
    For single-device (world_size=1), it behaves like a standard embedding.
    """
    xr.set_device_type("TT")

    # Reduced vocab and dim for fast testing
    vocab_size = 1024
    dim = 256

    embedding = ParallelEmbedding(vocab_size, dim)
    embedding.apply(init_weights)
    embedding = embedding.to(torch.bfloat16)
    embedding.eval()

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.99),
    )

    run_graph_test(
        embedding,
        [input_ids],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
    )


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [128, 512])
def test_parallel_embedding(batch_size, seq_len):
    """
    Test ParallelEmbedding module with full DeepSeek V4 dimensions.
    """
    xr.set_device_type("TT")

    # Full DeepSeek V4 dimensions
    args = ModelArgs()
    vocab_size = args.vocab_size  # 129280
    dim = args.dim  # 4096

    embedding = ParallelEmbedding(vocab_size, dim)
    embedding.apply(init_weights)
    embedding = embedding.to(torch.bfloat16)
    embedding.eval()

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.99),
    )

    run_graph_test(
        embedding,
        [input_ids],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
    )


# ============================================================================
# Linear Tests
# ============================================================================


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [16, 32])
def test_linear_mini(batch_size, seq_len):
    """
    Test Linear module with reduced dimensions.

    Linear supports BF16, FP8, and FP4 weight formats.
    For BF16 (default when dtype is not fp8), it uses standard F.linear.
    """
    xr.set_device_type("TT")

    # Reduced dimensions for fast testing
    in_features = 256
    out_features = 512

    linear = Linear(in_features, out_features, dtype=torch.bfloat16)
    linear.apply(init_weights)
    linear = linear.to(torch.bfloat16)
    linear.eval()

    x = torch.randn(batch_size, seq_len, in_features, dtype=torch.bfloat16)

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.99),
    )

    run_graph_test(
        linear,
        [x],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
    )


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [128, 512])
def test_linear(batch_size, seq_len):
    """
    Test Linear module with full DeepSeek V4 dimensions.
    """
    xr.set_device_type("TT")

    # Full DeepSeek V4 dimensions (e.g., wq_a projection)
    args = ModelArgs()
    in_features = args.dim  # 4096
    out_features = args.q_lora_rank  # 1024

    linear = Linear(in_features, out_features, dtype=torch.bfloat16)
    linear.apply(init_weights)
    linear = linear.to(torch.bfloat16)
    linear.eval()

    x = torch.randn(batch_size, seq_len, in_features, dtype=torch.bfloat16)

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.99),
    )

    run_graph_test(
        linear,
        [x],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
    )


# ============================================================================
# ParallelHead (LM Head) Tests
# ============================================================================


class ParallelHeadGetLogits(nn.Module):
    """Wrapper to test just the get_logits method of ParallelHead."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.head = ParallelHead(vocab_size, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head.get_logits(x)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [16, 32])
def test_parallel_head_get_logits_mini(batch_size, seq_len):
    """
    Test ParallelHead.get_logits with reduced dimensions.

    get_logits extracts the last token and projects to vocab size.
    Input: [batch, seq_len, dim] -> Output: [batch, vocab_size]
    """
    xr.set_device_type("TT")

    # Reduced dimensions for fast testing
    vocab_size = 1024
    dim = 256

    model = ParallelHeadGetLogits(vocab_size, dim)
    model.apply(init_weights)
    model.eval()

    x = torch.randn(batch_size, seq_len, dim, dtype=torch.bfloat16)

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.99),
    )

    run_graph_test(
        model,
        [x],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
    )


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len", [128, 512])
def test_parallel_head_get_logits(batch_size, seq_len):
    """
    Test ParallelHead.get_logits with full DeepSeek V4 dimensions.
    """
    xr.set_device_type("TT")

    # Full DeepSeek V4 dimensions
    args = ModelArgs()
    vocab_size = args.vocab_size  # 129280
    dim = args.dim  # 4096

    model = ParallelHeadGetLogits(vocab_size, dim)
    model.apply(init_weights)
    model.eval()

    x = torch.randn(batch_size, seq_len, dim, dtype=torch.bfloat16)

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.99),
    )

    run_graph_test(
        model,
        [x],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
    )


class ParallelHeadHCHead(nn.Module):
    """Wrapper to test just the hc_head method of ParallelHead."""

    def __init__(self, vocab_size: int, dim: int, hc_mult: int = 4):
        super().__init__()
        self.head = ParallelHead(vocab_size, dim)
        self.hc_mult = hc_mult
        hc_dim = hc_mult * dim
        # Initialize HC parameters
        self.hc_fn = nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head.hc_head(x, self.hc_fn, self.hc_scale, self.hc_base)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [16, 32])
def test_parallel_head_hc_head_mini(batch_size, seq_len):
    """
    Test ParallelHead.hc_head (Hyper-Connections head) with reduced dimensions.

    hc_head combines multiple hidden states via learned mixing weights.
    Input: [batch, seq_len, hc_mult, dim] -> Output: [batch, seq_len, dim]
    """
    xr.set_device_type("TT")

    # Reduced dimensions for fast testing
    vocab_size = 1024
    dim = 256
    hc_mult = 4

    model = ParallelHeadHCHead(vocab_size, dim, hc_mult)
    model.apply(init_weights)
    model.eval()

    # Input has shape [batch, seq_len, hc_mult, dim]
    x = torch.randn(batch_size, seq_len, hc_mult, dim, dtype=torch.bfloat16)

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.99),
    )

    run_graph_test(
        model,
        [x],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
    )


class ParallelHeadFull(nn.Module):
    """Wrapper to test the full ParallelHead forward pass."""

    def __init__(self, vocab_size: int, dim: int, hc_mult: int = 4, norm_eps: float = 1e-6):
        super().__init__()
        self.head = ParallelHead(vocab_size, dim, norm_eps=norm_eps)
        self.norm = RMSNorm(dim, norm_eps)
        self.hc_mult = hc_mult
        hc_dim = hc_mult * dim
        # Initialize HC parameters
        self.hc_fn = nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x, self.hc_fn, self.hc_scale, self.hc_base, self.norm)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("seq_len", [16, 32])
def test_parallel_head_full_mini(batch_size, seq_len):
    """
    Test full ParallelHead forward with reduced dimensions.

    Full forward: hc_head -> norm -> get_logits
    Input: [batch, seq_len, hc_mult, dim] -> Output: [batch, vocab_size]
    """
    xr.set_device_type("TT")

    # Reduced dimensions for fast testing
    vocab_size = 1024
    dim = 256
    hc_mult = 4

    model = ParallelHeadFull(vocab_size, dim, hc_mult)
    model.apply(init_weights)
    model.eval()

    # Input has shape [batch, seq_len, hc_mult, dim]
    x = torch.randn(batch_size, seq_len, hc_mult, dim, dtype=torch.bfloat16)

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.99),
    )

    run_graph_test(
        model,
        [x],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
    )
