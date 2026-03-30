# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.evaluators import TorchComparisonEvaluator
from infra.utilities.types import Framework
from torch.nn import functional as F
from tt_torch.composite_ops import (
    composite_gelu,
    composite_group_norm,
    composite_layer_norm,
    composite_rms_norm,
    composite_scaled_dot_product_attention,
    composite_topk,
    composite_topk_indices,
    composite_topk_values,
)

from tests.infra.evaluators.evaluation_config import ComparisonConfig
from tests.infra.testers.single_chip.graph.graph_tester import run_graph_test


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize("approx", ["none", "tanh"])
def test_composite_gelu(approx):
    class GeluModel(torch.nn.Module):
        def forward(self, x):
            return composite_gelu(x, approx)

    options = {"tt_enable_composite_ops": False}

    input = torch.randn(32, 32)
    model = GeluModel()

    # Disable inplace buffers for inductor compilation
    # so that we can compare the results with the golden model.
    with torch._inductor.config.patch({"inplace_buffers": False}):
        run_graph_test(
            model,
            [input],
            comparison_config=ComparisonConfig(),
            framework=Framework.TORCH,
            torch_options=options,
        )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize("approx", ["none", "tanh"])
def test_patched_gelu_functional(approx):
    class GeluModel(torch.nn.Module):
        def forward(self, x):
            return F.gelu(input=x, approximate=approx)

    options = {"tt_enable_composite_ops": True}

    input = torch.randn(32, 32)
    model = GeluModel()

    run_graph_test(
        model,
        [input],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        torch_options=options,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize("use_weight", [True, False])
@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size", [(1, 32, 32), (1, 128, 768), (1, 1024, 768)]
)
def test_patched_rms_norm_functional_single_device(
    use_weight, batch_size, seq_len, hidden_size
):

    class RMSNormModel(torch.nn.Module):
        def __init__(self, normalized_shape):
            super().__init__()
            self.normalized_shape = normalized_shape

        def forward(self, x, weight):
            return torch.nn.functional.rms_norm(x, (self.normalized_shape,), weight)

    options = {"tt_enable_composite_ops": True}

    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    weight = torch.randn(hidden_size) if use_weight else None
    model = RMSNormModel(hidden_size)

    run_graph_test(
        model,
        [input_tensor, weight],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        torch_options=options,
    )


@pytest.mark.nightly
@pytest.mark.xfail(
    reason="To be investigated - https://github.com/tenstorrent/tt-xla/issues/4138"
)
@pytest.mark.dual_chip
@pytest.mark.parametrize("use_weight", [True, False])
@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size", [(1, 32, 32), (1, 128, 768), (1, 1024, 768)]
)
def test_patched_rms_norm_functional_batch_parallel(
    use_weight, batch_size, seq_len, hidden_size
):

    class RMSNormModel(torch.nn.Module):
        def __init__(self, normalized_shape):
            super().__init__()
            self.normalized_shape = normalized_shape

        def forward(self, x, weight):
            return torch.nn.functional.rms_norm(x, (self.normalized_shape,), weight)

    options = {"tt_enable_composite_ops": True}

    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    weight = torch.randn(hidden_size) if use_weight else None
    model = RMSNormModel(hidden_size)

    # Create a mesh.
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = xs.Mesh(device_ids, mesh_shape, ("model", "batch"))

    # Mark sharding for inputs along batch dimension.
    shard_specs = {}
    shard_specs[input_tensor] = ("batch", None)
    if use_weight:
        shard_specs[weight] = (None,)

    run_graph_test(
        model,
        [input_tensor, weight],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        torch_options=options,
        mesh=mesh,
        shard_spec_fn=shard_specs,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize("use_weight", [True, False])
@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size", [(1, 32, 32), (1, 128, 768), (1, 1024, 768)]
)
def test_composite_rms_norm(use_weight, batch_size, seq_len, hidden_size):

    class RMSNormModel(torch.nn.Module):
        def __init__(self, normalized_shape):
            super().__init__()
            self.normalized_shape = normalized_shape

        def forward(self, x, weight):
            return composite_rms_norm(x, (self.normalized_shape,), weight)

    options = {"tt_enable_composite_ops": False}

    input_tensor = torch.randn(batch_size, seq_len, hidden_size)
    weight = torch.randn(hidden_size) if use_weight else None
    model = RMSNormModel(hidden_size)

    # Disable inplace buffers for inductor compilation
    # so that we can compare the results with the golden model.
    with torch._inductor.config.patch({"inplace_buffers": False}):
        run_graph_test(
            model,
            [input_tensor, weight],
            comparison_config=ComparisonConfig(),
            framework=Framework.TORCH,
            torch_options=options,
        )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize("elementwise_affine", [True, False])
@pytest.mark.parametrize(
    "batch_size, seq_len, embedding_dim",
    [(1, 32, 32), (1, 197, 768), (1, 1024, 768)],
)
def test_patched_layer_norm_module(
    elementwise_affine, batch_size, seq_len, embedding_dim
):
    class LayerNormModel(torch.nn.Module):
        def __init__(self, embedding_dim):
            super().__init__()
            self.ln = nn.LayerNorm(embedding_dim, elementwise_affine=elementwise_affine)

        def forward(self, x):
            return self.ln(x)

    options = {"tt_enable_composite_ops": True}

    input_tensor = torch.randn(batch_size, seq_len, embedding_dim, dtype=torch.bfloat16)

    model = LayerNormModel(embedding_dim)

    run_graph_test(
        model,
        [input_tensor],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        torch_options=options,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize(
    "use_weight, use_bias", [(True, True), (True, False), (False, False)]
)
@pytest.mark.parametrize(
    "batch_size, seq_len, embedding_dim",
    [(1, 32, 32), (1, 197, 768), (1, 1024, 768)],
)
def test_patched_layer_norm_functional(
    use_weight, use_bias, batch_size, seq_len, embedding_dim
):

    class LayerNormModel(torch.nn.Module):
        def __init__(self, normalized_shape):
            super().__init__()
            self.normalized_shape = normalized_shape

        def forward(self, x, weight=None, bias=None):
            return F.layer_norm(x, (self.normalized_shape,), weight, bias, eps=1e-5)

    options = {"tt_enable_composite_ops": True}

    input_tensor = torch.randn(batch_size, seq_len, embedding_dim, dtype=torch.bfloat16)
    weight = torch.randn(embedding_dim, dtype=torch.bfloat16) if use_weight else None
    bias = torch.randn(embedding_dim, dtype=torch.bfloat16) if use_bias else None

    model = LayerNormModel(embedding_dim)

    run_graph_test(
        model,
        [input_tensor, weight, bias],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        torch_options=options,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize(
    "use_weight, use_bias", [(True, True), (True, False), (False, False)]
)
@pytest.mark.parametrize(
    "batch_size, seq_len, embedding_dim",
    [(1, 32, 32), (1, 197, 768), (1, 1024, 768)],
)
def test_composite_layer_norm(use_weight, use_bias, batch_size, seq_len, embedding_dim):

    class LayerNormModel(torch.nn.Module):
        def __init__(self, normalized_shape):
            super().__init__()
            self.normalized_shape = normalized_shape

        def forward(self, x, weight=None, bias=None):
            return composite_layer_norm(
                x, self.normalized_shape, weight, bias, eps=1e-5
            )

    options = {"tt_enable_composite_ops": False}

    input_tensor = torch.randn(batch_size, seq_len, embedding_dim, dtype=torch.bfloat16)
    weight = torch.randn(embedding_dim, dtype=torch.bfloat16) if use_weight else None
    bias = torch.randn(embedding_dim, dtype=torch.bfloat16) if use_bias else None

    model = LayerNormModel(embedding_dim)

    # Disable inplace buffers for inductor compilation
    # so that we can compare the results with the golden model.
    with torch._inductor.config.patch({"inplace_buffers": False}):
        run_graph_test(
            model,
            [input_tensor, weight, bias],
            comparison_config=ComparisonConfig(),
            framework=Framework.TORCH,
            torch_options=options,
        )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize(["input_shape", "k"], [((1, 10), 5), ((1, 40), 5)])
def test_composite_topk_indices(input_shape, k):
    class TopK(torch.nn.Module):
        def __init__(self, k):
            super().__init__()
            self.k = k

        def forward(self, x):
            return composite_topk_indices(x, self.k)

    options = {"tt_enable_composite_ops": False}
    input = torch.randn(*input_shape)

    with torch._inductor.config.patch({"inplace_buffers": False}):
        run_graph_test(
            TopK(k),
            [input],
            comparison_config=ComparisonConfig(),
            framework=Framework.TORCH,
            torch_options=options,
        )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize(["input_shape", "k"], [((1, 10), 5), ((1, 40), 5)])
def test_composite_topk_values(input_shape, k):
    class TopK(torch.nn.Module):
        def __init__(self, k):
            super().__init__()
            self.k = k

        def forward(self, x):
            return composite_topk_values(x, self.k)

    options = {"tt_enable_composite_ops": False}
    input = torch.randn(*input_shape)

    with torch._inductor.config.patch({"inplace_buffers": False}):
        run_graph_test(
            TopK(k),
            [input],
            comparison_config=ComparisonConfig(),
            framework=Framework.TORCH,
            torch_options=options,
        )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize(["input_shape", "k"], [((1, 10), 5), ((1, 40), 5)])
def test_composite_topk_both(input_shape, k):
    class TopK(torch.nn.Module):
        def __init__(self, k):
            super().__init__()
            self.k = k

        def forward(self, x):
            return composite_topk(x, self.k)

    options = {"tt_enable_composite_ops": False}
    input = torch.randn(*input_shape)

    with torch._inductor.config.patch({"inplace_buffers": False}):
        run_graph_test(
            TopK(k),
            [input],
            comparison_config=ComparisonConfig(),
            framework=Framework.TORCH,
            torch_options=options,
        )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize(["input_shape", "k"], [((1, 10), 5), ((1, 40), 5)])
def test_patched_topk_indices(input_shape, k):
    """torch.topk patched — only indices output consumed → composite_topk_indices selected."""

    class TopKIndices(torch.nn.Module):
        def __init__(self, k):
            super().__init__()
            self.k = k

        def forward(self, x):
            return torch.topk(x, self.k)[1]

    options = {"tt_enable_composite_ops": True}
    input = torch.randn(*input_shape)

    run_graph_test(
        TopKIndices(k),
        [input],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        torch_options=options,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize(["input_shape", "k"], [((1, 10), 5), ((1, 40), 5)])
def test_patched_topk_values(input_shape, k):
    """torch.topk patched — only values output consumed → composite_topk_values selected."""

    class TopKValues(torch.nn.Module):
        def __init__(self, k):
            super().__init__()
            self.k = k

        def forward(self, x):
            return torch.topk(x, self.k)[0]

    options = {"tt_enable_composite_ops": True}
    input = torch.randn(*input_shape)

    run_graph_test(
        TopKValues(k),
        [input],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        torch_options=options,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize(["input_shape", "k"], [((1, 10), 5), ((1, 40), 5)])
def test_patched_topk_both(input_shape, k):
    """torch.topk patched — both outputs consumed → composite_topk selected."""

    class TopKBoth(torch.nn.Module):
        def __init__(self, k):
            super().__init__()
            self.k = k

        def forward(self, x):
            values, indices = torch.topk(x, self.k)
            return values, indices

    options = {"tt_enable_composite_ops": True}
    input = torch.randn(*input_shape)

    run_graph_test(
        TopKBoth(k),
        [input],
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        torch_options=options,
    )


# TODO: uncomment once https://github.com/tenstorrent/tt-metal/issues/40916 is fixed
# @pytest.mark.single_device
# @pytest.mark.parametrize("affine", [True, False])
# @pytest.mark.parametrize(
#     "batch_size, num_channels, height, width, num_groups",
#     [(1, 32, 8, 8, 8), (1, 64, 16, 16, 16), (1, 128, 32, 32, 32)],
# )
# def test_patched_group_norm_module(
#     affine, batch_size, num_channels, height, width, num_groups
# ):
#     class GroupNormModel(torch.nn.Module):
#         def __init__(self, num_groups, num_channels):
#             super().__init__()
#             self.gn = nn.GroupNorm(num_groups, num_channels, affine=affine)

#         def forward(self, x):
#             return self.gn(x)

#     options = {"tt_enable_composite_ops": True}

#     input_tensor = torch.randn(
#         batch_size, num_channels, height, width, dtype=torch.bfloat16
#     )

#     model = GroupNormModel(num_groups, num_channels)

#     run_graph_test(
#         model,
#         [input_tensor],
#         comparison_config=ComparisonConfig(),
#         framework=Framework.TORCH,
#         torch_options=options,
#     )

# TODO: uncomment once https://github.com/tenstorrent/tt-metal/issues/40916 is fixed
# @pytest.mark.single_device
# @pytest.mark.parametrize(
#     "use_weight, use_bias", [(True, True), (True, False), (False, False)]
# )
# @pytest.mark.parametrize(
#     "batch_size, num_channels, height, width, num_groups",
#     [(1, 32, 8, 8, 8), (1, 64, 16, 16, 16), (1, 128, 32, 32, 32)],
# )
# def test_patched_group_norm_functional(
#     use_weight, use_bias, batch_size, num_channels, height, width, num_groups
# ):

#     class GroupNormModel(torch.nn.Module):
#         def __init__(self, num_groups):
#             super().__init__()
#             self.num_groups = num_groups

#         def forward(self, x, weight=None, bias=None):
#             return F.group_norm(x, self.num_groups, weight, bias, eps=1e-5)

#     options = {"tt_enable_composite_ops": True}

#     input_tensor = torch.randn(
#         batch_size, num_channels, height, width, dtype=torch.bfloat16
#     )
#     weight = torch.randn(num_channels, dtype=torch.bfloat16) if use_weight else None
#     bias = torch.randn(num_channels, dtype=torch.bfloat16) if use_bias else None

#     model = GroupNormModel(num_groups)

#     run_graph_test(
#         model,
#         [input_tensor, weight, bias],
#         comparison_config=ComparisonConfig(),
#         framework=Framework.TORCH,
#         torch_options=options,
#     )


# TODO: uncomment once https://github.com/tenstorrent/tt-metal/issues/40916 is fixed
# @pytest.mark.single_device
# @pytest.mark.parametrize(
#     "use_weight, use_bias", [(True, True), (True, False), (False, False)]
# )
# @pytest.mark.parametrize(
#     "batch_size, num_channels, height, width, num_groups",
#     [(1, 32, 8, 8, 8), (1, 64, 16, 16, 16), (1, 128, 32, 32, 32)],
# )
# def test_composite_group_norm(
#     use_weight, use_bias, batch_size, num_channels, height, width, num_groups
# ):

#     class GroupNormModel(torch.nn.Module):
#         def __init__(self, num_groups):
#             super().__init__()
#             self.num_groups = num_groups

#         def forward(self, x, weight=None, bias=None):
#             return composite_group_norm(x, self.num_groups, weight, bias, eps=1e-5)

#     options = {"tt_enable_composite_ops": False}

#     input_tensor = torch.randn(
#         batch_size, num_channels, height, width, dtype=torch.bfloat16
#     )
#     weight = torch.randn(num_channels, dtype=torch.bfloat16) if use_weight else None
#     bias = torch.randn(num_channels, dtype=torch.bfloat16) if use_bias else None

#     model = GroupNormModel(num_groups)

#     # Disable inplace buffers for inductor compilation
#     # so that we can compare the results with the golden model.
#     with torch._inductor.config.patch({"inplace_buffers": False}):
#         run_graph_test(
#             model,
#             [input_tensor, weight, bias],
#             comparison_config=ComparisonConfig(),
#             framework=Framework.TORCH,
#             torch_options=options,
#         )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("use_attn_mask", [True, False])
@pytest.mark.parametrize(
    "batch_size, num_heads, seq_len, head_dim",
    [(1, 1, 32, 32), (1, 8, 64, 64), (2, 4, 128, 64)],
)
def test_composite_sdpa(
    is_causal, use_attn_mask, batch_size, num_heads, seq_len, head_dim
):
    # is_causal and attn_mask cannot both be set
    if is_causal and use_attn_mask:
        pytest.skip("is_causal and attn_mask cannot both be set")

    class SDPAModel(torch.nn.Module):
        def __init__(self, is_causal):
            super().__init__()
            self.is_causal = is_causal

        def forward(self, query, key, value, attn_mask=None):
            return composite_scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, is_causal=self.is_causal
            )

    options = {"tt_enable_composite_ops": False}

    query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    if use_attn_mask:
        # Additive causal mask: 0 for attend, -inf for mask out
        # TTIR requires 4D mask: [batch, heads, seq_len, seq_len]
        attn_mask = torch.zeros(
            batch_size, num_heads, seq_len, seq_len, dtype=torch.bfloat16
        )
        attn_mask.masked_fill_(
            ~torch.ones(seq_len, seq_len).bool().tril(), float("-inf")
        )
    else:
        attn_mask = None

    model = SDPAModel(is_causal)
    inputs = [query, key, value, attn_mask] if use_attn_mask else [query, key, value]

    with torch._inductor.config.patch({"inplace_buffers": False}):
        run_graph_test(
            model,
            inputs,
            comparison_config=ComparisonConfig(),
            framework=Framework.TORCH,
            torch_options=options,
        )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("use_attn_mask", [True, False])
@pytest.mark.parametrize(
    "batch_size, num_heads, seq_len, head_dim",
    [(1, 1, 32, 32), (1, 8, 64, 64), (2, 4, 128, 64)],
)
def test_patched_sdpa(
    is_causal, use_attn_mask, batch_size, num_heads, seq_len, head_dim
):
    # is_causal and attn_mask cannot both be set
    if is_causal and use_attn_mask:
        pytest.skip("is_causal and attn_mask cannot both be set")

    class SDPAModel(torch.nn.Module):
        def __init__(self, is_causal):
            super().__init__()
            self.is_causal = is_causal

        def forward(self, query, key, value, attn_mask=None):
            return F.scaled_dot_product_attention(
                query, key, value, attn_mask=attn_mask, is_causal=self.is_causal
            )

    options = {"tt_enable_composite_ops": True}

    query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    if use_attn_mask:
        # Additive causal mask: 0 for attend, -inf for mask out
        # TTIR requires 4D mask: [batch, heads, seq_len, seq_len]
        attn_mask = torch.zeros(
            batch_size, num_heads, seq_len, seq_len, dtype=torch.bfloat16
        )
        attn_mask.masked_fill_(
            ~torch.ones(seq_len, seq_len).bool().tril(), float("-inf")
        )
    else:
        attn_mask = None

    model = SDPAModel(is_causal)
    inputs = [query, key, value, attn_mask] if use_attn_mask else [query, key, value]

    print(model(query, key, value))
    print(attn_mask)

    run_graph_test(
        model,
        inputs,
        comparison_config=ComparisonConfig(),
        framework=Framework.TORCH,
        torch_options=options,
    )
