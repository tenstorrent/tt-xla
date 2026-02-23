# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from tests.infra.testers.single_chip.graph.graph_tester import run_graph_test
import torch
from infra import Framework, run_graph_test_with_random_inputs
from infra.evaluators import ComparisonConfig
from utils import Category


class SimpleConv3d(torch.nn.Module):
    """Simple Conv3d wrapper for testing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.conv = torch.nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            dtype=torch.bfloat16,
        )

    def forward(self, x):
        return self.conv(x)

class SpecificConv3D(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        
        # self.conv = torch.nn.Conv3d(
        #     in_channels=32,
        #     out_channels=32,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0,
        #     bias=False,
        #     dtype=torch.bfloat16,
        #     weight=conv_weight
        # )
        # self.conv = torch.nn.functional.conv3d
        # self.conv_weight = torch.ones(32, 32, 1 , 1, 1).to(torch.bfloat16)
        # index_weights = torch.zeros(32, 32, 1, 1, 1, dtype=torch.bfloat16)
        # for i in range(32):
        #     index_weights[i, i, 0, 0, 0] = float(i)
        # identity_weights = torch.zeros(32, 32, 1, 1, 1, dtype=torch.bfloat16)
        # for i in range(32):
        #     identity_weights[i, i, 0, 0, 0] = 1.0

        # random_conv_weight = torch.randn(32, 32, 1 , 1, 1).to(torch.bfloat16)
        out_channels = 32
        in_channels = 32
        alt_weights = torch.zeros(out_channels, in_channels, 1, 1, 1, dtype=torch.bfloat16)
        for i in range(out_channels):
            for j in range(in_channels):
                alt_weights[i, j, 0, 0, 0] = float(j)
        self.conv_weight = alt_weights


    def forward(self, x):
        return torch.nn.functional.conv3d(x, self.conv_weight)


def test_numerics():
    # NCDHW -> 1x32x1x2x2
    in_channels = 32
    inputs = torch.arange(9).reshape(3,3)
    inputs = inputs.unsqueeze(0)
    inputs = inputs.repeat(in_channels,1,1)
    inputs = inputs.unsqueeze(1)
    inputs = inputs.unsqueeze(0)
    inputs = inputs.to(torch.bfloat16)
    
    
    model = SpecificConv3D()
    run_graph_test(
        model,
        [inputs],
        framework=Framework.TORCH,
        comparison_config=ComparisonConfig(),  # Uses default PCC=0.99
    )


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,input_shape",
    [
        # Small configurations
        (1, 1, 3, (1, 1, 3, 8, 8)),  # Minimal: 1→1 channels
        (2, 2, 3, (1, 2, 3, 8, 8)),  # Small: 2→2 channels
        (3, 3, 3, (1, 3, 3, 8, 8)),  # RGB-like: 3→3 channels
        (4, 8, 3, (1, 4, 3, 8, 8)),  # Small expansion: 4→8
        (8, 16, 3, (1, 8, 3, 8, 8)),  # Medium expansion: 8→16
        # Medium configurations
        (16, 32, 3, (1, 16, 3, 16, 16)),  # 16→32 channels, larger spatial
        (32, 64, 3, (1, 32, 3, 16, 16)),  # 32→64 channels
        # Asymmetric spatial dimensions
        (8, 16, 3, (1, 8, 3, 8, 16)),  # Different H and W
        # (8, 16, 3, (1, 8, 3, 16, 8)),  # Different H and W (reversed) - breaks?
        # Different temporal depth
        (8, 16, 3, (1, 8, 5, 8, 8)),  # Depth=5
        # Different kernel sizes
        (8, 16, 1, (1, 8, 3, 8, 8)),  # 1×1×1 kernel (pointwise)
        (8, 16, 5, (1, 8, 5, 12, 12)),  # 5×5×5 kernel
    ],
    ids=[
        "minimal_1to1",
        "small_2to2",
        "rgb_3to3",
        "expand_4to8",
        "expand_8to16",
        "medium_16to32",
        "medium_32to64",
        "asymmetric_hw1",
        # "asymmetric_hw2", // breaks?
        "temporal_depth5",
        "kernel1x1x1",
        "kernel5x5x5",
    ],
)
def test_conv3d_simple(in_channels, out_channels, kernel_size, input_shape):
    """
    Test Conv3d with various simple configurations.

    Tests different combinations of:
    - Channel counts (1, 2, 3, 4, 8, 16, 32, 64)
    - Spatial dimensions (8×8, 16×16, asymmetric)
    - Temporal depths (1, 3, 5)
    - Kernel sizes (1, 3, 5)

    All tests use:
    - stride=1
    - padding=0
    - bfloat16 dtype
    """
    model = SimpleConv3d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=1,
        padding=0,
    )

    run_graph_test_with_random_inputs(
        model,
        [input_shape],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        comparison_config=ComparisonConfig(),  # Uses default PCC=0.99
    )


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.parametrize(
    "stride,padding",
    [
        (1, 0),  # No padding, stride 1
        (1, 1),  # Padding=1, stride 1
        (2, 0),  # No padding, stride 2
        (2, 1),  # Padding=1, stride 2
    ],
    ids=[
        "stride1_pad0",
        "stride1_pad1",
        "stride2_pad0",
        "stride2_pad1",
    ],
)
def test_conv3d_stride_padding(stride, padding):
    """
    Test Conv3d with different stride and padding configurations.

    Fixed config:
    - in_channels=8, out_channels=16
    - kernel_size=3
    - input_shape=(1, 8, 4, 16, 16)

    Tests combinations of:
    - stride: 1, 2
    - padding: 0, 1
    """
    model = SimpleConv3d(
        in_channels=8,
        out_channels=16,
        kernel_size=3,
        stride=stride,
        padding=padding,
    )

    run_graph_test_with_random_inputs(
        model,
        [(1, 8, 4, 16, 16)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        comparison_config=ComparisonConfig(),
    )

@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
def test_conv3d_simple_32ch():
    """
    Test Conv3d with 32-channel configuration from simple_conv3d_reproducer.py.

    Configuration:
    - in_channels=32, out_channels=96
    - kernel_size=3, stride=1, padding=0
    - input_shape=(1, 32, 3, 258, 258)

    Expected output: (1, 96, 1, 256, 256)
    """
    model = SimpleConv3d(
        in_channels=32,
        out_channels=96,
        kernel_size=3,
        stride=1,
        padding=0,
    )

    run_graph_test_with_random_inputs(
        model,
        [(1, 32, 3, 258, 258)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        comparison_config=ComparisonConfig(),
    )

@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.parametrize(
    "batch_size",
    [1, 2, 4],
    ids=["batch1", "batch2", "batch4"],
)
def test_conv3d_batch_size(batch_size):
    """
    Test Conv3d with different batch sizes.

    Fixed config:
    - in_channels=8, out_channels=16
    - kernel_size=3, stride=1, padding=0
    - spatial: (3, 8, 8)

    Tests batch sizes: 1, 2, 4
    """
    model = SimpleConv3d(
        in_channels=8,
        out_channels=16,
        kernel_size=3,
        stride=1,
        padding=0,
    )

    run_graph_test_with_random_inputs(
        model,
        [(batch_size, 8, 3, 8, 8)],
        dtype=torch.bfloat16,
        framework=Framework.TORCH,
        comparison_config=ComparisonConfig(),
    )
