# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pytest
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.comparators.torch_comparator import TorchComparator
from infra.connectors.torch_device_connector import TorchDeviceConnector
from torch_xla.distributed.spmd import Mesh

from tests.infra import RunMode, TorchModelTester
from tests.infra.comparators.comparison_config import AtolConfig, ComparisonConfig

# TODO(@LPanosTT): https://github.com/tenstorrent/tt-xla/issues/1137
# We would like to use the OpTester/GraphTester infra instead of manually
# calculating and comparing golden vs device results.


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("bias", [True, False])
def test_simple_mm_eager(bias):
    class MM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 32, bias=bias, dtype=torch.bfloat16)

        def forward(self, x):
            return self.linear(x)

    input_x = torch.randn(32, 32, dtype=torch.bfloat16)

    model = MM()
    golden = model(input_x)

    device = xm.xla_device()
    model = model.to(device)
    input_x = input_x.to(device)

    output = model(input_x).to("cpu")

    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(required_atol=0.02),
        )
    )
    comparator.compare(output, golden)


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("in_channels", [3, 64])
@pytest.mark.parametrize("out_channels", [3, 64])
@pytest.mark.parametrize("kernel_size", [2, 3])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
@pytest.mark.parametrize("dilation", [1, 2])
@pytest.mark.parametrize("bias", [True, False])
def test_conv2d_eager(
    in_channels, out_channels, kernel_size, stride, padding, dilation, bias
):
    class Conv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                1,
                bias,
                dtype=torch.bfloat16,
            )

        def forward(self, x):
            return self.conv(x)

    input_x = torch.randn(1, in_channels, 224, 224, dtype=torch.bfloat16)

    model = Conv()
    golden = model(input_x)

    device = xm.xla_device()
    model = model.to(device)
    input_x = input_x.to(device)

    output = model(input_x).to("cpu")

    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(enabled=False, required_atol=0.02),
        )
    )
    comparator.compare(output, golden)


eltwise_unary_ops = [
    torch.abs,
    torch.acos,
    torch.acosh,
    torch.angle,
    torch.asin,
    torch.asinh,
    torch.atan,
    torch.atanh,
    torch.bitwise_not,
    torch.ceil,
    lambda act: torch.clamp(act, -1, 1),  # needs min and max
    torch.conj_physical,
    torch.cos,
    torch.cosh,
    torch.deg2rad,
    torch.digamma,
    torch.erf,
    torch.erfc,
    torch.erfinv,
    torch.exp,
    torch.exp2,
    torch.expm1,
    torch.fix,
    torch.floor,
    torch.frac,
    torch.lgamma,
    torch.log,
    torch.log10,
    torch.log1p,
    torch.log2,
    torch.logit,
    torch.i0,
    torch.isnan,
    torch.nan_to_num,
    torch.neg,
    torch.negative,
    torch.positive,
    torch.rad2deg,
    torch.reciprocal,
    # torch.round, error: failed to legalize operation 'stablehlo.round_nearest_even'
    torch.rsqrt,
    torch.sigmoid,
    torch.sign,
    torch.sgn,
    torch.signbit,
    torch.sin,
    torch.sinc,
    torch.sinh,
    torch.sqrt,
    torch.square,
    torch.tan,
    torch.tanh,
    torch.trunc,
]

# Operations that fail only in eager mode due to different nan/inf handling
eager_failing_unary_ops = {
    torch.acos,
    torch.acosh,
    torch.asin,
    torch.atanh,
    torch.digamma,
    torch.erfinv,
    torch.log,
    torch.log10,
    torch.log1p,
    torch.log2,
    torch.logit,
    torch.rsqrt,
    torch.sqrt,
    torch.tan,
}

# Create eager version with xfail markers for failing ops
eltwise_unary_ops_eager = [
    (
        pytest.param(
            op,
            marks=pytest.mark.xfail(
                strict=True,
                reason="PCC comparison failed see issue https://github.com/tenstorrent/tt-xla/issues/1555",
            ),
        )
        if op in eager_failing_unary_ops
        else op
    )
    for op in eltwise_unary_ops
]


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("op", eltwise_unary_ops_eager)
def test_eltwise_unary_eager(op):
    class Unary(torch.nn.Module):
        def forward(self, x):
            return op(x)

    input_x = (
        torch.randn(32, 32, dtype=torch.bfloat16)
        if op is not torch.bitwise_not
        else torch.randint(-100, 100, (32, 32))
    )

    model = Unary()
    golden = model(input_x)

    device = xm.xla_device()
    model = model.to(device)
    input_x = input_x.to(device)

    output = model(input_x).to("cpu")

    # Not verifying data as many are wrong. Simply testing compile and execute
    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(enabled=False, required_atol=0.01),
        )
    )
    comparator.compare(output, golden)


eltwise_binary_ops = [
    torch.add,
    torch.atan2,
    torch.arctan2,
    torch.bitwise_and,
    torch.bitwise_or,
    torch.bitwise_xor,
    torch.bitwise_left_shift,
    torch.bitwise_right_shift,
    torch.div,
    torch.divide,
    pytest.param(
        torch.floor_divide,
        marks=pytest.mark.xfail(
            strict=True,
            reason="PCC comparison failed see issue https://github.com/tenstorrent/tt-xla/issues/1555",
        ),
    ),
    pytest.param(
        torch.fmod,
        marks=pytest.mark.xfail(
            strict=True,
            reason="PCC comparison failed see issue https://github.com/tenstorrent/tt-xla/issues/1555",
        ),
    ),
    torch.logaddexp,
    torch.logaddexp2,
    torch.mul,
    torch.multiply,
    torch.nextafter,
    pytest.param(
        torch.remainder,
        marks=pytest.mark.xfail(
            strict=True,
            reason="PCC comparison failed see issue https://github.com/tenstorrent/tt-xla/issues/1555",
        ),
    ),
    torch.sub,
    torch.subtract,
    torch.true_divide,
    torch.eq,
    torch.ne,
    torch.le,
    torch.ge,
    torch.greater,
    torch.greater_equal,
    torch.gt,
    torch.less_equal,
    torch.lt,
    torch.less,
    torch.maximum,
    torch.minimum,
    torch.fmax,
    torch.fmin,
    torch.not_equal,
]


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize("op", eltwise_binary_ops)
def test_eltwise_binary_eager(op):
    if op in [
        torch.bitwise_and,
        torch.bitwise_or,
        torch.bitwise_xor,
    ]:
        input_x = torch.randint(-100, 100, (32, 32))
        input_y = torch.randint(-100, 100, (32, 32))
    elif op in [torch.bitwise_left_shift, torch.bitwise_right_shift]:
        # TODO: enable test for these ops once issues is resolved (https://github.com/tenstorrent/tt-torch/issues/1127)
        pytest.skip(f"{op} not supported in tt backend yet. Skipping test.")
    else:
        input_x = torch.randn(32, 32, dtype=torch.bfloat16)
        input_y = torch.randn(32, 32, dtype=torch.bfloat16)

    class Binary(torch.nn.Module):
        def forward(self, x, y):
            return op(x, y)

    model = Binary()
    golden = model(input_x, input_y)

    device = xm.xla_device()
    model = model.to(device)
    input_x = input_x.to(device)
    input_y = input_y.to(device)

    output = model(input_x, input_y).to("cpu")

    # Not verifying data as many are wrong. Simply testing compile and execute
    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(enabled=False, required_atol=0.02),
        )
    )
    comparator.compare(output, golden)


@pytest.mark.single_device
@pytest.mark.parametrize("spmd_mode", [True, False])
def test_fully_replicated_graph(spmd_mode):
    if spmd_mode:
        xr.use_spmd()
    else:
        # There is no official way to unset SPMD mode in torch_xla.
        # So this is a workaround to delete the env var and reset the state.
        if os.environ.get("XLA_USE_SPMD") is not None:
            del os.environ["XLA_USE_SPMD"]

    class MM(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return x @ y

    input_x = torch.randn(32, 32, dtype=torch.bfloat16)
    input_y = torch.randn(32, 32, dtype=torch.bfloat16)
    model = MM()
    golden = model(input_x, input_y)

    device = xm.xla_device()
    model = model.to(device)
    input_x = input_x.to(device)
    input_y = input_y.to(device)
    output = model(input_x, input_y).to("cpu")
    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(enabled=False, required_atol=0.02),
        )
    )

    comparator.compare(output, golden)


@pytest.mark.nightly
@pytest.mark.push
@pytest.mark.dual_chip
@pytest.mark.parametrize("axis_names", [("x", "y")])
@pytest.mark.parametrize("input_shape", [(32, 32)])
@pytest.mark.parametrize("sharding_mode", ["fully_replicated", "partially_sharded"])
def test_spmd_sharding(axis_names, input_shape, sharding_mode):
    class LinearModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(32, 256, bias=False)

        def forward(self, x):
            return self.linear(x)

    def shard_spec_function(model):
        if sharding_mode == "partially_sharded":
            # Shard weight matrix along output dimension (dim 0)
            return {model.linear.weight: ("y", None)}
        else:
            # Do not shard anything, fully replicated
            return {}

    def setup_mesh(mesh_shape, axis_names):
        device_ids = np.arange(np.prod(mesh_shape))
        mesh = Mesh(device_ids=device_ids, mesh_shape=mesh_shape, axis_names=axis_names)
        return mesh

    inputs = torch.randn(input_shape)

    mesh_shape = (1, xr.global_runtime_device_count())
    mesh = setup_mesh(mesh_shape, axis_names)
    comparison_config = ComparisonConfig()

    run_graph_test(
        LinearModel(),
        [inputs],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
        mesh=mesh,
        shard_spec_fn=shard_spec_function,
    )
