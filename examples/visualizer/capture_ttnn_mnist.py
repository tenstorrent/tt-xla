# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Capture a graph report for the mnist network written directly in ttnn.

Same architecture as the model in examples/pytorch/mnist.py — conv(1->32,3x3)
relu, conv(32->64,3x3) relu, max_pool 2x2, flatten, linear(9216->128) relu,
linear(128->10), log_softmax — so the report it writes is a reference point for
what the same network looks like captured through tt-xla. Dropout is an
inference-time identity and is left out, as it is in the tt-xla graph.

Requires ttnn on the path, which the tt-xla venv does not provide; see the
README section on capturing pure ttnn for the two variables to set.
"""

import argparse
import pathlib

import torch
import ttnn

BATCH = 4


def build_weights():
    torch.manual_seed(42)
    return {
        name: torch.randn(shape, dtype=torch.bfloat16)
        for name, shape in (
            ("conv1.weight", (32, 1, 3, 3)),
            ("conv1.bias", (32,)),
            ("conv2.weight", (64, 32, 3, 3)),
            ("conv2.bias", (64,)),
            ("fc1.weight", (128, 9216)),
            ("fc1.bias", (128,)),
            ("fc2.weight", (10, 128)),
            ("fc2.bias", (10,)),
        )
    }


def conv_params(weights, name):
    # conv2d takes its weights host-side in row-major, bias as 1x1x1xC.
    weight = ttnn.from_torch(
        weights[f"{name}.weight"], layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
    )
    bias = ttnn.from_torch(
        weights[f"{name}.bias"].reshape((1, 1, 1, -1)),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
    )
    return weight, bias


def linear_params(weights, name, device):
    weight = ttnn.from_torch(
        weights[f"{name}.weight"].transpose(0, 1).contiguous(),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    bias = ttnn.from_torch(
        weights[f"{name}.bias"].reshape((1, -1)),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )
    return weight, bias


def forward(device, weights, x_torch):
    relu = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)

    # conv2d takes NHWC.
    x = ttnn.from_torch(
        x_torch.permute(0, 2, 3, 1).contiguous(),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
    )

    weight, bias = conv_params(weights, "conv1")
    x = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=weight,
        bias_tensor=bias,
        in_channels=1,
        out_channels=32,
        device=device,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(0, 0),
        batch_size=BATCH,
        input_height=28,
        input_width=28,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16, activation=relu),
        groups=1,
    )

    weight, bias = conv_params(weights, "conv2")
    x = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=weight,
        bias_tensor=bias,
        in_channels=32,
        out_channels=64,
        device=device,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(0, 0),
        batch_size=BATCH,
        input_height=26,
        input_width=26,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.bfloat16, activation=relu),
        groups=1,
    )

    x = ttnn.max_pool2d(
        x,
        batch_size=BATCH,
        input_h=24,
        input_w=24,
        channels=64,
        kernel_size=[2, 2],
        stride=[2, 2],
        padding=[0, 0],
        dilation=[1, 1],
        ceil_mode=False,
    )

    # Pooling returns a sharded L1 tensor; reshape needs it interleaved.
    x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.reshape(x, (BATCH, 9216))
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    # The flattened activations stay NHWC-ordered, so fc1's columns are reordered
    # to match rather than permuting the activations on device.
    nhwc_weights = dict(weights)
    nhwc_weights["fc1.weight"] = (
        weights["fc1.weight"]
        .reshape(128, 64, 12, 12)
        .permute(0, 2, 3, 1)
        .reshape(128, 9216)
    )

    weight, bias = linear_params(nhwc_weights, "fc1", device)
    x = ttnn.relu(ttnn.linear(x, weight, bias=bias))

    weight, bias = linear_params(weights, "fc2", device)
    x = ttnn.linear(x, weight, bias=bias)

    # ttnn has no log_softmax.
    return ttnn.log(ttnn.softmax(x, dim=-1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--report",
        default="graph_reports_ttnn/report.json",
        help="destination JSON file for the report (default: %(default)s)",
    )
    args = parser.parse_args()

    report = pathlib.Path(args.report).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)

    device = ttnn.open_device(device_id=0, l1_small_size=8192)
    try:
        with ttnn.graph.full_graph_capture(str(report)):
            x = torch.ones((BATCH, 1, 28, 28), dtype=torch.bfloat16)
            output = forward(device, build_weights(), x)
        print(f"output shape: {tuple(output.shape)}")
    finally:
        ttnn.close_device(device)

    sidecar = report.with_suffix(".python_io.json")
    print(f"report:  {report} ({report.stat().st_size} bytes)")
    if sidecar.exists():
        print(f"sidecar: {sidecar} ({sidecar.stat().st_size} bytes)")
    else:
        print(f"sidecar: {sidecar} MISSING")


if __name__ == "__main__":
    main()
