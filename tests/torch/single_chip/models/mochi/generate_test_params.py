#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Generate test parameters for test_conv3d_mochi_shapes from locations.json
"""

import json


def parse_shape(shape_str):
    """Parse shape string like '1x768x9x62x108' to tuple"""
    return tuple(map(int, shape_str.split("x")))


def parse_padding(padding_list):
    """
    Convert padding from [pad_T_before, pad_T_after, pad_H_before, pad_H_after, pad_W_before, pad_W_after]
    to PyTorch format (T_pad, H_pad, W_pad) assuming symmetric padding
    """
    if len(padding_list) != 6:
        return (0, 0, 0)

    pad_t = int(padding_list[0])
    pad_h = int(padding_list[2])
    pad_w = int(padding_list[4])

    return (pad_t, pad_h, pad_w)


def generate_test_params(json_file):
    """Generate test parameters from locations.json"""
    with open(json_file, "r") as f:
        convolutions = json.load(f)

    test_params = []

    for i, conv in enumerate(convolutions):
        # Parse tensor shapes
        input_shape = parse_shape(conv["tensor_types"]["input"]["shape"])
        weight_shape = parse_shape(conv["tensor_types"]["weight"]["shape"])
        output_shape = parse_shape(conv["tensor_types"]["output"]["shape"])

        # Extract parameters
        B, C_in, T_in, H_in, W_in = input_shape
        C_out, _, K_T, K_H, K_W = weight_shape

        # Get attributes
        attrs = conv["attributes"]
        kernel_size = (K_T, K_H, K_W)
        stride = tuple(map(int, attrs["window_strides"]))
        padding = parse_padding(attrs["padding"])

        # Determine padding mode
        # If all padding is 0, use zeros, otherwise replicate (based on the test patterns)
        padding_mode = "zeros" if all(p == 0 for p in padding) else "replicate"

        # Get location for ID
        location = conv["location"]
        num_occurrences = len(conv["occurrences"])

        param = {
            "input_shape": input_shape,
            "out_channels": C_out,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "padding_mode": padding_mode,
            "location": location,
            "occurrences": num_occurrences,
        }

        test_params.append(param)

    return test_params


def print_test_params(params):
    """Print test parameters in pytest parametrize format"""
    print("# Generated test parameters from Mochi decoder convolutions")
    print("# Total unique convolutions:", len(params))
    print()

    for i, param in enumerate(params):
        print(f"# Conv #{i+1}: {param['location'][:80]}...")
        print(f"# Occurrences: {param['occurrences']}")
        print(f"[")
        print(f"    {param['input_shape']},  # input_shape")
        print(f"    {param['out_channels']},  # out_channels")
        print(f"    {param['kernel_size']},  # kernel_size")
        print(f"    {param['stride']},  # stride")
        print(f"    {param['padding']},  # padding")
        print(f"    \"{param['padding_mode']}\",  # padding_mode")
        print(f"    None,  # blocking (to be determined)")
        print(f"],")
        print()


def print_pytest_format(params):
    """Print in full pytest parametrize format"""
    print("@pytest.mark.parametrize(")
    print(
        '    "input_shape, out_channels, kernel_size, stride, padding, padding_mode, blocking",'
    )
    print("    [")

    for i, param in enumerate(params):
        # Create a short ID
        loc_parts = param["location"].split("/")
        id_name = f"conv{i+1}"
        if "conv_in" in param["location"]:
            id_name = "conv_in"
        elif "conv1" in param["location"]:
            id_name = "conv1"
        elif "conv2" in param["location"]:
            id_name = "conv2"

        print(f"        [")
        print(f"            {param['input_shape']},")
        print(f"            {param['out_channels']},")
        print(f"            {param['kernel_size']},")
        print(f"            {param['stride']},")
        print(f"            {param['padding']},")
        print(f"            \"{param['padding_mode']}\",")
        print(f"            None,  # blocking (TODO: determine optimal blocking)")
        print(f"        ],  # {id_name} - {param['occurrences']} occurrences")

    print("    ],")
    ids_list = [f'"conv{i+1}"' for i in range(len(params))]
    print(f'    ids=[{", ".join(ids_list)}],')
    print(")")


if __name__ == "__main__":
    params = generate_test_params("locations.json")

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total unique convolutions: {len(params)}")
    print()

    for i, param in enumerate(params):
        print(
            f"{i+1}. Input: {param['input_shape']}, Out: {param['out_channels']}, "
            f"Kernel: {param['kernel_size']}, Stride: {param['stride']}, "
            f"Padding: {param['padding']}, Occurrences: {param['occurrences']}"
        )

    print()
    print("=" * 80)
    print("PYTEST PARAMETRIZE FORMAT")
    print("=" * 80)
    print()
    print_pytest_format(params)
