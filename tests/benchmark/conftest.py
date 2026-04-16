# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest


def make_validator_positive_int(option_name):
    """Create a positive integer validator with the option name in error messages."""

    def validate(value):
        try:
            int_value = int(value)
            if int_value <= 0:
                raise ValueError
            return int_value
        except (ValueError, TypeError):
            raise pytest.UsageError(
                f"Invalid value for {option_name}: '{value}'. Must be a positive integer (> 0)."
            )

    return validate


def pytest_addoption(parser):
    """Adds a custom command-line option to pytest."""
    parser.addoption(
        "--output-file",
        action="store",
        default=None,
        help="Path to save benchmark results as JSON.",
    )

    parser.addoption(
        "--num-layers",
        action="store",
        default=None,
        type=make_validator_positive_int("--num-layers"),
        help="Number of model layers (positive integer). Overrides model config when supported.",
    )

    parser.addoption(
        "--batch-size",
        action="store",
        default=None,
        type=make_validator_positive_int("--batch-size"),
        help="Batch size (positive integer). Overrides config value.",
    )

    parser.addoption(
        "--accuracy-testing",
        action="store_true",
        default=False,
        help="Enable accuracy testing mode. Uses reference data for TOP1/TOP5 accuracy.",
    )

    parser.addoption(
        "--max-output-tokens",
        action="store",
        default=None,
        type=make_validator_positive_int("--max-output-tokens"),
        help="Limit the maximum number of output tokens generated. Useful for profiling runs.",
    )

    parser.addoption(
        "--decode-only",
        action="store_true",
        default=False,
        help="Run prefill on CPU and only decode on device. Measures decode-only throughput.",
    )

    parser.addoption(
        "--layer-index",
        action="store",
        default=None,
        type=int,
        help="Specific layer index to isolate (0-based). Loads a 1-layer model with weights from this layer.",
    )

    parser.addoption(
        "--deinterleave",
        action="store_true",
        default=False,
        help="De-interleave gate_up_proj into separate gate_proj and up_proj for better quantization.",
    )


@pytest.fixture
def output_file(request):
    return request.config.getoption("--output-file")


@pytest.fixture
def num_layers(request):
    return request.config.getoption("--num-layers")


@pytest.fixture
def batch_size(request):
    return request.config.getoption("--batch-size")


@pytest.fixture
def accuracy_testing(request):
    return request.config.getoption("--accuracy-testing")


@pytest.fixture
def max_output_tokens(request):
    return request.config.getoption("--max-output-tokens")


@pytest.fixture
def decode_only(request):
    return request.config.getoption("--decode-only")


@pytest.fixture
def layer_index(request):
    return request.config.getoption("--layer-index")


@pytest.fixture
def deinterleave(request):
    return request.config.getoption("--deinterleave")
