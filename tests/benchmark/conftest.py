# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch


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


def make_validator_optimization_level(option_name):
    """Create an optimization level validator (0, 1, or 2)."""

    def validate(value):
        try:
            int_value = int(value)
            if int_value not in [0, 1, 2]:
                raise ValueError
            return int_value
        except (ValueError, TypeError):
            raise pytest.UsageError(
                f"Invalid value for {option_name}: '{value}'. Must be 0, 1, or 2."
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
        "--optimization-level",
        action="store",
        default=None,
        type=make_validator_optimization_level("--optimization-level"),
        help="Optimization level (0, 1, or 2). Overrides default value.",
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
        "--check-fusions",
        action="store_true",
        default=False,
        help=(
            "Verify that expected fusion ops are present in the compiled IR. "
            "Tests must declare expected_ops to use this feature."
        ),
    )

    parser.addoption(
        "--save-cpu-snapshots-to",
        action="store",
        default=None,
        help=(
            "If set, run the CPU decode loop step-by-step and save a snapshot "
            "(input_args + indexer_k_caches + per-step golden logits) per "
            "step into this directory. Used downstream by "
            "deepseek_codegen/capture_step_n_inputs.py to dump valid "
            "tensors_step{k}/ files via direct ttnn writes. Requires --decode-only."
        ),
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
def optimization_level(request):
    return request.config.getoption("--optimization-level")


@pytest.fixture
def max_output_tokens(request):
    return request.config.getoption("--max-output-tokens")


@pytest.fixture
def decode_only(request):
    return request.config.getoption("--decode-only")


@pytest.fixture
def check_fusions(request):
    return request.config.getoption("--check-fusions")


@pytest.fixture(autouse=True)
def seed_torch():
    """Seed torch before every benchmark test for reproducible PCC runs."""
    torch.manual_seed(0)


@pytest.fixture
def save_cpu_snapshots_to(request):
    return request.config.getoption("--save-cpu-snapshots-to")
