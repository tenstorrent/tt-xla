# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Optional

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
        "--pcc-only",
        action="store_true",
        default=False,
        help=(
            "LLMs only: skip warmup and the timed perf loop, run a single PCC "
            "iteration, and assert both prefill and decode."
        ),
    )

    parser.addoption(
        "--pcc-prefill",
        action="store_true",
        default=False,
        help="Like --pcc-only but assert prefill PCC only (implies --pcc-only).",
    )

    parser.addoption(
        "--pcc-decode",
        action="store_true",
        default=False,
        help="Like --pcc-only but assert decode PCC only (implies --pcc-only).",
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


@pytest.fixture
def pcc_only(request):
    return request.config.getoption("--pcc-only")


@pytest.fixture
def pcc_prefill(request):
    return request.config.getoption("--pcc-prefill")


@pytest.fixture
def pcc_decode(request):
    return request.config.getoption("--pcc-decode")


@dataclass
class CliOptions:
    """Bundle of the command-line benchmark options shared by every test.

    Lets a test declare a single ``cli`` fixture instead of repeating the full
    list of option fixtures, and centralizes their resolution (see the
    ``_run_llm`` helpers in ``test_llms.py``).
    """

    output_file: Optional[str]
    num_layers: Optional[int]
    batch_size: Optional[int]
    accuracy_testing: bool
    optimization_level: Optional[int]
    max_output_tokens: Optional[int]
    decode_only: bool
    check_fusions: bool
    pcc_only: bool
    pcc_prefill: bool
    pcc_decode: bool


@pytest.fixture
def cli(
    output_file,
    num_layers,
    batch_size,
    accuracy_testing,
    optimization_level,
    max_output_tokens,
    decode_only,
    check_fusions,
    pcc_only,
    pcc_prefill,
    pcc_decode,
):
    """Bundle the common command-line options into a single object."""
    return CliOptions(
        output_file=output_file,
        num_layers=num_layers,
        batch_size=batch_size,
        accuracy_testing=accuracy_testing,
        optimization_level=optimization_level,
        max_output_tokens=max_output_tokens,
        decode_only=decode_only,
        check_fusions=check_fusions,
        pcc_only=pcc_only,
        pcc_prefill=pcc_prefill,
        pcc_decode=pcc_decode,
    )


@pytest.fixture(autouse=True)
def seed_torch():
    """Seed torch before every benchmark test for reproducible PCC runs."""
    torch.manual_seed(0)
