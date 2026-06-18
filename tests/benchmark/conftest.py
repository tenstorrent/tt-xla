# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import json
import os
import pathlib
import re

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

    # --perf-report-dir / --perf-id are also defined in the top-level tests/conftest.py.
    # Benchmark perf jobs run with --confcutdir=tests/benchmark so that the top-level
    # conftest is NOT loaded (it would pull in device-connector init and CIv2 cache
    # cleanup that the benchmark suite intentionally avoids). Register them here too so
    # the options still exist under confcutdir. The try/except covers the local case of
    # collecting the whole tests/ tree, where both conftests load.
    for _name, _help in [
        (
            "--perf-report-dir",
            "Output directory for perf benchmark reports (one JSON per test).",
        ),
        ("--perf-id", "Perf ID used in per-test report filenames."),
    ]:
        try:
            parser.addoption(_name, action="store", default=None, help=_help)
        except ValueError:
            # Already registered by tests/conftest.py (whole-tree local run).
            pass


@pytest.fixture
def output_file(request):
    """Resolve where a benchmark test writes its JSON result.

    Precedence:
      1. ``--output-file``: explicit single-file path (local / single-test runs).
      2. ``--perf-report-dir`` (+ ``--perf-id``): one file per test named
         ``report_perf_<test>_<perf_id>.json``. This lets many benchmark tests be
         batched into a single CI job without overwriting each other's report.
         The trailing ``_<perf_id>`` is required by the downstream collect_data
         parser, which derives the GitHub job id from the filename's last token.

    ``--perf-report-dir`` / ``--perf-id`` are registered in the top-level
    ``tests/conftest.py`` (the same options the model tests use); ``default=None``
    keeps this safe if a benchmark test is ever run without that conftest loaded.
    """
    explicit = request.config.getoption("--output-file")
    if explicit:
        return explicit

    perf_dir = request.config.getoption("--perf-report-dir", default=None)
    if perf_dir:
        perf_id = request.config.getoption("--perf-id", default=None) or "local"
        os.makedirs(perf_dir, exist_ok=True)
        safe_name = re.sub(r"[^0-9A-Za-z._-]+", "_", request.node.name)
        return os.path.join(perf_dir, f"report_perf_{safe_name}_{perf_id}.json")

    return None


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


# Single source of truth for which hardware / run-type each benchmark test belongs
# to, migrated from the old perf-bench-matrix.json. Keyed by "<file>.py::<test name>"
# relative to this directory (params included for parametrized tests).
_PERF_MARKS_FILE = pathlib.Path(__file__).parent / "perf_marks.json"


def _load_perf_marks():
    try:
        with open(_PERF_MARKS_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def pytest_collection_modifyitems(items):
    """Apply perf-suite selection marks to benchmark tests.

    Marks — hardware (``n150`` / ``p150`` / ``n300_llmbox`` / ``galaxy_wh_6u`` /
    ``qb2_blackhole``), mode (``benchmark`` for perf, ``accuracy``), and ``vllm`` —
    are defined centrally in ``perf_marks.json`` and applied here so the perf CI can
    select tests with ``-m`` expressions (e.g. ``benchmark and n150 and not vllm``)
    exactly like the model tests, instead of listing one ``file::test`` per job.
    """
    mark_map = _load_perf_marks()
    if not mark_map:
        return

    bench_dir = pathlib.Path(__file__).parent
    matched = set()
    for item in items:
        fspath = str(getattr(item, "path", None) or item.fspath)
        try:
            rel = pathlib.Path(fspath).relative_to(bench_dir).as_posix()
        except ValueError:
            continue  # not a benchmark test; leave it untouched
        key = f"{rel}::{item.name}"
        marks = mark_map.get(key)
        if not marks:
            continue
        matched.add(key)
        for mark in marks:
            item.add_marker(getattr(pytest.mark, mark))

    # Diagnostic (opt-in): flag perf_marks.json keys that matched no collected test,
    # e.g. a renamed test or a stale vLLM param id. Useful during the CI migration.
    if os.environ.get("TT_XLA_DEBUG_PERF_MARKS"):
        unmatched = sorted(set(mark_map) - matched)
        if unmatched:
            print(
                "\n[perf_marks] WARNING: no collected test matched these "
                "perf_marks.json keys:\n  " + "\n  ".join(unmatched)
            )
