# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Register the custom pytest CLI options the tt-xla CI harness injects.

mixed_precision/ lives outside tests/, so tests/conftest.py — which normally
registers these options — is not loaded when the CI runs `pytest ./mixed_precision`.
The shared harness (.github/workflows/call-test.yml) always appends --log-memory
and, in the run step, --perf-report-dir / --perf-id. Without these registered,
pytest aborts arg parsing with "unrecognized arguments" before collecting anything.

We register them as accepted options (the CPU regression test doesn't use them),
plus a couple more the harness can inject, so collection and run don't error.
"""


def pytest_addoption(parser):
    parser.addoption("--log-memory", action="store_true", default=False)
    parser.addoption("--perf-report-dir", action="store", default=None)
    parser.addoption("--perf-id", action="store", default=None)
    parser.addoption("--dump-irs", action="store_true", default=False)
    parser.addoption("--arch", action="store", default=None)
