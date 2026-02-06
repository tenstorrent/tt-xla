# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


def pytest_addoption(parser):
    """Add custom command line options for pytest."""
    parser.addoption(
        "--folder",
        action="store",
        default="./collected_irs",
        help="Folder path containing IR files to process (default: ./collected_irs)",
    )
    parser.addoption(
        "--compile-only",
        action="store_true",
        default=False,
        help="Only compile ops without execution (default: False)",
    )
    parser.addoption(
        "--whitelist",
        action="store",
        default=None,
        help="Comma-separated list of operation names to test (e.g., 'stablehlo.add,stablehlo.multiply'). If set, only these operations will be tested.",
    )
    parser.addoption(
        "--blacklist",
        action="store",
        default=None,
        help="Comma-separated list of operation names to skip (e.g., 'stablehlo.custom_call'). Ignored if --whitelist is set.",
    )
    parser.addoption(
        "--debug-print",
        action="store_true",
        default=False,
        help="Enable debug printing during operation execution (default: False)",
    )
    parser.addoption(
        "--ir-file-prefix",
        action="store",
        default="",
        help="Filter files by subpath pattern and extract model names (e.g., 'irs/shlo_compiler' finds model_name/irs/shlo_compiler_*.mlir)",
    )
    parser.addoption(
        "--failed-ops-folder",
        action="store",
        default=None,
        help="Folder path to save MLIR modules of failed operations (default: None, no saving)",
    )
