#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Generate a MLIR_FILES list for test_load_split_and_execute.

For each test (one at a time) the script:
  1. Empties the artifacts directory to avoid stale files.
  2. Runs the test with --serialize.
  3. Scans the artifacts directory for generated MLIR files.
  4. Writes tests/builder/generated_mlir_files.py with those files.
  5. Immediately runs test_load_split_and_execute.
  6. Moves on to the next test.

Sources
-------
  (no --source)  Pass full pytest node IDs directly. Artifacts default to output_artifact/.
  --source llm   Shorthand for tests/benchmark/test_llms.py. Pass bare function names.
                 Artifacts default to modules/irs/.
  --source models  Shorthand for tests/runner/test_models.py::test_all_models_torch[CONFIG].
                 Pass the parametrize config ID. Artifacts default to output_artifact/.

Examples:
    # Any test — full node ID (op tests, graph tests, etc.)
    python tests/integrations/tt_mlir_builder/run_test_op_by_op.py \
        "tests/torch/ops/test_add.py::test_add"

    python tests/integrations/tt_mlir_builder/run_test_op_by_op.py \
        "tests/runner/test_models.py::test_all_models_torch[pytorch_resnet50-inference-single_device]" \
        "tests/runner/test_models.py::test_all_models_torch[pytorch_bert_base_uncased-inference-single_device]"

    # LLM benchmark tests (shorthand, artifacts in modules/irs/)
    python tests/integrations/tt_mlir_builder/run_test_op_by_op.py test_phi1_5 test_gemma_2_2b --source llm

    # Model runner tests (shorthand, artifacts in output_artifact/)
    python tests/integrations/tt_mlir_builder/run_test_op_by_op.py \
        pytorch_resnet50-inference-single_device \\
        pytorch_bert_base_uncased-inference-single_device \\
        --source models

    # Run only TTIR tests from an LLM
    python tests/integrations/tt_mlir_builder/run_test_op_by_op.py test_llama_3_2_1b --source llm --target ttir

    # Run multiple targets (TTIR and TTNN, skip StableHLO)
    python tests/integrations/tt_mlir_builder/run_test_op_by_op.py test_phi1_5 --source llm --target ttir --target ttnn
"""

import argparse
import os
import sys
from pathlib import Path

from test_utils import (
    build_name_filter,
    collect_mlir_files,
    filter_mlir_files_by_target,
    resolve_test_ids_and_artifacts,
    run_pytest,
    snapshot_mlir_files,
    write_and_display_mlir_files,
)


BUILDER_TEST = "tests/integrations/tt_mlir_builder/load_split_and_execute.py::test_load_split_and_execute"
GENERATED_MLIR_LIST = Path("tests/integrations/tt_mlir_builder/generated_mlir_files.py")
DEFAULT_ARTIFACTS_DIR = Path("output_artifact")


def run_tests(
    test_ids: list[str],
    artifacts_dir: Path,
    source: str | None,
    target_filters: list[str] | None,
    no_run: bool,
) -> int:
    """Run tests, collect MLIR artifacts, and execute builder tests.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    failed_tests: list[str] = []
    builder_test_failures: list[str] = []

    # Handle --no-run: collect existing files without running tests
    if no_run:
        mlir_files = collect_mlir_files(artifacts_dir)
        mlir_files = filter_mlir_files_by_target(mlir_files, target_filters)
        if not mlir_files:
            target_note = (
                f" (filtered to {', '.join(target_filters)})" if target_filters else ""
            )
            print(
                f"No MLIR files found in {artifacts_dir}/{target_note}", file=sys.stderr
            )
            return 1
        write_and_display_mlir_files(mlir_files, GENERATED_MLIR_LIST)
        return run_pytest(BUILDER_TEST)

    # Run each test and collect its MLIR artifacts
    for i, test_id in enumerate(test_ids, 1):
        print(f"\n{'='*60}\n[{i}/{len(test_ids)}] {test_id}\n{'='*60}\n")

        # Prepare artifacts directory and track pre-existing files
        pre_existing = snapshot_mlir_files(artifacts_dir)
        if pre_existing:
            note = (
                "will ignore them (LLM tests timestamp files)"
                if source == "llm"
                else "will include matching ones (other tests overwrite files)"
            )
            print(f"Found {len(pre_existing)} pre-existing files — {note}")

        # Run test with serialization
        name_filter = build_name_filter(source, test_id)
        exit_code = run_pytest(test_id, ["--serialize"])
        if exit_code != 0:
            print(f"Warning: test failed with exit code {exit_code}", file=sys.stderr)
            failed_tests.append(test_id)

        # Collect MLIR files (exclude pre-existing for LLM tests only)
        exclude = pre_existing if source == "llm" else None
        mlir_files = collect_mlir_files(
            artifacts_dir, exclude=exclude, name_filter=name_filter
        )
        mlir_files = filter_mlir_files_by_target(mlir_files, target_filters)

        if not mlir_files:
            target_note = (
                f" (filtered to {', '.join(target_filters)})" if target_filters else ""
            )
            print(
                f"Warning: no MLIR files found{target_note}, skipping builder test",
                file=sys.stderr,
            )
            continue

        # Write files list and run builder test
        target_summary = (
            f" ({', '.join(target_filters)} only)" if target_filters else ""
        )
        print(f"Collected {len(mlir_files)} MLIR files{target_summary}")
        write_and_display_mlir_files(mlir_files, GENERATED_MLIR_LIST)

        if run_pytest(BUILDER_TEST) != 0:
            builder_test_failures.append(test_id)

    if failed_tests:
        print(
            f"\nWarning: tests that did not pass: {', '.join(failed_tests)}",
            file=sys.stderr,
        )
    if builder_test_failures:
        print(
            f"\nWarning: builder integration test failed for: {', '.join(builder_test_failures)}",
            file=sys.stderr,
        )

    return 1 if (failed_tests or builder_test_failures) else 0


def main() -> int:
    """Parse arguments and run tests."""
    parser = argparse.ArgumentParser(
        description="Run tests with --serialize and feed the resulting MLIR files into test_load_split_and_execute.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "test_names",
        nargs="+",
        metavar="name",
        help=(
            "Without --source: full pytest node IDs. "
            "With --source llm: bare function names (e.g. test_phi1_5). "
            "With --source models: parametrize config IDs "
            "(e.g. pytorch_resnet50-inference-single_device)."
        ),
    )
    parser.add_argument(
        "--source",
        choices=["llm", "models"],
        default=None,
        help=(
            "Convenience shorthand: 'llm' targets tests/benchmark/test_llms.py "
            "(artifacts: modules/irs/), 'models' targets "
            "tests/runner/test_models.py::test_all_models_torch[...] "
            "(artifacts: output_artifact/). "
            "Omit to pass full pytest node IDs directly."
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory to scan for generated MLIR files. "
            "Overrides the default for the selected --source. "
            f"Default without --source: {DEFAULT_ARTIFACTS_DIR}"
        ),
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        default=False,
        help="Skip running tests; only collect files already in --artifacts-dir",
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=["stablehlo", "ttir", "ttnn"],
        help="Only test specific MLIR target(s). Can be specified multiple times. Default: all targets",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    os.chdir(repo_root)

    test_ids, artifacts_dir = resolve_test_ids_and_artifacts(
        args.source, args.test_names, args.artifacts_dir, DEFAULT_ARTIFACTS_DIR
    )

    return run_tests(
        test_ids=test_ids,
        artifacts_dir=artifacts_dir,
        source=args.source,
        target_filters=args.target,
        no_run=args.no_run,
    )


if __name__ == "__main__":
    sys.exit(main())