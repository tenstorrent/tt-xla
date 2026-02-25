#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Generate a MLIR_FILES list for test_serialize_and_builder_integration.

For each test (one at a time) the script:
  1. Empties the artifacts directory to avoid stale files.
  2. Runs the test with --serialize.
  3. Scans the artifacts directory for generated MLIR files.
  4. Writes tests/torch/mlir_files_generated.py with those files.
  5. Immediately runs test_serialize_and_builder_integration.
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
    python scripts/gen_mlir_files_list.py \\
        "tests/jax/single_chip/ops/test_add.py::test_add"

    python scripts/gen_mlir_files_list.py \\
        "tests/runner/test_models.py::test_all_models_torch[pytorch_resnet50-inference-single_device]" \\
        "tests/runner/test_models.py::test_all_models_torch[pytorch_bert_base_uncased-inference-single_device]"

    # LLM benchmark tests (shorthand, artifacts in modules/irs/)
    python scripts/gen_mlir_files_list.py test_phi1_5 test_gemma_2_2b --source llm

    # Model runner tests (shorthand, artifacts in output_artifact/)
    python scripts/gen_mlir_files_list.py \\
        pytorch_resnet50-inference-single_device \\
        pytorch_bert_base_uncased-inference-single_device \\
        --source models
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUILDER_TEST = (
    "tests/torch/test_torch_filecheck.py::test_serialize_and_builder_integration"
)
GENERATED_MLIR_LIST = Path("tests/torch/mlir_files_generated.py")
DEFAULT_ARTIFACTS_DIR = Path("output_artifact")

# Source shorthands
LLM_TEST_FILE = "tests/benchmark/test_llms.py"
LLM_ARTIFACTS_DIR = Path("modules/irs")

MODELS_TEST_FILE = "tests/runner/test_models.py"
MODELS_TEST_FN = "test_all_models_torch"
MODELS_ARTIFACTS_DIR = Path("output_artifact")

# Prefix-based classification (modules/irs/ style)
# e.g. shlo_compiler_phi1_bs32_...mlir, ttir_phi1_...mlir
_PREFIX_TO_TARGET = {
    "shlo_compiler_": "stablehlo",
    "ttir_": "ttir",
    "ttnn_": "ttnn",
}
_SKIP_PREFIXES = ("shlo_compiler_cleaned_",)

# Suffix-based classification (output_artifact/ style)
# e.g. test_all_models_torch_resnet50_..._ttir.mlir
_SUFFIX_TO_TARGET = {
    "_ttir": "ttir",
    "_ttnn": "ttnn",
}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_mlir_file(filename: str) -> str | None:
    """Return the MLIR target for a filename, or None to skip it.

    Handles both naming conventions:
      - prefix-based  (modules/irs/ output):    shlo_compiler_* / ttir_* / ttnn_*
      - suffix-based  (output_artifact/ output): *_ttir.mlir / *_ttnn.mlir
    """
    if not filename.endswith(".mlir"):
        return None

    # Explicit skips first (more specific than the prefix entries below)
    for skip in _SKIP_PREFIXES:
        if filename.startswith(skip):
            return None

    # Prefix-based
    for prefix, target in _PREFIX_TO_TARGET.items():
        if filename.startswith(prefix):
            return target

    # Suffix-based
    stem = filename[: -len(".mlir")]
    for suffix, target in _SUFFIX_TO_TARGET.items():
        if stem.endswith(suffix):
            return target

    return None


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------


def empty_dir(directory: Path) -> None:
    """Remove all contents of directory without deleting it."""
    if directory.exists():
        for item in directory.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        print(f"Emptied {directory}/")
    else:
        directory.mkdir(parents=True)
        print(f"Created {directory}/")


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def _sanitize(name: str) -> str:
    """Mirror the sanitize_test_name logic used by the test infra to name output files."""
    return re.sub(r"[\[\](),\-\s/:]+", "_", name).rstrip("_")


def _llm_model_name(test_fn_name: str) -> str:
    """Derive the model name embedded in LLM IR filenames from a test function name.

    test_ministral_8b -> ministral_8b  (strip leading 'test_')
    """
    return test_fn_name.removeprefix("test_")


def _matches_llm_model(filename: str, model_name: str) -> bool:
    """Return True if filename belongs to the given LLM model.

    LLM files look like: {type_prefix}{model_name}_{batch_size}_...mlir
    e.g. ttnn_ministral_8b_5lyr_bs32_isl128_runc72f_g1_1772039706530.mlir
    """
    for prefix in _PREFIX_TO_TARGET:
        if filename.startswith(prefix + model_name + "_"):
            return True
    return False


def build_name_filter(source: str | None, node_id: str):
    """Return a callable(filename) -> bool that accepts only files for this test.

    output_artifact/ files are prefixed with the sanitized node name.
    modules/irs/ LLM files embed the model name after the type prefix.
    """
    if source == "llm":
        # node_id is the full path e.g. tests/benchmark/test_llms.py::test_ministral_8b
        test_fn = node_id.split("::")[-1]
        model_name = _llm_model_name(test_fn)
        print(f"Will collect files matching LLM model name: {model_name!r}")
        return lambda f, m=model_name: _matches_llm_model(f, m)
    else:
        node_name = node_id.split("::")[-1] if "::" in node_id else node_id
        prefix = _sanitize(node_name)
        print(f"Will collect files matching prefix: {prefix!r}")
        return lambda f, p=prefix: f.startswith(p)


def snapshot_mlir_files(artifacts_dir: Path) -> set[Path]:
    """Return the set of .mlir files currently in artifacts_dir."""
    if not artifacts_dir.exists():
        return set()
    return {f for f in artifacts_dir.glob("*.mlir")}


def collect_mlir_files(
    artifacts_dir: Path,
    exclude: set[Path] | None = None,
    name_filter=None,
) -> list[tuple[str, str]]:
    """Scan artifacts_dir and return a sorted list of (target, path) tuples.

    exclude     – set of pre-existing files to skip (snapshot taken before the
                  test ran). Pass this for LLM tests, where each run generates
                  new uniquely-timestamped files alongside any old ones.
                  Do NOT pass this for output_artifact/ tests, where the tester
                  overwrites files in-place (the pre-existing file IS the fresh result).

    name_filter – callable(filename: str) -> bool; when provided, only matching
                  files are collected.
    """
    entries = []
    for mlir_file in sorted(artifacts_dir.glob("*.mlir")):
        if exclude and mlir_file in exclude:
            continue
        if name_filter and not name_filter(mlir_file.name):
            continue
        target = classify_mlir_file(mlir_file.name)
        if target is not None:
            entries.append((target, str(mlir_file)))
    return entries


# ---------------------------------------------------------------------------
# pytest invocations
# ---------------------------------------------------------------------------


def run_test(node_id: str) -> int:
    cmd = ["pytest", "-sv", node_id, "--serialize"]
    print(f"Running: {' '.join(cmd)}\n")
    return subprocess.run(cmd).returncode


def run_builder_integration_test() -> int:
    cmd = ["pytest", "-sv", BUILDER_TEST]
    print(f"\nRunning: {' '.join(cmd)}\n")
    return subprocess.run(cmd).returncode


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_mlir_files_list(entries: list[tuple[str, str]], output_path: Path) -> None:
    lines = [
        "# AUTO-GENERATED by scripts/gen_mlir_files_list.py — do not edit by hand.\n",
        "# Re-run the script to regenerate this file.\n",
        "\n",
        "MLIR_FILES = [\n",
    ]
    for target, path in entries:
        lines.append(f'    ("{target}", "{path}"),\n')
    lines.append("]\n")
    output_path.write_text("".join(lines))
    print(f"Wrote {len(entries)} entries to {output_path}")


def _print_mlir_files(entries: list[tuple[str, str]]) -> None:
    print("\n# MLIR_FILES:\n")
    print("MLIR_FILES = [")
    for target, path in entries:
        print(f'    ("{target}", "{path}"),')
    print("]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def resolve_test_ids_and_artifacts(
    source: str | None, names: list[str], artifacts_dir_override: str | None
) -> tuple[list[str], Path]:
    """Return (pytest_node_ids, artifacts_dir) based on --source and --artifacts-dir."""
    if source == "llm":
        node_ids = [f"{LLM_TEST_FILE}::{name}" for name in names]
        artifacts_dir = (
            Path(artifacts_dir_override)
            if artifacts_dir_override
            else LLM_ARTIFACTS_DIR
        )
    elif source == "models":
        node_ids = [f"{MODELS_TEST_FILE}::{MODELS_TEST_FN}[{name}]" for name in names]
        artifacts_dir = (
            Path(artifacts_dir_override)
            if artifacts_dir_override
            else MODELS_ARTIFACTS_DIR
        )
    else:
        # Raw mode: names are already full pytest node IDs
        node_ids = names
        artifacts_dir = (
            Path(artifacts_dir_override)
            if artifacts_dir_override
            else DEFAULT_ARTIFACTS_DIR
        )
    return node_ids, artifacts_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run tests with --serialize and feed the resulting MLIR files into test_serialize_and_builder_integration.",
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
        "--no-empty",
        action="store_true",
        default=False,
        help="Skip emptying the artifacts directory before each test (useful with --no-run)",
    )
    parser.add_argument(
        "--no-builder-test",
        action="store_true",
        default=False,
        help="Skip running test_serialize_and_builder_integration after each test",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    os.chdir(repo_root)

    test_ids, artifacts_dir = resolve_test_ids_and_artifacts(
        args.source, args.test_names, args.artifacts_dir
    )

    failed_tests: list[str] = []
    builder_test_failures: list[str] = []

    if args.no_run:
        entries = collect_mlir_files(artifacts_dir)
        if not entries:
            print(f"No matching MLIR files found in {artifacts_dir}/.", file=sys.stderr)
            return 1
        write_mlir_files_list(entries, GENERATED_MLIR_LIST)
        _print_mlir_files(entries)
        if not args.no_builder_test:
            return run_builder_integration_test()
        return 0

    for i, test_id in enumerate(test_ids):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(test_ids)}] {test_id}")
        print(f"{'='*60}\n")

        if not args.no_empty:
            empty_dir(artifacts_dir)
            pre_existing: set[Path] = set()
        else:
            pre_existing = snapshot_mlir_files(artifacts_dir)
            if pre_existing:
                if args.source == "llm":
                    print(
                        f"Noted {len(pre_existing)} pre-existing MLIR file(s) — "
                        "will ignore them (LLM tests generate new timestamped files each run)."
                    )
                else:
                    print(
                        f"Noted {len(pre_existing)} pre-existing MLIR file(s) — "
                        "will include matching ones (output_artifact files are overwritten in-place)."
                    )

        name_filter = build_name_filter(args.source, test_id)

        exit_code = run_test(test_id)
        if exit_code != 0:
            print(
                f"\nWarning: test exited with code {exit_code}. "
                "Collecting any MLIR files that were generated anyway.",
                file=sys.stderr,
            )
            failed_tests.append(test_id)

        # For LLM tests, pass the pre-existing snapshot so that old
        # uniquely-timestamped files from the same model are excluded.
        # For output_artifact/ tests, files are overwritten in-place so the
        # snapshot must not be used (the "pre-existing" file IS the fresh result).
        snapshot_to_exclude = pre_existing if args.source == "llm" else None
        entries = collect_mlir_files(
            artifacts_dir, exclude=snapshot_to_exclude, name_filter=name_filter
        )
        if not entries:
            print(
                f"Warning: no MLIR files found for {test_id}, skipping builder test.",
                file=sys.stderr,
            )
            continue

        print(f"Collected {len(entries)} MLIR files.")
        write_mlir_files_list(entries, GENERATED_MLIR_LIST)
        _print_mlir_files(entries)

        if not args.no_builder_test:
            builder_exit_code = run_builder_integration_test()
            if builder_exit_code != 0:
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


if __name__ == "__main__":
    sys.exit(main())
