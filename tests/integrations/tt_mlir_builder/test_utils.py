#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Utilities for MLIR file collection and testing."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Source shorthands
LLM_TEST_FILE = "tests/benchmark/test_llms.py"
LLM_ARTIFACTS_DIR = Path("modules/irs")

MODELS_TEST_FILE = "tests/runner/test_models.py"
MODELS_TEST_FN = "test_all_models_torch"
MODELS_ARTIFACTS_DIR = Path("output_artifact")

# MLIR file naming patterns for classification
PREFIX_TO_TARGET = {
    "shlo_compiler_": "stablehlo",
    "ttir_": "ttir",
    "ttnn_": "ttnn",
}
SKIP_PREFIXES = ("shlo_compiler_cleaned_",)
SUFFIX_TO_TARGET = {
    "_ttir": "ttir",
    "_ttnn": "ttnn",
}


def classify_mlir_file(filename: str) -> str | None:
    """Return the MLIR target for a filename, or None to skip it.

    Handles both naming conventions:
      - prefix-based  (modules/irs/ output):    shlo_compiler_* / ttir_* / ttnn_*
      - suffix-based  (output_artifact/ output): *_ttir.mlir / *_ttnn.mlir
    """
    if not filename.endswith(".mlir"):
        return None

    # Skip explicitly ignored patterns
    if any(filename.startswith(skip) for skip in SKIP_PREFIXES):
        return None

    # Check prefix-based naming (modules/irs/ style)
    for prefix, target in PREFIX_TO_TARGET.items():
        if filename.startswith(prefix):
            return target

    # Check suffix-based naming (output_artifact/ style)
    stem = filename[:-5]  # Remove ".mlir"
    for suffix, target in SUFFIX_TO_TARGET.items():
        if stem.endswith(suffix):
            return target

    return None


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


def sanitize_test_name(name: str) -> str:
    """Convert test name to filesystem-safe format."""
    return re.sub(r"[\[\](),\-\s/:]+", "_", name).rstrip("_")


def extract_llm_model_name(test_fn_name: str) -> str:
    """Extract model name from test function name (test_ministral_8b -> ministral_8b)."""
    return test_fn_name.removeprefix("test_")


def matches_llm_model(filename: str, model_name: str) -> bool:
    """Check if filename belongs to the given LLM model."""
    return any(
        filename.startswith(f"{prefix}{model_name}_") for prefix in PREFIX_TO_TARGET
    )


def build_name_filter(source: str | None, node_id: str):
    """Return a filter function that matches files for this test."""
    if source == "llm":
        test_fn = node_id.split("::")[-1]
        model_name = extract_llm_model_name(test_fn)
        print(f"Collecting files for LLM model: {model_name!r}")
        return lambda filename: matches_llm_model(filename, model_name)

    # For non-LLM tests, match by sanitized test name prefix
    test_name = node_id.split("::")[-1] if "::" in node_id else node_id
    prefix = sanitize_test_name(test_name)
    print(f"Collecting files with prefix: {prefix!r}")
    return lambda filename: filename.startswith(prefix)


def snapshot_mlir_files(artifacts_dir: Path) -> set[Path]:
    """Return current .mlir files in artifacts directory."""
    return set(artifacts_dir.glob("*.mlir")) if artifacts_dir.exists() else set()


def collect_mlir_files(
    artifacts_dir: Path,
    exclude: set[Path] | None = None,
    name_filter=None,
) -> list[tuple[str, str]]:
    """Collect MLIR files and return list of (target, path) tuples."""
    mlir_files = []
    for path in sorted(artifacts_dir.glob("*.mlir")):
        if exclude and path in exclude:
            continue
        if name_filter and not name_filter(path.name):
            continue
        target = classify_mlir_file(path.name)
        if target:
            mlir_files.append((target, str(path)))
    return mlir_files


def run_pytest(test_path: str, extra_args: list[str] = None) -> int:
    """Run pytest with given test path and optional extra arguments."""
    cmd = [sys.executable, "-m", "pytest", "-sv", test_path]
    if extra_args:
        cmd.extend(extra_args)
    print(f"Running: {' '.join(cmd)}\n")
    return subprocess.run(cmd, env=os.environ.copy()).returncode

def filter_mlir_files_by_target(
    mlir_files: list[tuple[str, str]], targets: list[str] | None
) -> list[tuple[str, str]]:
    """Filter MLIR files to only include specified targets."""
    if not targets:
        return mlir_files
    target_set = set(targets)
    return [(target, path) for target, path in mlir_files if target in target_set]


def write_and_display_mlir_files(
    mlir_files: list[tuple[str, str]], output_path: Path
) -> None:
    """Write MLIR files list to Python module and display it."""
    lines = [
        "# AUTO-GENERATED by tests/integrations/tt_mlir_builder/run_test_op_by_op.py — do not edit by hand.\n",
        "# Re-run the script to regenerate this file.\n\n",
        "MLIR_FILES = [\n",
    ]
    for target, path in mlir_files:
        lines.append(f'    ("{target}", "{path}"),\n')
    lines.append("]\n")

    output_path.write_text("".join(lines))
    print(f"\nWrote {len(mlir_files)} files to {output_path}")
    print("\nMLIR_FILES = [")
    for target, path in mlir_files:
        print(f'    ("{target}", "{path}"),')
    print("]")


def resolve_test_ids_and_artifacts(
    source: str | None,
    names: list[str],
    artifacts_dir_override: str | None,
    default_artifacts_dir: Path,
) -> tuple[list[str], Path]:
    """Convert test names to full pytest node IDs and determine artifacts directory."""
    # Map source types to their test file patterns and default artifact dirs
    source_config = {
        "llm": ([f"{LLM_TEST_FILE}::{name}" for name in names], LLM_ARTIFACTS_DIR),
        "models": (
            [f"{MODELS_TEST_FILE}::{MODELS_TEST_FN}[{name}]" for name in names],
            MODELS_ARTIFACTS_DIR,
        ),
        None: (
            names,
            default_artifacts_dir,
        ),  # Raw mode: names are already full node IDs
    }

    node_ids, default_artifacts = source_config[source]
    artifacts_dir = (
        Path(artifacts_dir_override) if artifacts_dir_override else default_artifacts
    )
    return node_ids, artifacts_dir