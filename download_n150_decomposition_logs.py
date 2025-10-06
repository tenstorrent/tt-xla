#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Script to download n150 decomposition logs from GitHub Actions artifacts.

This script downloads all n150 test logs from expected_passing (jobs 0-9)
and xfail test groups, extracts decomposition logs, and combines them.

Usage:
    python download_n150_decomposition_logs.py

Requirements:
    - GitHub CLI (gh) must be installed and authenticated
    - Run from the tt-xla repository root
"""

import subprocess
import json
import os
import zipfile
import tempfile
import shutil
from pathlib import Path

# Configuration
RUN_ID = "17772896580"
REPO = "tenstorrent/tt-xla"
OUTPUT_DIR = "n150_decomposition_logs"

# All n150 artifacts to download (by artifact ID)
ARTIFACTS = {
    # Expected Passing Jobs 0-9 (successful)
    "4027078597": "test-log-n150-expected_passing-50513912961",  # job 0
    "4027341294": "test-log-n150-expected_passing-50513912979",  # job 3
    "4027093386": "test-log-n150-expected_passing-50513912954",  # job 4
    "4027228974": "test-log-n150-expected_passing-50513912934",  # job 5
    "4027207828": "test-log-n150-expected_passing-50513912953",  # job 6
    "4027364507": "test-log-n150-expected_passing-50513912952",  # job 7
    "4027315389": "test-log-n150-expected_passing-50513912942",  # job 9
    # Expected Passing Jobs 0-9 (failed)
    "4027147975": "test-log-n150--50513912963",  # job 1
    "4027271229": "test-log-n150--",  # job 2
    "4027282476": "test-log-n150--50513912925",  # job 8
    # XFail Tests
    "4026151019": "test-log-n150-known_failure_xfail_or_not_supported_skip_or_placeholder-50513912818",
    "4026291220": "test-log-n150-known_failure_xfail_or_not_supported_skip_or_placeholder-",
    "4026352368": "test-log-n150-known_failure_xfail_or_not_supported_skip_or_placeholder-50513912828",
}


def run_command(cmd, check=True):
    """Run a shell command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=check)
        return result.stdout.strip(), result.stderr.strip()
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        if check:
            raise
        return "", e.stderr


def download_artifact(artifact_id, artifact_name, category):
    """Download a specific artifact."""
    print(f"Downloading artifact: {artifact_name}")

    # Create category directory
    category_dir = os.path.join(OUTPUT_DIR, category)
    os.makedirs(category_dir, exist_ok=True)

    # Create artifact directory
    artifact_dir = os.path.join(category_dir, artifact_name)
    os.makedirs(artifact_dir, exist_ok=True)

    # Download to temporary file
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_file:
        cmd = ["gh", "api", f"repos/{REPO}/actions/artifacts/{artifact_id}/zip"]

        try:
            # Download the artifact
            result = subprocess.run(cmd, check=True, stdout=tmp_file)
            tmp_file.flush()

            # Extract the zip file
            with zipfile.ZipFile(tmp_file.name, "r") as zip_file:
                zip_file.extractall(artifact_dir)

            print(f"  ✓ Extracted to: {artifact_dir}")
            return artifact_dir

        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to download artifact {artifact_name}: {e}")
            shutil.rmtree(artifact_dir, ignore_errors=True)
            return None
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file.name)
            except OSError:
                pass


def extract_decomposition_logs(artifact_dir, artifact_name, category):
    """Extract decomposition logs from a downloaded artifact."""
    decomp_logs = []

    # Search for pytest.log files or any files containing decomposition logs
    for root, dirs, files in os.walk(artifact_dir):
        for file in files:
            if file == "pytest.log" or "log" in file.lower():
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                        # Look for decomposition log sections
                        if "DECOMPOSITION OPERATIONS LOG" in content:
                            # Extract the structured decomposition section
                            start_marker = "DECOMPOSITION OPERATIONS LOG"
                            lines = content.split("\n")
                            start_idx = -1
                            end_idx = -1

                            for i, line in enumerate(lines):
                                if start_marker in line:
                                    start_idx = i
                                elif start_idx != -1 and line.startswith("=" * 60):
                                    end_idx = i
                                    break

                            if start_idx != -1 and end_idx != -1:
                                decomp_section = "\n".join(
                                    lines[start_idx + 2 : end_idx]
                                )
                                if decomp_section.strip():
                                    decomp_logs.append(
                                        {
                                            "content": decomp_section.strip(),
                                            "source": f"{category}/{artifact_name}/{file}",
                                            "type": "structured",
                                        }
                                    )
                                    print(
                                        f"  ✓ Found structured decomposition log in: {file}"
                                    )

                        # Also look for individual decomposition lines
                        decomp_lines = []
                        for line in content.split("\n"):
                            line = line.strip()
                            if any(
                                pattern in line
                                for pattern in [
                                    "=== MODEL:",
                                    "decomposition_core_aten:",
                                    "decomposition_custom:",
                                    "decomposition_default:",
                                ]
                            ):
                                decomp_lines.append(line)

                        if decomp_lines:
                            decomp_logs.append(
                                {
                                    "content": "\n".join(decomp_lines),
                                    "source": f"{category}/{artifact_name}/{file}",
                                    "type": "individual_lines",
                                }
                            )
                            print(
                                f"  ✓ Found {len(decomp_lines)} decomposition lines in: {file}"
                            )

                except Exception as e:
                    print(f"  ⚠ Error reading file {file_path}: {e}")

    return decomp_logs


def categorize_artifact(artifact_name):
    """Determine the category for an artifact."""
    if "expected_passing" in artifact_name or artifact_name.startswith(
        "test-log-n150--"
    ):
        return "expected_passing"
    elif "known_failure_xfail" in artifact_name:
        return "xfail"
    else:
        return "other"


def main():
    """Main function to download and process artifacts."""
    print(f"Downloading n150 decomposition logs from GitHub Actions run {RUN_ID}")
    print(f"Repository: {REPO}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Total artifacts to download: {len(ARTIFACTS)}")
    print()

    # Check if gh CLI is available
    try:
        run_command(["gh", "--version"])
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: GitHub CLI (gh) is not installed or not in PATH")
        print("Please install it from: https://cli.github.com/")
        return 1

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Download each artifact
    all_decomp_logs = []
    expected_passing_logs = []
    xfail_logs = []

    for artifact_id, artifact_name in ARTIFACTS.items():
        category = categorize_artifact(artifact_name)
        print(
            f"\n[{len(all_decomp_logs)+1}/{len(ARTIFACTS)}] Processing: {artifact_name} ({category})"
        )

        artifact_dir = download_artifact(artifact_id, artifact_name, category)
        if artifact_dir:
            logs = extract_decomposition_logs(artifact_dir, artifact_name, category)
            all_decomp_logs.extend(logs)

            if category == "expected_passing":
                expected_passing_logs.extend(logs)
            elif category == "xfail":
                xfail_logs.extend(logs)

    # Create combined log file
    combined_log_file = os.path.join(OUTPUT_DIR, "combined_decomposition_logs.txt")

    print(f"\n📝 Creating combined decomposition log...")
    with open(combined_log_file, "w") as f:
        f.write(f"# N150 Decomposition Logs from GitHub Actions Run {RUN_ID}\n")
        f.write(f"# Repository: {REPO}\n")
        f.write(f"# Downloaded on: {os.popen('date').read().strip()}\n")
        f.write(f"# Total artifacts processed: {len(ARTIFACTS)}\n")
        f.write(f"# Expected passing logs: {len(expected_passing_logs)}\n")
        f.write(f"# XFail logs: {len(xfail_logs)}\n\n")

        if expected_passing_logs:
            f.write("=" * 80 + "\n")
            f.write("EXPECTED PASSING TESTS (jobs 0-9)\n")
            f.write("=" * 80 + "\n\n")

            for log in expected_passing_logs:
                f.write(f"# Source: {log['source']}\n")
                f.write(f"# Type: {log['type']}\n")
                f.write(log["content"] + "\n\n")

        if xfail_logs:
            f.write("=" * 80 + "\n")
            f.write("XFAIL TESTS\n")
            f.write("=" * 80 + "\n\n")

            for log in xfail_logs:
                f.write(f"# Source: {log['source']}\n")
                f.write(f"# Type: {log['type']}\n")
                f.write(log["content"] + "\n\n")

    print(f"✅ Download complete!")
    print(f"📁 Artifacts downloaded to: {OUTPUT_DIR}/")
    print(f"📄 Combined decomposition logs: {combined_log_file}")
    print(f"📊 Summary:")
    print(f"   - Total artifacts: {len(ARTIFACTS)}")
    print(f"   - Expected passing logs found: {len(expected_passing_logs)}")
    print(f"   - XFail logs found: {len(xfail_logs)}")
    print(f"   - Total decomposition log sections: {len(all_decomp_logs)}")

    return 0


if __name__ == "__main__":
    exit(main())
