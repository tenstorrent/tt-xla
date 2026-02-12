# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Extract failure context from JUnit XML reports and pytest logs for Claude Code.

Produces a structured summary of test failures suitable as input context for
Claude Code CLI when auto-fixing transformers compatibility issues.
"""

import glob
import os
import re
import sys
import xml.etree.ElementTree as ET

MAX_OUTPUT_BYTES = 50_000


def extract_failures_from_xml(xml_dir):
    """Extract failed test names and error messages from JUnit XML reports."""
    xml_files = glob.glob(os.path.join(xml_dir, "**", "*.xml"), recursive=True)
    failures = []

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except ET.ParseError as e:
            print(f"Warning: could not parse {xml_file}: {e}", file=sys.stderr)
            continue

        for testsuite in root.findall("testsuite"):
            for testcase in testsuite.findall("testcase"):
                classname = testcase.get("classname", "")
                name = testcase.get("name", "")
                test_id = f"{classname.replace('.', '/')}.py::{name}"

                failure = testcase.find("failure")
                error = testcase.find("error")
                element = failure if failure is not None else error
                if element is not None:
                    message = element.get("message", "")
                    text = (element.text or "")[:2000]  # cap per-failure
                    failures.append(
                        {
                            "test": test_id,
                            "type": element.tag,
                            "message": message,
                            "traceback": text,
                        }
                    )

    return failures


def extract_errors_from_logs(logs_dir):
    """Grep pytest logs for import/attribute errors and failure summaries."""
    log_files = glob.glob(os.path.join(logs_dir, "**", "pytest.log"), recursive=True)
    error_patterns = re.compile(
        r"(FAILED|ERROR|ModuleNotFoundError|ImportError|AttributeError"
        r"|NameError|TypeError.*argument|cannot import name)",
        re.IGNORECASE,
    )
    excerpts = []

    for log_file in log_files:
        try:
            with open(log_file, "r", errors="replace") as f:
                lines = f.readlines()
        except OSError:
            continue

        matching_lines = []
        for i, line in enumerate(lines):
            if error_patterns.search(line):
                # Include context: 2 lines before, 10 lines after
                start = max(0, i - 2)
                end = min(len(lines), i + 11)
                chunk = "".join(lines[start:end])
                matching_lines.append(chunk)

        if matching_lines:
            # Deduplicate consecutive overlapping chunks
            seen = set()
            unique = []
            for chunk in matching_lines:
                key = chunk[:200]
                if key not in seen:
                    seen.add(key)
                    unique.append(chunk)
            excerpts.append(
                {"log_file": os.path.basename(os.path.dirname(log_file)), "errors": unique[:30]}
            )

    return excerpts


def format_output(xml_failures, log_excerpts):
    """Format failures into a structured text summary."""
    parts = []

    if xml_failures:
        parts.append(f"=== FAILED TESTS ({len(xml_failures)} total) ===\n")
        for f in xml_failures:
            parts.append(f"TEST: {f['test']}")
            parts.append(f"TYPE: {f['type']}")
            if f["message"]:
                parts.append(f"MESSAGE: {f['message']}")
            if f["traceback"]:
                parts.append(f"TRACEBACK:\n{f['traceback']}")
            parts.append("---")

    if log_excerpts:
        parts.append("\n=== ERROR EXCERPTS FROM LOGS ===\n")
        for excerpt in log_excerpts:
            parts.append(f"--- Log: {excerpt['log_file']} ---")
            for error in excerpt["errors"]:
                parts.append(error)
                parts.append("~~~")

    output = "\n".join(parts)

    # Enforce size cap
    if len(output.encode("utf-8")) > MAX_OUTPUT_BYTES:
        output = output[: MAX_OUTPUT_BYTES - 100]
        output += "\n\n[TRUNCATED - output exceeded 50KB limit]"

    return output


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python extract_transformers_failure_context.py <xml_dir> <logs_dir> [output_file]"
        )
        sys.exit(1)

    xml_dir = sys.argv[1]
    logs_dir = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "failure_context.txt"

    xml_failures = extract_failures_from_xml(xml_dir)
    log_excerpts = extract_errors_from_logs(logs_dir)

    if not xml_failures and not log_excerpts:
        print("No failures found.")
        with open(output_file, "w") as f:
            f.write("No failures found in test reports or logs.\n")
        return

    output = format_output(xml_failures, log_excerpts)
    with open(output_file, "w") as f:
        f.write(output)

    print(f"Wrote failure context to {output_file} ({len(output)} bytes)")
    print(f"  XML failures: {len(xml_failures)}")
    print(f"  Log excerpts: {sum(len(e['errors']) for e in log_excerpts)}")


if __name__ == "__main__":
    main()
