#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import ast
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET

MAX_CHARS = 400
MAX_SUMMARY_CHARS = 50


def parse_junit_xml(xml_file, verbose=False):
    """Parse JUnit XML file and extract test information."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Find all testcase elements
        for testcase in root.findall(".//testcase"):
            # Extract specific_test_case from tags property
            specific_test_case = "Not found"
            error_message = ""
            bringup_status = "Not specified"

            # Look for properties in this testcase
            properties = testcase.find("properties")
            if properties is not None:
                for prop in properties:
                    if prop.get("name") == "tags":
                        tags_value = prop.get("value", "{}")
                        try:
                            # Use literal_eval for safety; tags_value is expected to be Python dict syntax
                            tags_dict = ast.literal_eval(tags_value)
                            specific_test_case = tags_dict.get(
                                "specific_test_case", "Not found"
                            )
                            bringup_status = tags_dict.get(
                                "bringup_status", "Not specified"
                            )
                        except:
                            specific_test_case = "Failed to parse tags"
                    elif prop.get("name") == "error_message":
                        error_message = prop.get("value", "N/A")

            if verbose:
                # Trim error_message to MAX_CHARS characters for verbose output
                verbose_error = error_message
                if len(verbose_error) > MAX_CHARS:
                    verbose_error = verbose_error[: MAX_CHARS - 3] + "..."
                print(f"specific_test_case: {specific_test_case}")
                print(f"error_message: {verbose_error}")
                print(f"bringup_status: {bringup_status}")
                print()  # Empty line between tests
            else:
                # One-line concise output: bringup_status specific_test_case error_message (trimmed to 50 chars)
                one_line_error = " ".join(error_message.splitlines())
                if len(one_line_error) > MAX_SUMMARY_CHARS:
                    one_line_error = one_line_error[: MAX_SUMMARY_CHARS - 3] + "..."
                print(
                    f"{bringup_status:<20} {specific_test_case:<140} {one_line_error}"
                )

    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"File not found: {xml_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Parse JUnit XML file and extract test information"
    )
    parser.add_argument(
        "--xml",
        nargs="+",
        required=True,
        help="Path(s) or glob pattern(s) to JUnit XML file(s), e.g. --xml results.xml or --xml 'dir/*.xml'",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Display full details for each test"
    )

    args = parser.parse_args()
    # Expand globs and collect files
    matched_files = []
    seen = set()
    for pattern in args.xml:
        # First try glob expansion
        for path in glob.glob(pattern):
            if path not in seen:
                matched_files.append(path)
                seen.add(path)
        # If no glob pattern, treat as direct path if it exists
        if not glob.has_magic(pattern) and os.path.exists(pattern):
            if pattern not in seen:
                matched_files.append(pattern)
                seen.add(pattern)

    if not matched_files:
        print("No XML files matched the provided --xml pattern(s).", file=sys.stderr)
        sys.exit(1)

    for xml_file in matched_files:
        parse_junit_xml(xml_file, verbose=args.verbose)


if __name__ == "__main__":
    main()
