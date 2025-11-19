#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Extract testcase names and fallback_ops from pytest XML report.
Outputs a CSV file that can be imported into Google Sheets.
"""

import ast
import csv
import sys
import xml.etree.ElementTree as ET


def parse_test_report(xml_file):
    """Parse the XML test report and extract testcase name and fallback_ops."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    results = []

    # Iterate through all testcases
    for testcase in root.findall(".//testcase"):
        test_name = testcase.get("name", "Unknown")

        # Find the tags property
        fallback_ops = None
        for prop in testcase.findall('.//property[@name="tags"]'):
            tags_value = prop.get("value", "{}")
            try:
                # Parse the string representation of the dict
                tags_dict = ast.literal_eval(tags_value)
                fallback_ops = tags_dict.get("fallback_ops", "N/A")

                # Convert list to string for CSV
                if isinstance(fallback_ops, list):
                    if len(fallback_ops) == 0:
                        fallback_ops = "[]"
                    else:
                        fallback_ops = str(fallback_ops)

            except (ValueError, SyntaxError) as e:
                print(
                    f"Warning: Could not parse tags for {test_name}: {e}",
                    file=sys.stderr,
                )
                fallback_ops = "Parse Error"

        if fallback_ops is None:
            fallback_ops = "Not Found"

        results.append({"test_name": test_name, "fallback_ops": fallback_ops})

    return results


def write_csv(results, output_file):
    """Write results to CSV file."""
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["test_name", "fallback_ops"])
        writer.writeheader()
        writer.writerows(results)


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_fallback_ops.py <xml_report_file> [output_csv]")
        print(
            "Example: python extract_fallback_ops.py push_torch_model_test_report.xml fallback_ops.csv"
        )
        sys.exit(1)

    xml_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "fallback_ops.csv"

    try:
        results = parse_test_report(xml_file)
        write_csv(results, output_file)
        print(f"Successfully extracted {len(results)} test cases to {output_file}")

        # Print summary
        has_fallback = sum(
            1
            for r in results
            if r["fallback_ops"] not in ["[]", "Not Found", "Parse Error"]
        )
        print(f"Tests with fallback ops: {has_fallback}/{len(results)}")

    except FileNotFoundError:
        print(f"Error: File '{xml_file}' not found", file=sys.stderr)
        sys.exit(1)
    except ET.ParseError as e:
        print(f"Error: Could not parse XML file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
