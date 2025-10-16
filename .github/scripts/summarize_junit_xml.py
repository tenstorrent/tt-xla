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


# Hack to set arch based on filename since not present inside xml file currently.
def get_arch_from_filename(filename):
    """
    Get the arch from the filename.
    """
    if "report_" in filename:
        arch = filename.split("report_")[0]
        # Remove trailing underscore if it exists
        if arch.endswith("_"):
            arch = arch[:-1]
    else:
        arch = "unknown"
    return arch


def parse_junit_xml(xml_file, verbose=False, print_header=True):
    """Parse JUnit XML file and extract test information for test_all_models cases.

    Note: verbose output is intentionally ignored; this function always prints a single-line summary per test.
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        arch = get_arch_from_filename(xml_file)

        if print_header:
            print(
                f"{'BringupStatus':<20} {'PCCThreshold':<15} {'PCC':<10} {'Flag':<15} {'ModelGroup':<12} {'Arch':<12} {'SpecificTestCase'}"
            )

        # Find all testcase elements
        for testcase in root.findall(".//testcase"):
            # Only consider tests with 'test_all_models' in the junit testcase name
            testcase_name = testcase.get("name", "")
            if "test_all_models" not in testcase_name:
                continue

            specific_test_case = "Not found"
            bringup_status = "Not specified"
            pcc_value = None
            pcc_threshold_value = None
            pcc_assertion_enabled = None
            model_group = "Not specified"

            # Look for properties in this testcase
            properties = testcase.find("properties")
            if properties is not None:
                for prop in properties:

                    if prop.get("name") == "group":
                        model_group = prop.get("value", "Not specified")

                    if prop.get("name") == "tags":
                        tags_value = prop.get("value", "{}")
                        try:
                            # Use literal_eval for safety; tags_value is expected to be Python dict syntax
                            tags_dict = ast.literal_eval(tags_value)
                            specific_test_case = tags_dict.get(
                                "specific_test_case", specific_test_case
                            )
                            bringup_status = tags_dict.get(
                                "bringup_status", bringup_status
                            )
                            pcc_assertion_enabled = tags_dict.get(
                                "pcc_assertion_enabled", pcc_assertion_enabled
                            )
                            pcc_value = float(tags_dict.get("pcc", pcc_value))
                            pcc_threshold_value = float(
                                tags_dict.get("pcc_threshold", pcc_threshold_value)
                            )
                        except Exception:
                            specific_test_case = "Failed to parse tags"

            # Compute promotion flag if possible for interesting cases that can be acted upon.
            promotion_flag = "NONE"
            try:
                if pcc_value is not None and pcc_threshold_value is not None:

                    # Want to detect appreciable PCC bumps based on first 2 decimal places.
                    pcc_rounded_value = round(pcc_value, 2)
                    pcc_rounded_threshold = round(pcc_threshold_value, 2)
                    pcc_bump = pcc_rounded_value > pcc_rounded_threshold

                    # PCC Checking is enabled, but PCC threshold can be raised.
                    if (
                        pcc_assertion_enabled is not None
                        and pcc_assertion_enabled == True
                    ):
                        if pcc_bump and pcc_threshold_value < 0.99:
                            if pcc_value > 0.99:
                                promotion_flag = "RAISE_PCC_099"
                            else:
                                promotion_flag = "RAISE_PCC"

                    else:
                        if pcc_value >= pcc_threshold_value:

                            if pcc_threshold_value < 0.99 and pcc_value > 0.99:
                                promotion_flag = "ENABLE_PCC_099"
                            else:
                                promotion_flag = "ENABLE_PCC"

                    # # Case 3: PCC is greater than threshold and threshold is less than 0.99 and PCC is less than 0.99
                    # if pcc_value > pcc_threshold_value and pcc_threshold_value < 0.99 and pcc_value < 0.99:
                    #     promotion_flag = "ENABLE_PCC"
                    # if pcc_value > pcc_threshold_value and pcc_threshold_value == 0.99 and pcc_value >= 0.99:
                    #     promotion_flag = "ENABLE_PCC_099"

                    # List the cases we care about:
                    # - Can raise threshold higher (good) RAISE_PCC
                    # - Can raise threshold to 0.99 (amazing) RAISE_PCC_099
                    # - Can enable PCC checking with current threshold (good) ENABLE_PCC
                    # - Can enable PCC checking with 0.99 threshold (amazing) ENABLE_PCC_099

            except Exception:
                # If conversion fails, leave promotion_flag empty
                promotion_flag = ""

            # Format numeric columns safely
            def fmt_float(value):
                try:
                    if value is None:
                        return ""
                    return f"{float(value):.6f}"
                except Exception:
                    return str(value)

            pcc_str = fmt_float(pcc_value)
            pcc_threshold_str = fmt_float(pcc_threshold_value)

            # Always print a single concise line: status, pcc_threshold, pcc, flag, test identifier
            # Keep columns compact but readable
            print(
                f"{bringup_status:<20} {pcc_threshold_str:<15} {pcc_str:<10} {promotion_flag:<15} {model_group:<12} {arch:<12} {specific_test_case}"
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

    # Print header only once.
    print_header = True
    for xml_file in matched_files:
        parse_junit_xml(xml_file, verbose=args.verbose, print_header=print_header)
        print_header = False


if __name__ == "__main__":
    main()
