#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import argparse
import ast
import glob
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

MAX_CHARS = 400
MAX_SUMMARY_CHARS = 50


def parse_junit_xml(xml_file, verbose=False, print_header=True):
    """Parse JUnit XML file and extract test information for test_all_models cases.

    Note: verbose output is intentionally ignored; this function always prints a single-line summary per test.
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        if print_header:
            print(
                f"{'BringupStatus':<20} {'PCCThreshold':<15} {'PCC':<10} {'Flag':<15} {'ModelGroup':<12} {'Arch':<12} {'SpecificTestCase'}"
            )

        results_dict = {}

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

                            arch = tags_dict.get("arch", "unknown")

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
                    pcc_rounded_value = math.floor(pcc_value * 100) / 100
                    pcc_rounded_threshold = math.floor(pcc_threshold_value * 100) / 100
                    pcc_bump = pcc_rounded_value > pcc_rounded_threshold

                    # Don't bump up unless over halfway point.
                    margin = 0.004

                    print(
                        f"pcc_rounded_value: {pcc_rounded_value} pcc_rounded_threshold: {pcc_rounded_threshold} pcc_bump: {pcc_bump} for testcase: {testcase_name}"
                    )

                    # PCC Checking is enabled, but PCC threshold can be raised.
                    if (
                        pcc_assertion_enabled is not None
                        and pcc_assertion_enabled == True
                    ):
                        if pcc_bump and pcc_threshold_value < 0.99:
                            if pcc_value > (0.990 + margin):
                                promotion_flag = "RAISE_PCC_099"
                            else:
                                promotion_flag = "RAISE_PCC_LESS"

                    else:
                        if pcc_value >= (pcc_threshold_value + margin):

                            if pcc_threshold_value < 0.99 and pcc_value > (
                                0.99 + margin
                            ):
                                promotion_flag = "ENABLE_PCC_RAISE_099"
                            elif pcc_threshold_value == 0.99 and pcc_value >= (
                                0.99 + margin
                            ):
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

            # Add to results_dict
            results_dict[testcase_name] = {
                "xml_file": xml_file,
                "bringup_status": bringup_status,
                "pcc_value": pcc_value,
                "pcc_threshold_value": pcc_threshold_value,
                "promotion_flag": promotion_flag,
                "model_group": model_group,
                "arch": arch,
            }

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

    return results_dict


def handle_results(result_dicts):
    """
    Handle the results.
    """

    # Combine the results from different archs.
    # combined_results = {}
    combined_results = defaultdict(lambda: defaultdict(list))
    model_groups = {}

    for result_dict in result_dicts:

        for test_name in result_dict:
            test_result = result_dict[test_name]
            if "xml_file" in test_result:
                xml_file = test_result["xml_file"]
                arch = test_result["arch"]
                promotion_flag = test_result["promotion_flag"]
                # Track model_group per test_name for combined printout
                model_groups[test_name] = test_result.get(
                    "model_group", "Not specified"
                )
                # print(
                #     f"XML file: {xml_file} - arch: {arch} - test name: {test_name} - promotion flag: {promotion_flag}"
                # )
                # combined_results[test_name][arch] = test_result

                combined_results[test_name][promotion_flag].append(arch)
                # We want to find cases where promotion flag matches for all archs.

    # Explicitly list supported_archs in tests makes things simpler.

    # Go through the combined results and
    for test_name in combined_results:
        for promotion_flag in combined_results[test_name]:

            # FIXME - Dirty Hack
            has_both = len(combined_results[test_name][promotion_flag]) == 2
            has_both_flag = "YES_BOTH_ARCH" if has_both else "NOT_BOTH_ARCH"
            archs_str = ", ".join(combined_results[test_name][promotion_flag])
            model_group = model_groups.get(test_name, "Not specified")
            print(
                f"COMBINED: promote: {promotion_flag:<15} group: {model_group:<12} arch: {archs_str:<15} {has_both_flag:<15} {test_name}"
            )


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

    result_dicts = []

    # Print header only once.
    print_header = True
    for xml_file in matched_files:
        result_dicts.append(
            parse_junit_xml(xml_file, verbose=args.verbose, print_header=print_header)
        )
        print_header = False

    handle_results(result_dicts)

    # TODO - Need some way to collect results in a dict then figure out what cases pass for both arch n150 and p150.

    # Need to return an object with fields instead of just printing.
    # How would automation work?
    # Go through all results

    # Query results from somewhere (artifacts or database)
    # Iterate over them all.
    # TODO - Need to map a result to particular test_config file to update (inference/training, single_device/tensor_parallel/data_parallel). Based on test name.
    # TODO - Need to determine if results/action (passing, raise PCC, enable PCC etc) is the same for all archs tested.
    # TODO - Need to read in file, do required updates. Do we update each arch and then collapse?
    # TODO - Need to convert python file to yaml file, and add ingestion logic (convert python enums to yaml or somethning)


if __name__ == "__main__":
    main()
