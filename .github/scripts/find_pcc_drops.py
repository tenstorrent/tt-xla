# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET


def find_test_case_pcc(xml_file):
    """
    Extract test cases with PCC values from a JUnit XML report.

    Args:
        xml_file (str): Path to the XML file.

    Returns:
        array: A list of test cases names, pcc, and pcc threshold.
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        pcc_test_cases = []

        for testsuite in root.findall("testsuite"):
            # Iterate over all <testcase> elements within the current <testsuite>
            for testcase in testsuite.findall("testcase"):
                try:
                    path = testcase.get("classname").replace(".", "/")
                    name = testcase.get("name")

                    # Check if this test case has PCC tag
                    properties = testcase.find("properties")
                    if properties is not None:
                        for prop in properties.findall("property"):
                            if prop.get("name") == "tags":
                                # Replace Python-style values with JSON-compatible ones
                                tag_value = (
                                    prop.get("value", "")
                                    .replace("'", '"')
                                    .replace("None", "null")
                                    .replace("True", "true")
                                    .replace("False", "false")
                                    .replace("nan", "null")
                                    .replace("inf", "null")
                                )
                                try:
                                    tags_json = json.loads(tag_value)
                                    if "pcc" not in tags_json:
                                        continue
                                    pcc_raw_value = tags_json["pcc"]
                                    if not pcc_raw_value:
                                        continue
                                    pcc_value = float(pcc_raw_value)
                                    pcc_threshold = float(
                                        tags_json.get("pcc_threshold", 0.0)
                                    )
                                    pcc_test_cases.append(
                                        {
                                            "test": f"{path}.py::{name}",
                                            "pcc": pcc_value,
                                            "pcc_threshold": pcc_threshold,
                                        }
                                    )
                                except (
                                    json.JSONDecodeError,
                                    ValueError,
                                    TypeError,
                                ) as err:
                                    print(
                                        f"Error parsing PCC tags in {xml_file} for test case '{name}': {err}"
                                    )
                except:
                    print(f"Error encountered in {xml_file} for test case '{name}'")

    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_file}: {e}")

    return pcc_test_cases


def get_pcc_from_xml_files(dir):
    """
    Find all *.xml files in the given directory and its subdirectories.
    Extract and return PCC test cases.

    Args:
        dir (str): Root directory to search in
    """
    xml_files = glob.glob(os.path.join(dir, "**", "*.xml"), recursive=True)

    pcc_test_cases = []
    for xml_file in xml_files:
        pcc_test_cases.extend(find_test_case_pcc(xml_file))

    return pcc_test_cases


def bisect_pcc(dir_a, dir_b, margin=0.01):
    pcc_a = get_pcc_from_xml_files(dir_a)
    pcc_b = get_pcc_from_xml_files(dir_b)

    # Create a dictionary for quick lookup of test cases from pcc_b
    pcc_b_dict = {case["test"]: case for case in pcc_b}
    pcc_test_cases = []

    # Find matching test cases where pcc_a has lower PCC than pcc_b with margin
    for case_a in pcc_a:
        test_name = case_a["test"]
        if test_name in pcc_b_dict:
            case_b = pcc_b_dict[test_name]
            # Check if pcc_a is lower than pcc_b by at least the margin
            if case_a["pcc"] < (case_b["pcc"] - margin):
                case_a["diff"] = case_b["pcc"] - case_a["pcc"]
                case_a["good"] = case_a["pcc"] < case_a["pcc_threshold"]
                pcc_test_cases.append(case_a)

    if len(pcc_test_cases) > 0:
        # sort by difference
        pcc_test_cases.sort(key=lambda x: x["diff"], reverse=True)
        # Save PCC test cases to a file
        print(f"\nFound {len(pcc_test_cases)} PCC test cases below threshold:")
        with open("pcc_results.md", "w") as f:
            f.write(
                f"# PCC Drop Bisect Results: \n\n{len(pcc_test_cases)} test cases found with PCC drop below margin {margin}:\n\n"
            )
            f.write("| Good | Test | Drop | PCC | PCC Threshold |\n")
            f.write("|------|------|------|-----|---------------|\n")
            for pcc_case in pcc_test_cases:
                good_icon = "✅" if pcc_case["good"] else "❌"
                f.write(
                    f"| {good_icon} | {pcc_case['test']} | {pcc_case['diff']:.4f} | {pcc_case['pcc']:.4f} | {pcc_case['pcc_threshold']:.4f} |\n"
                )
                print(
                    f" - {good_icon} | {pcc_case['test']} | {pcc_case['diff']:.4f} | {pcc_case['pcc']:.4f} | {pcc_case['pcc_threshold']:.4f}"
                )
    else:
        print("No PCC test cases with pcc drop found.")


if __name__ == "__main__":
    if len(sys.argv) == 4:
        margin = float(sys.argv[3])
        bisect_pcc(sys.argv[1], sys.argv[2], margin)
    elif len(sys.argv) == 3:
        bisect_pcc(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python find_pcc_drops.py <dir_a> <dir_b> [margin]")
        sys.exit(1)
