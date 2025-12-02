# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import glob
import os
import sys
import xml.etree.ElementTree as ET


def find_failed_test_case(xml_file):
    """
    Extract failed test case names from a JUnit XML report.

    Args:
        xml_file (str): Path to the XML file.

    Returns:
        array: A list of failed test case names.
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        failed_test_cases = []

        for testsuite in root.findall("testsuite"):
            # Iterate over all <testcase> elements within the current <testsuite>
            for testcase in testsuite.findall("testcase"):
                try:
                    path = testcase.get("classname").replace(".", "/")
                    name = testcase.get("name")
                    if (testcase.find("failure") is not None) or (
                        testcase.find("error") is not None
                    ):
                        failed_test_cases.append(f"{path}.py::{name}")
                except:
                    print(f"Error encountered in {xml_file} for test case '{name}'")

    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_file}: {e}")

    return failed_test_cases


def find_xml_files(dir):
    """
    Find all *.xml files in the given directory and its subdirectories.
    Saves and prints all failed test cases to a file.

    Args:
        dir (str): Root directory to search in
    """
    xml_files = glob.glob(os.path.join(dir, "**", "*.xml"), recursive=True)

    failed_test_cases = []
    for xml_file in xml_files:
        failed_test_cases.extend(find_failed_test_case(xml_file))

    if len(failed_test_cases) > 0:
        # Save failed test cases to a file
        print(f"Found {len(failed_test_cases)} failed test cases:")
        for test_case in failed_test_cases:
            print(f" - {test_case}")

        with open(".pytest_tests_to_run", "w") as f:
            for test_case in failed_test_cases:
                f.write(f"{test_case}\n")
    else:
        print("No failed test cases found.")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        find_xml_files(".")
    elif len(sys.argv) == 2:
        find_xml_files(sys.argv[1])
    else:
        print("Usage: python find_all_failed_tests.py [directory]")
        sys.exit(1)
