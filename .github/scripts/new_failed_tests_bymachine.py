# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET


def find_test_cases(xml_file):
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
        success_test_cases = []

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
                    else:
                        success_test_cases.append(f"{path}.py::{name}")
                except:
                    print(f"Error encountered in {xml_file} for test case '{name}'")

    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_file}: {e}")

    return failed_test_cases, success_test_cases


def find_xml_files(dir):
    """
    Find all subdirectories with report xml files in the given directory and its subdirectories.
    Parse failed tests in each of the xml files

    Args:
        dir (str): Root directory to search in
    Returns:
        failed tests in an array
    """
    # Find directories matching the regex pattern and extract the matched group
    failed_tests = {}
    success_tests = {}
    pattern = r"test-reports-[^-]+-[^-]+-(.+)-[^-]*-[^-]+"

    for root, dirs, _ in os.walk(dir):
        for d in dirs:
            match = re.match(pattern, d)
            if match:
                machine_name = match.group(1)
                dir_path = os.path.join(root, d)
                # Check if directory contains XML files
                xml_files = glob.glob(os.path.join(dir_path, "*.xml"))
                if xml_files:
                    machine_failed_tests = []
                    machine_success_tests = []
                    for xml_file in xml_files:
                        file_failed_tests, file_success_tests = find_test_cases(
                            xml_file
                        )
                        machine_failed_tests.extend(file_failed_tests)
                        machine_success_tests.extend(file_success_tests)

                    if machine_failed_tests:
                        if machine_name not in failed_tests:
                            failed_tests[machine_name] = []
                        failed_tests[machine_name].extend(machine_failed_tests)
                    if machine_success_tests:
                        if machine_name not in success_tests:
                            success_tests[machine_name] = []
                        success_tests[machine_name].extend(machine_success_tests)
    return failed_tests, success_tests


def print_failed_tests_by_machine(failed_tests):
    """
    Print failed tests by machine.

    Args:
        failed_tests (dict): Dictionary with machine names as keys and lists of failed tests as values.
    """
    for machine, tests in failed_tests.items():
        print(f"Machine: {machine} {len(tests)} failed tests")
        # for test in tests:
        #     print(f"  {test}")


def find_new_failed_tests(dir_a, dir_b, filter):
    """
    Find newly failed tests removing all tests that are in dir_b and not in dir_a.

    Args:
        dir_a, dir_b with test reports

    Returns:
        List containing tests that are in dir_a and not in dir_b by machine.
    """
    print(f"------ Finding failed tests in {dir_a}")
    failed_a, _ = find_xml_files(dir_a)
    print_failed_tests_by_machine(failed_a)
    print(f"------ Finding failed tests in {dir_b}")
    failed_b, succeed_b = find_xml_files(dir_b)
    print_failed_tests_by_machine(failed_b)

    # Apply test filter if provided
    if filter:
        failed_a = {
            k: [t for t in v if re.search(filter, t)] for k, v in failed_a.items()
        }
        failed_b = {
            k: [t for t in v if re.search(filter, t)] for k, v in failed_b.items()
        }

    # Bisect failed tests by removing all tests that are in dir_b and not in dir_a
    new_failed_tests = {}
    for machine, tests in failed_a.items():
        if machine not in failed_b:
            new_failed_tests[machine] = tests
        else:
            new_failed_tests[machine] = [t for t in tests if t not in failed_b[machine]]

    indetermined_tests = {}
    for machine, tests in failed_a.items():
        if machine not in failed_b and machine not in succeed_b:
            indetermined_tests[machine] = tests
        else:
            indetermined_tests[machine] = [
                t
                for t in tests
                if t not in failed_b.get(machine, [])
                and t not in succeed_b.get(machine, [])
            ]

    # Remove machines with empty test arrays
    new_failed_tests = {
        machine: tests for machine, tests in new_failed_tests.items() if tests
    }

    # Skip galaxy machines - limited runner availability makes bisecting impractical
    new_failed_tests = {
        machine: tests
        for machine, tests in new_failed_tests.items()
        if "galaxy" not in machine and "glx" not in machine
    }

    print(f"New failed tests: {new_failed_tests}")
    return new_failed_tests, indetermined_tests


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print(
            "Usage: python find_failed_bymachine_tests.py reports_dir_a reports_dir_b reports_dir_out form_commit to_commit test_filter num_runs"
        )
        sys.exit(1)

    dir_a = sys.argv[1]
    dir_b = sys.argv[2]
    dir_out = sys.argv[3]
    fromc = sys.argv[4]
    toc = sys.argv[5]
    test_filter = sys.argv[6]
    num_runs = int(sys.argv[7])
    if num_runs < 1:
        print("WARNING: num_runs must be at least 1")
        num_runs = 1

    nft, indetermined_tests = find_new_failed_tests(dir_a, dir_b, test_filter)
    if not nft:
        print("No new failed tests found.")
        sys.exit(3)

    matrix = []
    # Create output matrix
    for machine, tests in nft.items():
        for i in range(num_runs):
            for test in tests:
                indeterminate = test in indetermined_tests.get(machine, [])
                matrix.append(
                    {
                        "runs-on": machine,
                        "fromc": fromc,
                        "toc": toc,
                        "shared-runners": machine.startswith("tt-"),
                        "test": test,
                        "indeterminate": indeterminate,
                        "run_no": i,
                    }
                )

    # Write matrix to .matrix.json file
    with open(".matrix.json", "w") as f:
        json.dump(matrix, f)

    sys.exit(0)
