# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import glob
import json
import os
import re
import sys

from find_all_failed_tests import find_failed_test_case


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
                    # print(f"Found directory for machine: {machine_name} at {dir_path}")
                    machine_failed_tests = []
                    for xml_file in xml_files:
                        file_failed_tests = find_failed_test_case(xml_file)
                        machine_failed_tests.extend(file_failed_tests)

                    if machine_failed_tests:
                        # print(f"  Found {len(machine_failed_tests)} failed tests for machine {machine_name}")
                        if machine_name not in failed_tests:
                            failed_tests[machine_name] = []
                        failed_tests[machine_name].extend(machine_failed_tests)

    return failed_tests


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
    failed_a = find_xml_files(dir_a)
    print_failed_tests_by_machine(failed_a)
    print(f"------ Finding failed tests in {dir_b}")
    failed_b = find_xml_files(dir_b)
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

    # Remove machines with empty test arrays
    new_failed_tests = {
        machine: tests for machine, tests in new_failed_tests.items() if tests
    }

    print(f"New failed tests: {new_failed_tests}")
    return new_failed_tests


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

    nft = find_new_failed_tests(dir_a, dir_b, test_filter)
    if not nft:
        print("No new failed tests found.")
        sys.exit(3)

    matrix = []
    upload_list = []
    # Create output directories and write test files
    for machine, tests in nft.items():
        machine_dir = os.path.join(dir_out, machine)
        os.makedirs(machine_dir, exist_ok=True)

        tests_file = os.path.join(machine_dir, ".tests_to_run")
        with open(tests_file, "w") as f:
            for test in tests:
                f.write(f"{test}\n")

        upload_list.append({"name": f"{machine}-test-to-run", "path": tests_file})
        for i in range(num_runs):
            matrix.append(
                {
                    "runs-on": machine,
                    "fromc": fromc,
                    "toc": toc,
                    "shared-runners": machine.startswith("tt-"),
                    "run_no": i,
                }
            )

    # Write upload_list to .upload_list.json file
    with open(".upload_list.json", "w") as f:
        json.dump(upload_list, f, indent=2)

    # Write matrix to .matrix.json file
    with open(".matrix.json", "w") as f:
        json.dump(matrix, f)

    sys.exit(0)
