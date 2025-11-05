# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

if __name__ == "__main__":
    with open(".pytest_tests_to_run", "r") as fd:
        test_list = [line.strip() for line in fd.readlines()]

    # Optional mode: run each test in a separate pytest invocation
    args = sys.argv[1:]
    separate = False
    if "--separate-tests" in args:
        separate = True
        args = [a for a in args if a != "--separate-tests"]

    if separate:
        exit_code = 0
        for test in test_list:
            print(f"======================== Running test individually: {test}")
            # Mark the current test for crash reporting compatibility
            try:
                with open(".pytest_current_test_executing", "w") as f:
                    f.write(test)
            except Exception:
                pass
            rc = pytest.main([test] + args)
            if rc != 0 and exit_code == 0:
                exit_code = rc
        sys.exit(exit_code)
    else:
        sys.exit(pytest.main(test_list + args))
