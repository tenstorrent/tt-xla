# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import sys
import os

if __name__ == "__main__":
    test_list = []
    if os.path.exists(".pytest_tests_to_run"):
        with open(".pytest_tests_to_run", "r") as fd:
            test_list = [line.strip() for line in fd.readlines()]
    sys.exit(pytest.main(test_list + sys.argv[1:]))
