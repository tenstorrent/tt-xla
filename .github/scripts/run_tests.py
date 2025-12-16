# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest

if __name__ == "__main__":
    with open(".tests_to_run", "r") as fd:
        test_list = [line.strip() for line in fd.readlines()]
    sys.exit(pytest.main(test_list + sys.argv[1:]))
