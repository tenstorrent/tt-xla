# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import json
import sys

TESTS_TO_PARALLELIZE = [{"runs-on": "n150", "name": "run_jax"}]


def modify_by_test_case(test_matrix, test_to_parallelize, test_group_cnt):
    target_object = None
    for obj in test_matrix:
        if obj.get("runs-on") == test_to_parallelize.get("runs-on") and obj.get(
            "name"
        ) == test_to_parallelize.get("name"):
            target_object = obj
            break

    if not target_object:
        print("Test to parallelize not found")
        return

    test_matrix[:] = [obj for obj in test_matrix if obj != target_object]

    for i in range(1, test_group_cnt + 1):
        new_object = target_object.copy()
        new_object["test_group_cnt"] = test_group_cnt
        new_object["test_group_id"] = i
        test_matrix.append(new_object)


def modify_test_matrix(file_path, test_group_cnt):
    with open(file_path, "r") as f:
        test_matrix = json.load(f)

    for test_to_parallelize in TESTS_TO_PARALLELIZE:
        modify_by_test_case(test_matrix, test_to_parallelize, test_group_cnt)

    return json.dumps(test_matrix, indent=4)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python modify_test_matrix.py <file_path> <test_group_cnt>")
        sys.exit(1)

    file_path = sys.argv[1]
    test_group_cnt = int(sys.argv[2])

    modified_matrix = modify_test_matrix(file_path, test_group_cnt)
    print(modified_matrix)
