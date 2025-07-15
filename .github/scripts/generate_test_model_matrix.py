# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import json
import sys


def modify_test_matrix(file_path, test_group_cnt):
    # Load the JSON file
    with open(file_path, "r") as f:
        test_matrix = json.load(f)

    # Find the object to duplicate
    target_object = None
    for obj in test_matrix:
        if obj.get("runs-on") == "n150" and obj.get("name") == "run_jax":
            target_object = obj
            break

    if not target_object:
        print("Target object not found in the JSON file.")
        return

    # Remove the original object
    test_matrix = [obj for obj in test_matrix if obj != target_object]

    # Create new objects based on TEST_GROUP_CNT
    for i in range(1, test_group_cnt + 1):
        new_object = target_object.copy()
        new_object["test_group_cnt"] = test_group_cnt
        new_object["test_group_id"] = i
        test_matrix.append(new_object)

    # Return the modified matrix as JSON
    return json.dumps(test_matrix, indent=4)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python modify_test_matrix.py <file_path> <test_group_cnt>")
        sys.exit(1)

    file_path = sys.argv[1]
    test_group_cnt = int(sys.argv[2])

    modified_matrix = modify_test_matrix(file_path, test_group_cnt)
    print(modified_matrix)
