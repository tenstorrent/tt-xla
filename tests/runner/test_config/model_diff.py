# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import sys

import yaml

new_file = sys.argv[1]
old_file = sys.argv[2]


# Load YAML files
def load_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


# Load both files
try:
    new_data = load_yaml(new_file)
    old_data = load_yaml(old_file)
except FileNotFoundError as e:
    print(f"Error: File not found - {e}", file=sys.stderr)
    sys.exit(1)
except yaml.YAMLError as e:
    print(f"Error: Failed to parse YAML - {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)

push_changed = False

if "test_config" in old_data and "test_config" in new_data:
    old_dict = old_data["test_config"]
    new_dict = new_data["test_config"]
    all_keys = set(old_dict.keys()) | set(new_dict.keys())
    for key in all_keys:
        if key not in old_dict:
            print(f"ADDED: {key}", file=sys.stderr)
            print(f"  Value: {new_dict[key]}", file=sys.stderr)
            if "push" in new_dict[key].get("markers", []):
                print(
                    "  Note: This test is marked as 'push', which may indicate it is a new test added in this PR.",
                    file=sys.stderr,
                )
                push_changed = True
        elif key not in new_dict:
            print(f"REMOVED: {key}", file=sys.stderr)
            print(f"  Value: {old_dict[key]}", file=sys.stderr)
        elif old_dict[key] != new_dict[key]:
            print(f"MODIFIED: {key}", file=sys.stderr)
            print(f"  Old: {old_dict[key]}", file=sys.stderr)
            print(f"  New: {new_dict[key]}", file=sys.stderr)
            if "push" in new_dict[key].get("markers", []):
                print(
                    "  Note: This test is marked as 'push', which may indicate it is a new test added in this PR.",
                    file=sys.stderr,
                )
                push_changed = True
else:
    print("test_config section missing in one of the files", file=sys.stderr)
    sys.exit(1)

if push_changed:
    print("1")
else:
    print("0")
