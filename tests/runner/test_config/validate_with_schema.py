# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Validate YAML files against the JSON Schema.

Usage: python3 validate_with_schema.py <file> <schema>
"""

import json
import sys
from pathlib import Path

import yaml
from jsonschema import ValidationError, validate


def validate_yaml_file(yaml_file: Path, schema: dict) -> bool:
    """Validate a YAML file against the schema."""

    try:
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)

        if not data:
            print(f"Empty YAML file {yaml_file}")
            return False
    except yaml.YAMLError as e:
        print(f"YAML parsing error: {e}")
        return False
    except Exception as e:
        print(f"Error reading file: {e}")
        return False

    # Validate against schema
    try:
        validate(instance=data, schema=schema)
        return True
    except ValidationError as e:
        # Format the error message
        path = " -> ".join(str(p) for p in e.path) if e.path else "root"
        print(f"Validation error at {path}: {e.message}")
        return False
    except Exception as e:
        print(f"Schema validation error: {e}")
        return False


if len(sys.argv) != 3:
    print("Usage: python3 validate_with_schema.py <file> <schema>")
    sys.exit(1)

fn = sys.argv[1]
schema_fn = sys.argv[2]

# Load schema
try:
    with open(schema_fn, "r") as f:
        schema = json.load(f)
except Exception as e:
    print(f"ERROR: Failed to load schema from {schema_fn}: {e}")
    sys.exit(1)

# Determine files to validate
print(f"Validating {fn}:", end=" ")
if validate_yaml_file(fn, schema):
    print("success")
    sys.exit(0)
else:
    print("FAIL")
    sys.exit(1)
