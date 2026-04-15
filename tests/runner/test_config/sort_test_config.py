#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Sort test entries under test_config by test name."""

import sys
from pathlib import Path

path = Path(sys.argv[1])
lines = path.read_text().splitlines()

# Split into header (everything up to and including "test_config:") and entries
header = []
entries = {}  # key -> list of lines (including the key line)
current_key = None

for line in lines:
    if current_key is None and not line.replace("#", "").startswith("  "):
        header.append(line)
    elif line.replace("#", "").startswith("  ") and not line.replace("#", "").startswith("    "):
        # Level-2 key line
        current_key = line.strip().rstrip(":")
        entries[current_key] = [line]
    elif current_key is not None:
        entries[current_key].append(line)

out = "\n".join(header)
for key in sorted(entries):
    out += "\n" + "\n".join(entries[key])
out += "\n"

path.write_text(out)
print(f"Sorted {len(entries)} entries.")
