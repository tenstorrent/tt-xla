# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np


def load_tokens(file_path):
    return np.loadtxt(file_path, dtype=int).flatten()


def tokens_match(a, b):
    return np.array_equal(a, b)


def main():
    root = "results"

    files = {
        "HF": os.path.join(root, "hf.txt"),
        "Single Chip": os.path.join(root, "single_chip.txt"),
        "Multi Chip": os.path.join(root, "multi_chip.txt"),
    }

    print("📦 Loading token outputs...")

    outputs = {name: load_tokens(path) for name, path in files.items()}

    pairs = [("HF", "Single Chip"), ("HF", "Multi Chip"), ("Single Chip", "Multi Chip")]
    for a, b in pairs:
        match = tokens_match(outputs[a], outputs[b])
        if match:
            print(f"✅ Tokens match: {a} == {b}")
        else:
            for i in range(len(outputs[a])):
                if outputs[a][i] != outputs[b][i]:
                    print(f"❌ Missmatch at {i}th token.")
                    print(f"Generation lenght: {len(outputs[a])}")
                    break


if __name__ == "__main__":
    main()
