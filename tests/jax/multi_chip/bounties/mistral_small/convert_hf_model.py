# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Convert Mistral HF model to model loadable by NNX model.

This will load the hf model, reshape and transform parameter arrays as needed,
and then save them using orbax checkpointer. It will also write model and
tokenizer configs.
"""

import argparse
import mistral_nnx
from pathlib import Path
from transformers import AutoTokenizer
import shutil


def main():
    parser = argparse.ArgumentParser(
        description="Convert HF Mistral model to NNX model."
    )
    parser.add_argument("HF_MODEL", type=str)
    parser.add_argument("OUTPUT_DIR", type=Path)
    args = parser.parse_args()

    mistral_nnx.convert_hf_model(args.HF_MODEL, args.OUTPUT_DIR)


if __name__ == "__main__":
    main()
