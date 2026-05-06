# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Demonstrates codegen for Llama-3.2-3B from HuggingFace

import shutil
from pathlib import Path

import torch
import torch_xla.runtime as xr
from transformers import AutoModelForCausalLM
from tt_torch import codegen_py


def main():
    # Set up XLA runtime for TT backend
    xr.set_device_type("TT")

    model_name = "meta-llama/Llama-3.2-3B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, use_cache=False
    )
    model.eval()

    # Small prefill-style input: batch=1, seq_len=32
    input_ids = torch.randint(0, model.config.vocab_size, (1, 32), dtype=torch.long)

    compiler_options = {
        "codegen_split_files": True,
        "export_tensors": True,
        "optimization_level": 1,
    }

    codegen_py(
        model,
        input_ids,
        export_path="llama_3_2_3b_codegen",
        compiler_options=compiler_options,
    )


def test_llama_codegen():
    """Test that codegen for Llama-3.2-3B creates the expected output folder."""
    output_dir = Path("llama_3_2_3b_codegen")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    try:
        main()
        assert (
            output_dir.exists()
        ), f"Expected output folder '{output_dir}' was not created"
        assert output_dir.is_dir(), f"'{output_dir}' exists but is not a directory"
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)


if __name__ == "__main__":
    main()
