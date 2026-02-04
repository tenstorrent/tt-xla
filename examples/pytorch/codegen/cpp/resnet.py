# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Demonstrates codegen for ResNet-50 from HuggingFace

import shutil
from pathlib import Path

import torch
import torch_xla.runtime as xr
from transformers import ResNetForImageClassification
from tt_torch import codegen_cpp


def main():
    # Set up XLA runtime for TT backend
    xr.set_device_type("TT")

    # Load ResNet-50 from HuggingFace
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    model.eval()
    x = torch.randn(1, 3, 224, 224)

    codegen_cpp(model, x, export_path="resnet50_codegen")


def test_resnet_codegen():
    """Test that codegen for ResNet-50 creates the expected output folder."""
    output_dir = Path("resnet50_codegen")
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
