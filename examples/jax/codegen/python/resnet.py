# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Demonstrates codegen for ResNet-50 from HuggingFace

import shutil
from pathlib import Path

import jax
from transformers.models.resnet.modeling_flax_resnet import FlaxResNetForImageClassification
from tt_jax import codegen_py


def main():
    with jax.default_device(jax.devices("cpu")[0]):
        # Load ResNet-50 from HuggingFace
        model = FlaxResNetForImageClassification.from_pretrained("microsoft/resnet-50")

        # Create input tensor in NCHW format (FlaxResNet expects NCHW and transposes internally)
        key = jax.random.key(0)
        x = jax.random.normal(key, (1, 3, 224, 224))

    def forward(params, x):
        return model(pixel_values=x, params=params)

    codegen_py(forward, model.params, x, export_path="resnet50_codegen")


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
