# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Demonstrates codegen for ResNet-50 from HuggingFace

import jax
from transformers import FlaxResNetForImageClassification

with jax.default_device(jax.devices("cpu")[0]):
    # Load ResNet-50 from HuggingFace
    model = FlaxResNetForImageClassification.from_pretrained("microsoft/resnet-50")

    # Create input tensor
    key = jax.random.key(0)
    x = jax.random.normal(key, (1, 224, 224, 3))


def forward(params, x):
    return model(pixel_values=x, params=params)


compiler_options = {
    "backend": "codegen_py",
    "export_path": "resnet50_codegen",
    # "enable_optimizer": True,
    # "enable_memory_layout_analysis": True,
    # "enable_l1_interleaved": False,
    # "experimental_enable_fusing_conv2d_with_multiply_pattern": True,
}

# Compile the model. Make sure to pass the code generation options.
fun = jax.jit(
    forward,
    compiler_options=compiler_options,
)

# Run the model. This triggers code generation.
fun(model.params, x)
