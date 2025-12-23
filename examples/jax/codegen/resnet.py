# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Demonstrates codegen for ResNet-50 from HuggingFace

import jax
from transformers import FlaxResNetForImageClassification
from tt_jax import codegen_py

with jax.default_device(jax.devices("cpu")[0]):
    # Load ResNet-50 from HuggingFace
    model = FlaxResNetForImageClassification.from_pretrained("microsoft/resnet-50")

    # Create input tensor
    key = jax.random.key(0)
    x = jax.random.normal(key, (1, 224, 224, 3))


def forward(params, x):
    return model(pixel_values=x, params=params)


codegen_py(forward, model.params, x, export_path="resnet50_codegen")
