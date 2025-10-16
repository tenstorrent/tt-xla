# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from .test_config_inference_single_device import (
    test_config as test_config_inference_single_device,
)

# Empty placeholder models dictionary since all JAX models are discovered from tt-forge-models
PLACEHOLDER_MODELS = {}

# For now we only have inference single device config
# Add more configs as they are created (e.g., training, tensor_parallel)
test_config = test_config_inference_single_device
