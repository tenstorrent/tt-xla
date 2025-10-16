# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from .test_config_inference_single_device import (
    test_config as test_config_inference_single_device,
)
from .test_config_inference_tensor_parallel import (
    test_config as test_config_inference_tensor_parallel,
)

# Empty placeholder models dictionary since all JAX models are discovered from tt-forge-models
PLACEHOLDER_MODELS = {}

# Merge all test configs
test_config = (
    test_config_inference_single_device
    | test_config_inference_tensor_parallel
)
