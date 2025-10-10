# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from .test_config_inference_data_parallel import (
    test_config as test_config_inference_data_parallel,
)
from .test_config_inference_single_device import (
    PLACEHOLDER_MODELS as PLACEHOLDER_MODELS_INFERENCE,
)
from .test_config_inference_single_device import (
    test_config as test_config_inference_single_device,
)
from .test_config_inference_tensor_parallel import (
    test_config as test_config_inference_tensor_parallel,
)
from .test_config_training_single_device import (
    test_config as test_config_training_single_device,
)

PLACEHOLDER_MODELS = PLACEHOLDER_MODELS_INFERENCE

test_config = (
    test_config_inference_single_device
    | test_config_inference_tensor_parallel
    | test_config_inference_data_parallel
    | test_config_training_single_device
)
