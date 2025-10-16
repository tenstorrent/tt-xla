# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from .test_config_inference_data_parallel import (
    test_config as test_config_inference_data_parallel,
)
from .test_config_inference_data_parallel import (
    test_config_common as test_config_inference_data_parallel_common,
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


def _apply_common(common_defaults: dict, cfg: dict) -> dict:
    """Merge common_defaults into each per-test config dict, allowing per-test overrides."""
    merged = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            merged[key] = {**(common_defaults or {}), **value}
        else:
            merged[key] = value
    return merged


test_config = (
    test_config_inference_single_device
    | test_config_inference_tensor_parallel
    | _apply_common(
        test_config_inference_data_parallel_common,
        test_config_inference_data_parallel,
    )
    | test_config_training_single_device
)
