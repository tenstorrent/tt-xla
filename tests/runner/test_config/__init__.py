# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from .test_config_inference import PLACEHOLDER_MODELS as PLACEHOLDER_MODELS_INFERENCE
from .test_config_inference import test_config as test_config_inference
from .test_config_training import test_config as test_config_training

PLACEHOLDER_MODELS = PLACEHOLDER_MODELS_INFERENCE

test_config = test_config_inference | test_config_training
