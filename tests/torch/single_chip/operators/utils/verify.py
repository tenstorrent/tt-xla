# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Verify utilities

import torch

from typing import List
from loguru import logger

from .datatypes import VerifyConfig

from .features import TestSweepsFeatures

from .frontend import SweepsFrontend

from .compat import (
    create_torch_inputs,
    verify_module_for_inputs,
    TestTensorsUtils,
)


class VerifyUtils:
    """Utility functions for Forge verification"""

    @classmethod
    def verify(cls, verify_config: VerifyConfig):
        if verify_config.inputs is None:
            inputs = cls.create_torch_inputs(verify_config=verify_config)
            verify_config.inputs = inputs
        cls.prepare(verify_config=verify_config)
        cls.verify_module_for_inputs(verify_config=verify_config)

    @classmethod
    def prepare(cls, verify_config: VerifyConfig):

        if verify_config.model_dtype:
            # Transfer model to model_dtype if specified
            verify_config.model.to(verify_config.model_dtype)

        if verify_config.skip_forge_verification is None:
            verify_config.skip_forge_verification = TestSweepsFeatures.params.skip_forge_verification

        # Conclude if we should convert to forge data format/
        if verify_config.convert_to_forge is None and SweepsFrontend.is_forge_module(verify_config.model):
            verify_config.convert_to_forge = True

    @classmethod
    def create_torch_inputs(cls, verify_config: VerifyConfig) -> List[torch.Tensor]:

        inputs = create_torch_inputs(
            input_shapes=verify_config.input_shapes,
            dev_data_format=verify_config.dev_data_format,
            value_range=verify_config.value_range,
            random_seed=verify_config.random_seed,
        )

        return inputs

    @classmethod
    def verify_module_for_inputs(cls, verify_config: VerifyConfig):
        if verify_config.convert_to_forge:
            verify_config.inputs = TestTensorsUtils.convert_to_forge_tensors(
                verify_config.inputs, verify_config.dev_data_format
            )
        if verify_config.skip_forge_verification:
            if SweepsFrontend.is_forge_module(verify_config.model):
                logger.warning("Nothing to validate while skipping Forge verification for Forge module")
            else:
                SweepsFrontend.verify_torch(verify_config=verify_config)
        else:
            verify_module_for_inputs(verify_config=verify_config)
