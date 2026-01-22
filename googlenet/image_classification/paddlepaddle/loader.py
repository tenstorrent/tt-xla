# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GoogleNet PaddlePaddle model loader implementation.
"""

from typing import Optional

import paddle
from paddle.vision.models import googlenet

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from ....tools.utils import print_compiled_model_results


class ModelVariant(StrEnum):
    """Available GoogleNet model variants (Paddle)."""

    DEFAULT = "googlenet"


class ModelLoader(ForgeModel):
    """GoogleNet PaddlePaddle model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="googlenet",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DEFAULT

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="googlenet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.CUSTOM,
            framework=Framework.PADDLE,
        )

    def load_model(self, dtype_override=None):
        """Load pretrained GoogleNet model (Paddle)."""
        model = googlenet(pretrained=True)
        return model

    def load_inputs(self, dtype_override=None, batch_size: int = 1):
        """Prepare sample input for GoogleNet model (Paddle)."""
        inputs = paddle.rand([batch_size, 3, 224, 224])
        return [inputs]

    def print_results(self, compiled_model=None, inputs=None):
        """Print results for GoogleNet model (Paddle)."""
        compiled_model_out = compiled_model(inputs)
        print_compiled_model_results(compiled_model_out)
