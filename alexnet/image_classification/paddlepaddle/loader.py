# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AlexNet PaddlePaddle model loader implementation.
"""

from typing import Optional

import paddle
from paddle.vision.models import alexnet

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
    """Available AlexNet model variants (Paddle)."""

    DEFAULT = "alexnet"


class ModelLoader(ForgeModel):
    """AlexNet PaddlePaddle model loader implementation."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="alexnet",
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
            model="alexnet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.CUSTOM,
            framework=Framework.PADDLE,
        )

    def load_model(self, dtype_override=None):
        """Load pretrained AlexNet model (Paddle)."""
        model = alexnet(pretrained=True)
        return model

    def load_inputs(self, dtype_override=None, batch_size: int = 1):
        """Prepare sample input for AlexNet model (Paddle)."""
        inputs = paddle.rand([batch_size, 3, 224, 224])
        return [inputs]

    def print_results(self, compiled_model=None, inputs=None):
        """Print results for AlexNet model (Paddle)."""
        compiled_model_out = compiled_model(inputs)
        print_compiled_model_results(compiled_model_out)
