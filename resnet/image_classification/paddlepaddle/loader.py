# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ResNet PaddlePaddle model loader implementation.
"""

from typing import Optional

import paddle
from paddle.vision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)

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
    """Available ResNet model variants (Paddle)."""

    RESNET18 = "resnet18"
    RESNET34 = "resnet34"
    RESNET50 = "resnet50"
    RESNET101 = "resnet101"
    RESNET152 = "resnet152"


class ModelLoader(ForgeModel):
    """ResNet PaddlePaddle model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.RESNET18: ModelConfig(
            pretrained_model_name="resnet18",
        ),
        ModelVariant.RESNET34: ModelConfig(
            pretrained_model_name="resnet34",
        ),
        ModelVariant.RESNET50: ModelConfig(
            pretrained_model_name="resnet50",
        ),
        ModelVariant.RESNET101: ModelConfig(
            pretrained_model_name="resnet101",
        ),
        ModelVariant.RESNET152: ModelConfig(
            pretrained_model_name="resnet152",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.RESNET18

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        return ModelInfo(
            model="resnet",
            variant=variant.value,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.CUSTOM,
            framework=Framework.PADDLE,
        )

    def load_model(self, dtype_override=None):
        """Load pretrained ResNet model for this instance's variant (Paddle)."""
        variant = self._variant
        if variant == ModelVariant.RESNET18:
            model = resnet18(pretrained=True)
        elif variant == ModelVariant.RESNET34:
            model = resnet34(pretrained=True)
        elif variant == ModelVariant.RESNET50:
            model = resnet50(pretrained=True)
        elif variant == ModelVariant.RESNET101:
            model = resnet101(pretrained=True)
        elif variant == ModelVariant.RESNET152:
            model = resnet152(pretrained=True)
        else:
            raise ValueError(f"Unsupported variant: {variant}")
        return model

    def load_inputs(self, dtype_override=None, batch_size: int = 1):
        """Prepare sample input for ResNet model (Paddle)."""
        inputs = paddle.rand([batch_size, 3, 224, 224])
        return [inputs]

    def print_results(self, compiled_model=None, inputs=None):
        """Print results for ResNet model (Paddle)."""
        compiled_model_out = compiled_model(inputs)
        print_compiled_model_results(compiled_model_out)
