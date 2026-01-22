# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ALBERT PaddlePaddle model loader implementation for masked language modeling.
"""

from typing import Optional, List

import paddle
from paddlenlp.transformers import AlbertForMaskedLM, AlbertTokenizer

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel
from pytorch.loader import ModelLoader as PyTorchModelLoader


class ModelVariant(StrEnum):
    """Available ALBERT model variants for masked language modeling (Paddle)."""

    ALBERT_CHINESE_TINY = "albert-chinese-tiny"


class ModelLoader(ForgeModel):
    """ALBERT Paddle model loader implementation for masked language modeling."""

    _VARIANTS = {
        ModelVariant.ALBERT_CHINESE_TINY: LLMModelConfig(
            pretrained_model_name="albert-chinese-tiny",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.ALBERT_CHINESE_TINY

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.tokenizer: Optional[AlbertTokenizer] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="albert-maskedlm",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.PADDLE,
        )

    def _get_sample_text(self) -> str:
        """Return the sample masked sentence used by the test."""
        return ["一，[MASK]，三，四"]

    def load_model(self, dtype_override=None):
        """Load Paddle ALBERT model for masked language modeling."""
        model_name = self._variant_config.pretrained_model_name
        # Initialize tokenizer
        self.tokenizer = AlbertTokenizer.from_pretrained(model_name)

        base_model = AlbertForMaskedLM.from_pretrained(model_name)

        class AlbertWrapper(paddle.nn.Layer):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids, token_type_ids, position_ids, attention_mask):
                return self.model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                )

        wrapped = AlbertWrapper(base_model)
        return wrapped

    def load_inputs(self, dtype_override=None) -> List[paddle.Tensor]:
        """Prepare sample inputs for ALBERT masked language modeling (Paddle)."""
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        sample_text = self._get_sample_text()
        encoded = self.tokenizer(
            sample_text,
            return_token_type_ids=True,
            return_position_ids=True,
            return_attention_mask=True,
        )
        inputs = [paddle.to_tensor(value) for value in encoded.values()]
        return inputs

    def decode_output(self, outputs, inputs=None):
        """Decode the model output for masked language modeling."""
        if outputs is None or self.tokenizer is None:
            return None
        pytorch_model_loader = PyTorchModelLoader(self.DEFAULT_VARIANT)
        pytorch_model_loader.decode_output(outputs, inputs)
