# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BERT PaddlePaddle model loader implementation for masked language modeling.
"""

from typing import Optional, List

import paddle
from paddlenlp.transformers import BertForMaskedLM, BertTokenizer

from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available BERT model variants for masked language modeling (Paddle)."""

    BERT_BASE_UNCASED = "bert-base-uncased"
    BERT_BASE_JAPANESE = "cl-tohoku/bert-base-japanese"
    CHINESE_ROBERTA_BASE = "uer/chinese-roberta-base"


class ModelLoader(ForgeModel):
    """BERT Paddle model loader implementation for masked language modeling."""

    _VARIANTS = {
        ModelVariant.BERT_BASE_UNCASED: LLMModelConfig(
            pretrained_model_name="bert-base-uncased",
        ),
        ModelVariant.BERT_BASE_JAPANESE: LLMModelConfig(
            pretrained_model_name="cl-tohoku/bert-base-japanese",
        ),
        ModelVariant.CHINESE_ROBERTA_BASE: LLMModelConfig(
            pretrained_model_name="uer/chinese-roberta-base",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BERT_BASE_UNCASED

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.tokenizer: Optional[BertTokenizer] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="bert-maskedlm",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.PADDLE,
        )

    def _get_sample_text(self) -> str:
        """Return a sample masked sentence based on variant language."""
        model_name = self._variant_config.pretrained_model_name
        if "japanese" in model_name:
            return ["一つ、[MASK]、三、四"]
        if "chinese" in model_name or "roberta" in model_name:
            return ["一，[MASK]，三，四"]
        return ["One, [MASK], three, four"]

    def _get_max_length(self) -> int:
        return getattr(self._variant_config, "max_length", 128) or 128

    def load_model(self, dtype_override=None):
        """Load Paddle BERT model for masked language modeling."""
        model_name = self._variant_config.pretrained_model_name
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        model = BertForMaskedLM.from_pretrained(model_name)
        return model

    def load_inputs(self, dtype_override=None) -> List[paddle.Tensor]:
        """Prepare sample inputs for BERT masked language modeling (Paddle)."""

        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        sample_text = self._get_sample_text()
        encoded_input = self.tokenizer(
            sample_text,
            return_token_type_ids=True,
            return_position_ids=True,
            return_attention_mask=True,
        )
        self.inputs = [paddle.to_tensor(value) for value in encoded_input.values()]
        return self.inputs

    def decode_output(self, outputs=None):
        """Decode the model output for masked language modeling."""
        if outputs is None or self.inputs is None or self.tokenizer is None:
            return None
        logits = outputs[0]
        input_ids = self.inputs[0]
        mask_token_index = (
            (input_ids == self.tokenizer.mask_token_id)[0]
            .nonzero(as_tuple=True)[0]
            .item()
        )
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1).item()
        return self.tokenizer.decode(predicted_token_id)
