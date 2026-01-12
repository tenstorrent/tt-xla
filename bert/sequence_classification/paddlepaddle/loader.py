# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BERT PaddlePaddle model loader implementation for sequence classification.
"""

from typing import Optional, List, Union
import random
import numpy as np
import paddle
from paddlenlp.transformers import BertForSequenceClassification, BertTokenizer

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


class ModelVariant(StrEnum):
    """Available BERT model variants for sequence classification (Paddle)."""

    BERT_BASE_UNCASED = "bert-base-uncased"
    BERT_BASE_JAPANESE = "cl-tohoku/bert-base-japanese"
    CHINESE_ROBERTA_BASE = "uer/chinese-roberta-base"


class ModelLoader(ForgeModel):
    """BERT Paddle model loader implementation for sequence classification."""

    # Dictionary of available model variants using structured configs
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
            model="bert-seqcls",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_SEQUENCE_CLASSIFICATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.PADDLE,
        )

    def _get_sample_text(self) -> List[str]:
        """Return a sample sentence based on variant language (matches test inputs_map['...']['sequence'])."""
        model_name = self._variant_config.pretrained_model_name
        if "japanese" in model_name:
            return ["こんにちは、私の犬はかわいいです"]
        if "chinese" in model_name or "roberta" in model_name:
            return ["你好，我的狗很可爱"]
        return ["Hello, my dog is cute"]

    def load_model(self, dtype_override=None):
        """Load Paddle BERT model for sequence classification."""
        model_name = self._variant_config.pretrained_model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        model = BertForSequenceClassification.from_pretrained(model_name, num_classes=2)
        return model

    def load_inputs(
        self,
        dtype_override=None,
    ) -> List[paddle.Tensor]:
        """Prepare sample inputs for BERT sequence classification (Paddle)."""
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)
        sample_text = self._get_sample_text()
        encoded_input = self.tokenizer(
            sample_text,
            return_token_type_ids=True,
            return_position_ids=True,
            return_attention_mask=True,
        )
        inputs = [paddle.to_tensor(value) for value in encoded_input.values()]
        return inputs
