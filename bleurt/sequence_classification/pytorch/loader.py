# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BLEURT-20 model loader implementation for sequence classification.

BLEURT is a learned evaluation metric for natural language generation that scores
how well a candidate sentence matches a reference sentence, producing a regression
score where values closer to 1.0 indicate a better match.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available BLEURT model variants for sequence classification."""

    BLEURT_20 = "BLEURT-20"


class ModelLoader(ForgeModel):
    """BLEURT-20 model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.BLEURT_20: ModelConfig(
            pretrained_model_name="lucadiliello/BLEURT-20",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BLEURT_20

    # Sample reference-candidate pairs for testing
    sample_pairs = [
        ("a bird chirps by the window", "a bird chirps by the window"),
        ("a bird chirps by the window", "this looks like a random sentence"),
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="BLEURT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False, "trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        references = [pair[0] for pair in self.sample_pairs]
        candidates = [pair[1] for pair in self.sample_pairs]

        inputs = self.tokenizer(
            references,
            candidates,
            padding="longest",
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype == torch.float32:
                        inputs[key] = value.to(dtype_override)

        return inputs
