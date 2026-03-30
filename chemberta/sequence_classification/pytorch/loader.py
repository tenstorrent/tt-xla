# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ChemBERTa model loader implementation for sequence classification (regression).

ChemBERTa-5M-MTR is a small RoBERTa-based model trained for multi-task regression
on molecular properties from SMILES strings.

Reference: https://huggingface.co/DeepChem/ChemBERTa-5M-MTR
"""

from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
    """Available ChemBERTa model variants for sequence classification."""

    CHEMBERTA_5M_MTR = "ChemBERTa_5M_MTR"


class ModelLoader(ForgeModel):
    """ChemBERTa model loader implementation for sequence classification (regression)."""

    _VARIANTS = {
        ModelVariant.CHEMBERTA_5M_MTR: ModelConfig(
            pretrained_model_name="DeepChem/ChemBERTa-5M-MTR",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CHEMBERTA_5M_MTR

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="ChemBERTa",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        # SMILES string for ethanol as a sample molecular input
        test_input = "CCO"

        inputs = self.tokenizer(test_input, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs):
        if isinstance(outputs, list):
            logits = outputs[0].logits if hasattr(outputs[0], "logits") else outputs[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

        # Multi-task regression output: show first few predicted property values
        values = logits[0][:5].detach().tolist()
        formatted = ", ".join(f"{v:.4f}" for v in values)

        return (
            f"Output (first 5 of {logits.shape[-1]} regression targets): [{formatted}]"
        )
