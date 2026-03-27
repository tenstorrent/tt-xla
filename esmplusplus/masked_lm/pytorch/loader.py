# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ESM++ model loader implementation for masked language modeling on protein sequences.
"""
from transformers import AutoModelForMaskedLM
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
    """Available ESM++ model variants."""

    ESMPLUSPLUS_LARGE = "Synthyra/ESMplusplus_large"


class ModelLoader(ForgeModel):
    """ESM++ model loader implementation for masked language modeling on protein sequences."""

    _VARIANTS = {
        ModelVariant.ESMPLUSPLUS_LARGE: ModelConfig(
            pretrained_model_name="Synthyra/ESMplusplus_large",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ESMPLUSPLUS_LARGE

    # Short protein sequence for testing
    sample_sequence = "MGSSHHHHHHSSGLVPRGSHMASK"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ESMplusplus",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )

        # ESM++ exposes its tokenizer via the model object
        self.tokenizer = model.tokenizer

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model()

        # Use a protein sequence with <mask> token for masked LM task
        masked_sequence = "MGSSHHHHHHSSGLVPRGSHM<mask>GSSHHHHHHSSGLVPRGSHM"

        inputs = self.tokenizer(
            masked_sequence,
            return_tensors="pt",
            add_special_tokens=True,
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        if self.tokenizer is None:
            self.load_model()

        if inputs is None:
            inputs = self.load_inputs()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        mask_token_index = (inputs["input_ids"] == self.tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        predicted_tokens = self.tokenizer.decode(predicted_token_id)

        return predicted_tokens
