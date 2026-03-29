# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ESM-1v model loader implementation for masked language modeling on protein sequences.
"""
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
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
    """Available ESM-1v model variants."""

    ESM1V_T33_650M_UR90S_1 = "facebook/esm1v_t33_650M_UR90S_1"
    ESM1V_T33_650M_UR90S_2 = "facebook/esm1v_t33_650M_UR90S_2"
    ESM1V_T33_650M_UR90S_3 = "facebook/esm1v_t33_650M_UR90S_3"
    ESM1V_T33_650M_UR90S_4 = "facebook/esm1v_t33_650M_UR90S_4"
    ESM1V_T33_650M_UR90S_5 = "facebook/esm1v_t33_650M_UR90S_5"


class ModelLoader(ForgeModel):
    """ESM-1v model loader implementation for masked language modeling on protein sequences."""

    _VARIANTS = {
        ModelVariant.ESM1V_T33_650M_UR90S_1: ModelConfig(
            pretrained_model_name="facebook/esm1v_t33_650M_UR90S_1",
        ),
        ModelVariant.ESM1V_T33_650M_UR90S_2: ModelConfig(
            pretrained_model_name="facebook/esm1v_t33_650M_UR90S_2",
        ),
        ModelVariant.ESM1V_T33_650M_UR90S_3: ModelConfig(
            pretrained_model_name="facebook/esm1v_t33_650M_UR90S_3",
        ),
        ModelVariant.ESM1V_T33_650M_UR90S_4: ModelConfig(
            pretrained_model_name="facebook/esm1v_t33_650M_UR90S_4",
        ),
        ModelVariant.ESM1V_T33_650M_UR90S_5: ModelConfig(
            pretrained_model_name="facebook/esm1v_t33_650M_UR90S_5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ESM1V_T33_650M_UR90S_3

    # Short protein sequence for testing
    sample_sequence = "MGSSHHHHHHSSGLVPRGSHM"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ESM1v",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
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

        model = AutoModelForMaskedLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

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
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        mask_token_index = (inputs["input_ids"] == self.tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        predicted_tokens = self.tokenizer.decode(predicted_token_id)

        return predicted_tokens
