# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PubChem10M SMILES BPE Masked LM model loader implementation
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
    """Available PubChem10M SMILES BPE Masked LM model variants."""

    PUBCHEM10M_SMILES_BPE_450K = "PubChem10M_SMILES_BPE_450k"


class ModelLoader(ForgeModel):
    """PubChem10M SMILES BPE Masked LM model loader for masked language modeling on chemical SMILES."""

    _VARIANTS = {
        ModelVariant.PUBCHEM10M_SMILES_BPE_450K: ModelConfig(
            pretrained_model_name="seyonec/PubChem10M_SMILES_BPE_450k",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PUBCHEM10M_SMILES_BPE_450K

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="PubChem10M_SMILES_BPE",
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
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        # Sample SMILES string with mask token for chemical masked LM
        test_input = f"CC(=O)OC1=CC=CC=C1C(=O){self.tokenizer.mask_token}"

        inputs = self.tokenizer(test_input, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs):
        if self.tokenizer is None:
            self._load_tokenizer()

        if isinstance(outputs, list):
            logits = outputs[0].logits if hasattr(outputs[0], "logits") else outputs[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

        inputs = self.load_inputs()

        mask_token_index = (inputs["input_ids"] == self.tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]

        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)

        output = self.tokenizer.decode(predicted_token_id)

        return f"Output: {output}"
