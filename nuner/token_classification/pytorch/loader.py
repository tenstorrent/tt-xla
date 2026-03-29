# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NuNER model loader implementation for token classification.
"""

import torch
from transformers import AutoModel, AutoTokenizer
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
    """Available NuNER model variants."""

    NUMIND_NUNER_V0_1 = "numind/NuNER-v0.1"


class ModelLoader(ForgeModel):
    """NuNER model loader implementation for token classification."""

    _VARIANTS = {
        ModelVariant.NUMIND_NUNER_V0_1: ModelConfig(
            pretrained_model_name="numind/NuNER-v0.1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NUMIND_NUNER_V0_1

    def __init__(self, variant=None):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self.sample_text = "NuMind is an AI company based in Paris and USA."
        self.max_length = 128

    @classmethod
    def _get_model_info(cls, variant_name=None):
        if variant_name is None:
            variant_name = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="NuNER",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        model_kwargs = {"output_hidden_states": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        inputs = self.load_inputs()
        # co_out[0] is last_hidden_state: token-level embeddings
        last_hidden_state = co_out[0]

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        active_tokens = [
            t
            for t, m in zip(tokens, inputs["attention_mask"][0])
            if m == 1 and t not in ("<s>", "</s>", "<pad>")
        ]

        print(f"Context: {self.sample_text}")
        print(f"Embedding shape: {last_hidden_state.shape}")
        print(f"Active tokens: {active_tokens}")
