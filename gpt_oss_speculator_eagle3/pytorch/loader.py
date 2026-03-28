# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GPT-OSS 20B EAGLE3 speculator model loader implementation for speculative decoding.
"""

import torch
from transformers import AutoModel
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available GPT-OSS EAGLE3 speculator model variants."""

    GPT_OSS_20B_EAGLE3 = "20B_Eagle3"


class ModelLoader(ForgeModel):
    """GPT-OSS 20B EAGLE3 speculator model loader for speculative decoding.

    Loads the RedHatAI GPT-OSS-20B EAGLE3 speculator draft model, which accelerates
    inference of the openai/gpt-oss-20b verifier model via speculative decoding.
    """

    _VARIANTS = {
        ModelVariant.GPT_OSS_20B_EAGLE3: ModelConfig(
            pretrained_model_name="RedHatAI/gpt-oss-20b-speculator.eagle3",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GPT_OSS_20B_EAGLE3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="GPT-OSS Speculator EAGLE3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the GPT-OSS 20B EAGLE3 speculator model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The EAGLE3 speculator model instance.
        """
        cfg = self._variant_config

        model_kwargs = {"trust_remote_code": True}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            cfg.pretrained_model_name,
            **model_kwargs,
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample inputs for the EAGLE3 speculator model.

        The speculator takes hidden states from the verifier model (gpt-oss-20b)
        and input token IDs.

        Args:
            dtype_override: Optional torch.dtype to override input dtype.

        Returns:
            dict: Input tensors containing hidden states and input IDs.
        """
        dtype = dtype_override or torch.bfloat16
        hidden_size = 2880  # gpt-oss-20b hidden size
        seq_len = 1

        torch.manual_seed(42)
        hidden_states = torch.randn(1, seq_len, hidden_size, dtype=dtype)
        input_ids = torch.randint(0, 201088, (1, seq_len))

        return {"hidden_states": hidden_states, "input_ids": input_ids}
