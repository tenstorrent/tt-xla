# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SparseGPT Bridges model loader implementation for causal language modeling.

Loads sparse transformer models trained using the bridges procedure from Gao et al. (2025).
These models use weight and activation sparsity with bridge components.
"""
import json
import torch
from huggingface_hub import hf_hub_download
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
from .src.sparse_gpt import SparseGPT


class ModelVariant(StrEnum):
    """Available SparseGPT Bridges model variants."""

    D3072_F0_005 = "D3072_F0_005"


class ModelLoader(ForgeModel):
    """SparseGPT Bridges model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.D3072_F0_005: ModelConfig(
            pretrained_model_name="jacobcd52/ss_bridges_d3072_f0.005",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.D3072_F0_005

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="SparseGPT Bridges",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_config(self):
        """Download and parse the model config from HuggingFace."""
        config_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="config.json",
        )
        with open(config_path, "r") as f:
            full_config = json.load(f)
        return full_config["model_config"]

    def load_model(self, *, dtype_override=None, **kwargs):
        model_config = self._load_config()
        model = SparseGPT(model_config)

        # Load pretrained sparse model weights
        sparse_model_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="sparse_model.bin",
        )
        state_dict = torch.load(sparse_model_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        model_config = self._load_config()
        vocab_size = model_config["vocab_size"]
        n_ctx = model_config["n_ctx"]

        # Generate random token input within the vocabulary range
        input_ids = torch.randint(0, vocab_size, (batch_size, n_ctx), dtype=torch.long)

        return {"input_ids": input_ids}
