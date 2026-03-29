# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Expected Attention Statistics model loader for alessiodevoto/exp_att_stats_MaxJeblick_llama2-0b-unit-test_kmfoda_booksum_100_1000_4.

This model stores pre-computed expected attention statistics derived from running
a tiny LLaMA-2 unit-test model (MaxJeblick/llama2-0b-unit-test) on the
kmfoda/booksum dataset. It uses PyTorchModelHubMixin (no custom model class in
the HF repo), so we load the safetensors weights directly into a simple wrapper.
"""
import json
from typing import Optional

import torch
import torch.nn as nn

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available model variants."""

    EXP_ATT_STATS_LLAMA2_0B = "exp_att_stats_llama2_0b_unit_test"


class ExpectedAttentionStats(nn.Module):
    """Wrapper module for pre-computed attention statistics loaded from safetensors."""

    def __init__(self, num_layers, num_heads, head_dim, state_dict):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        for name, tensor in state_dict.items():
            self.register_buffer(name.replace(".", "_"), tensor)

    def forward(self, x):
        return x


class ModelLoader(ForgeModel):
    """Loader for expected attention statistics model."""

    _VARIANTS = {
        ModelVariant.EXP_ATT_STATS_LLAMA2_0B: ModelConfig(
            pretrained_model_name="alessiodevoto/exp_att_stats_MaxJeblick_llama2-0b-unit-test_kmfoda_booksum_100_1000_4",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EXP_ATT_STATS_LLAMA2_0B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="ExpectedAttentionStats",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.ATOMIC_ML,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        repo_id = self._variant_config.pretrained_model_name

        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        with open(config_path) as f:
            config = json.load(f)

        weights_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        state_dict = load_file(weights_path)

        model = ExpectedAttentionStats(
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            head_dim=config["head_dim"],
            state_dict=state_dict,
        )

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        input_ids = torch.randint(0, 100, (1, 128))
        return {"x": input_ids}
