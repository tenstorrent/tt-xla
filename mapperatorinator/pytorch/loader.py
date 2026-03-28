# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mapperatorinator model loader implementation.

OliBomby/Mapperatorinator-v30 is a custom Whisper-based encoder-decoder model
that generates osu! beatmaps from audio spectrograms. The model architecture
is defined in the external GitHub repository and requires cloning at runtime.
"""
import os
import subprocess
import sys
from typing import Optional

import torch
import torch.nn as nn

# Patch is_flash_attn_greater_or_equal_2_10 which was removed in newer
# transformers but is imported by the RoPEWhisper modeling code.
import transformers.utils

if not hasattr(transformers.utils, "is_flash_attn_greater_or_equal_2_10"):

    def _is_flash_attn_gte_2_10():
        return False

    transformers.utils.is_flash_attn_greater_or_equal_2_10 = _is_flash_attn_gte_2_10
    sys.modules["transformers.utils"].__dict__[
        "is_flash_attn_greater_or_equal_2_10"
    ] = _is_flash_attn_gte_2_10

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

REPO_URL = "https://github.com/OliBomby/Mapperatorinator.git"
CACHE_DIR = os.path.join(
    os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
    "mapperatorinator_src",
)


def _ensure_repo_cloned():
    """Clone the Mapperatorinator repo if not already cached."""
    if not os.path.isdir(os.path.join(CACHE_DIR, ".git")):
        subprocess.run(
            ["git", "clone", "--depth", "1", REPO_URL, CACHE_DIR],
            check=True,
            capture_output=True,
        )
    model_path = os.path.join(CACHE_DIR, "osuT5")
    if model_path not in sys.path:
        sys.path.insert(0, model_path)


class MapperatorinatorWrapper(nn.Module):
    """Wrapper that provides a clean forward pass for the Mapperatorinator model.

    The full model forward accepts many optional conditioning inputs (difficulty,
    mapper style, song position). This wrapper bundles synthetic conditioning so
    the model can be called with just audio frames and decoder token ids.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, frames, decoder_input_ids, difficulty, mapper_idx, song_position):
        return self.model(
            frames=frames,
            decoder_input_ids=decoder_input_ids,
            difficulty=difficulty,
            mapper_idx=mapper_idx,
            song_position=song_position,
        )


class ModelVariant(StrEnum):
    """Available Mapperatorinator model variants."""

    MAPPERATORINATOR_V30 = "v30"


class ModelLoader(ForgeModel):
    """Mapperatorinator model loader implementation."""

    _VARIANTS = {
        ModelVariant.MAPPERATORINATOR_V30: ModelConfig(
            pretrained_model_name="OliBomby/Mapperatorinator-v30",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MAPPERATORINATOR_V30

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Mapperatorinator",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        _ensure_repo_cloned()

        from osuT5.model.configuration_mapperatorinator import MapperatorinatorConfig
        from osuT5.model.modeling_mapperatorinator import Mapperatorinator
        from transformers import AutoConfig, AutoModel

        AutoConfig.register("mapperatorinator", MapperatorinatorConfig)
        AutoModel.register(MapperatorinatorConfig, Mapperatorinator)

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {
            "attn_implementation": "eager",
        }
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return MapperatorinatorWrapper(model)

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        batch_size = 1
        # 1 second of 16kHz audio
        n_samples = 16000
        frames = torch.randn(batch_size, n_samples, dtype=dtype)
        # Short decoder sequence with tokens in valid range (vocab_size_in=8339)
        decoder_input_ids = torch.randint(0, 100, (batch_size, 8), dtype=torch.long)
        # Conditioning inputs
        difficulty = torch.tensor([5.0], dtype=dtype)
        mapper_idx = torch.tensor([0], dtype=torch.long)
        song_position = torch.tensor([[0.0, 0.1]], dtype=dtype)
        return (frames, decoder_input_ids, difficulty, mapper_idx, song_position)
