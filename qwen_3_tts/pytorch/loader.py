# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-TTS-12Hz-0.6B-Base model loader implementation for text-to-speech tasks.
"""
import sys
import types
import os

import torch
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


def _stub_qwen_tts_deps():
    """Stub heavy qwen_tts dependencies to avoid importing torchaudio/sox."""
    if "qwen_tts" in sys.modules:
        return

    # The qwen_tts top-level __init__ imports torchaudio (via tokenizer_25hz)
    # which requires sox and native libs not always available. Stub the package
    # hierarchy so we can import only config + modeling modules.
    site_pkg = os.path.join(
        sys.prefix,
        "lib",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
        "site-packages",
        "qwen_tts",
    )

    for name, suffix in [
        ("qwen_tts", ""),
        ("qwen_tts.core", "/core"),
        ("qwen_tts.core.models", "/core/models"),
        ("qwen_tts.inference", "/inference"),
    ]:
        m = types.ModuleType(name)
        m.__path__ = [site_pkg + suffix]
        m.__package__ = name
        sys.modules[name] = m

    tok = types.ModuleType("qwen_tts.inference.qwen3_tts_tokenizer")
    tok.Qwen3TTSTokenizer = type(
        "Qwen3TTSTokenizer",
        (),
        {"from_pretrained": classmethod(lambda cls, *a, **kw: None)},
    )
    sys.modules["qwen_tts.inference.qwen3_tts_tokenizer"] = tok


class ModelVariant(StrEnum):
    """Available Qwen3-TTS model variants."""

    TTS_12HZ_0_6B_BASE = "TTS_12Hz_0.6B_Base"


class ModelLoader(ForgeModel):
    """Qwen3-TTS model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.TTS_12HZ_0_6B_BASE: ModelConfig(
            pretrained_model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TTS_12HZ_0_6B_BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qwen3TTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen3-TTS talker model."""
        _stub_qwen_tts_deps()

        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
        from qwen_tts.core.models.modeling_qwen3_tts import (
            Qwen3TTSForConditionalGeneration,
        )
        from transformers import AutoConfig, AutoModel

        pretrained_model_name = self._variant_config.pretrained_model_name

        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)

        full_model = AutoModel.from_pretrained(
            pretrained_model_name,
            dtype=dtype_override if dtype_override is not None else torch.float32,
        )

        full_model.eval()
        self.model = full_model
        return full_model.talker

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Qwen3-TTS talker model."""
        talker_config = self.model.config.talker_config

        seq_len = 128
        hidden_size = talker_config.hidden_size

        # The talker expects inputs_embeds (not input_ids) for prefill mode
        inputs_embeds = torch.randn(1, seq_len, hidden_size)
        attention_mask = torch.ones(1, seq_len, dtype=torch.long)

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
        }
