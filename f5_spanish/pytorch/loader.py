# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
F5-Spanish TTS model loader implementation.

F5-Spanish is a zero-shot voice cloning TTS model for Spanish,
fine-tuned from SWivid/F5-TTS using a DiT-based flow matching architecture.
"""
import torch
import torch.nn as nn
from typing import Optional

from huggingface_hub import hf_hub_download

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


class F5DiTWrapper(nn.Module):
    """Wrapper around the F5-TTS DiT transformer backbone.

    Wraps the DiT model to provide a clean forward signature by
    fixing the classifier-free guidance flags to inference mode.
    """

    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(self, x, cond, text, time, mask):
        return self.dit(
            x=x,
            cond=cond,
            text=text,
            time=time,
            drop_audio_cond=False,
            drop_text=False,
            mask=mask,
        )


class ModelVariant(StrEnum):
    """Available F5-Spanish model variants."""

    F5_SPANISH = "spanish"


class ModelLoader(ForgeModel):
    """F5-Spanish TTS model loader implementation."""

    _VARIANTS = {
        ModelVariant.F5_SPANISH: ModelConfig(
            pretrained_model_name="jpgallegoar/F5-Spanish",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.F5_SPANISH

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="F5-Spanish",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from f5_tts.model import DiT
        from f5_tts.model.utils import load_checkpoint

        repo_id = self._variant_config.pretrained_model_name

        vocab_path = hf_hub_download(repo_id=repo_id, filename="vocab.txt")
        with open(vocab_path) as f:
            vocab_char_map = {}
            for i, char in enumerate(f.read()):
                vocab_char_map[char] = i
        vocab_size = len(vocab_char_map)

        model = DiT(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4,
            text_num_embeds=vocab_size,
            mel_dim=100,
        )

        ckpt_path = hf_hub_download(
            repo_id=repo_id,
            filename="model_1200000.safetensors",
        )
        load_checkpoint(model, ckpt_path, device="cpu", use_ema=True)

        wrapper = F5DiTWrapper(model)
        wrapper.eval()

        if dtype_override is not None:
            wrapper = wrapper.to(dtype=dtype_override)

        return wrapper

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        seq_len = 64
        mel_dim = 100
        text_len = 32

        x = torch.randn(1, seq_len, mel_dim, dtype=dtype)
        cond = torch.randn(1, seq_len, mel_dim, dtype=dtype)
        text = torch.randint(0, 100, (1, text_len))
        time = torch.tensor([0.5], dtype=dtype)
        mask = torch.ones(1, seq_len, dtype=torch.bool)

        return x, cond, text, time, mask
