# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Habibi-TTS model loader implementation for text-to-speech tasks.

Habibi-TTS is a dialectal Arabic TTS system built on F5-TTS (flow-matching
with a DiT backbone).  This loader exposes the DiT transformer so that
the forward pass can be compiled and profiled independently of the ODE
sampling loop.
"""
import torch
import torch.nn as nn
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


class HabibiDiTWrapper(nn.Module):
    """Wrapper that exposes the DiT backbone forward pass.

    The full Habibi-TTS pipeline uses an ODE solver to iteratively denoise
    mel-spectrograms.  Each solver step calls the DiT transformer with:
        x   – noisy mel  (batch, seq_len, mel_dim)
        cond – conditioning mel (batch, seq_len, mel_dim)
        text – token ids  (batch, text_len)
        time – diffusion timestep (batch,)
        mask – sequence mask (batch, seq_len)

    This wrapper packages those inputs into a single forward call suitable
    for tracing / compilation.
    """

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, x, cond, text, time, mask):
        return self.transformer(x=x, cond=cond, text=text, time=time, mask=mask)


class ModelVariant(StrEnum):
    """Available Habibi-TTS model variants."""

    UNIFIED = "Unified"


class ModelLoader(ForgeModel):
    """Habibi-TTS model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.UNIFIED: ModelConfig(
            pretrained_model_name="SWivid/Habibi-TTS",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.UNIFIED

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Habibi-TTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from f5_tts.model.backbones.dit import DiT
        from f5_tts.infer.utils_infer import load_model

        model_cfg = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            text_mask_padding=True,
            conv_layers=4,
        )

        model = load_model(
            DiT,
            model_cfg,
            ckpt_path="hf://SWivid/Habibi-TTS/Unified/model_200000.safetensors",
            vocab_file="hf://SWivid/Habibi-TTS/Unified/vocab.txt",
            device="cpu",
        )

        wrapper = HabibiDiTWrapper(model.transformer)
        wrapper.eval()
        return wrapper

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        batch = 1
        seq_len = 64
        mel_dim = 100
        text_len = 32

        x = torch.randn(batch, seq_len, mel_dim, dtype=dtype)
        cond = torch.randn(batch, seq_len, mel_dim, dtype=dtype)
        text = torch.randint(0, 256, (batch, text_len))
        time = torch.tensor([0.5], dtype=dtype)
        mask = torch.ones(batch, seq_len, dtype=torch.bool)

        return x, cond, text, time, mask
