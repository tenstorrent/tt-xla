# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Fish Speech S2 Pro model loader implementation for text-to-speech tasks
"""
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


class ModelVariant(StrEnum):
    """Available Fish Speech S2 Pro model variants."""

    S2_PRO = "S2_Pro"


class ModelLoader(ForgeModel):
    """Fish Speech S2 Pro model loader implementation for text-to-speech tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.S2_PRO: ModelConfig(
            pretrained_model_name="fishaudio/s2-pro",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.S2_PRO

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="FishSpeechS2Pro",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Fish Speech S2 Pro model."""
        from huggingface_hub import snapshot_download
        from fish_speech.models.text2semantic.llama import BaseTransformer

        pretrained_model_name = self._variant_config.pretrained_model_name

        # Download model files from HuggingFace Hub
        local_path = snapshot_download(repo_id=pretrained_model_name)

        # Load the model (auto-detects DualARTransformer from config)
        model = BaseTransformer.from_pretrained(local_path, load_weights=True)

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Fish Speech S2 Pro model."""
        config = self.model.config

        seq_len = 128
        num_codebooks = config.num_codebooks

        # inp shape: [batch, num_codebooks + 1, seq_len]
        # Channel 0 is text/semantic tokens, channels 1..num_codebooks are audio codebooks
        inp = torch.zeros(1, num_codebooks + 1, seq_len, dtype=torch.long)
        inp[:, 0, :] = torch.randint(0, config.vocab_size, (1, seq_len))
        for i in range(num_codebooks):
            inp[:, i + 1, :] = torch.randint(0, config.codebook_size, (1, seq_len))

        # labels are required by DualARTransformer.forward() for the fast transformer
        # Use same shape as inp; tokens outside semantic range trigger dummy fast path
        labels = inp.clone()

        return {"inp": inp, "labels": labels}
