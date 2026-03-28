# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Zyphra ZUNA EEG Foundation Model loader implementation.

ZUNA is a 380M-parameter masked diffusion autoencoder for EEG signal
processing, supporting denoising, reconstruction, and channel prediction.

Requires the ZUNA repository to be cloned at /tmp/zuna_repo.
"""
import json
import os
import sys

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
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

ZUNA_REPO_PATH = "/tmp/zuna_repo"

REPO_ID = "Zyphra/ZUNA"
WEIGHTS_FILE = "model-00001-of-00001.safetensors"
CONFIG_FILE = "config.json"


def _ensure_zuna_importable():
    """Ensure the ZUNA repo is cloned and importable."""
    if not os.path.isdir(ZUNA_REPO_PATH):
        import subprocess

        subprocess.check_call(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "https://github.com/Zyphra/zuna.git",
                ZUNA_REPO_PATH,
            ]
        )

    lingua_path = os.path.join(
        ZUNA_REPO_PATH, "src", "zuna", "inference", "AY2l", "lingua"
    )
    if lingua_path not in sys.path:
        sys.path.insert(0, lingua_path)


class ModelVariant(StrEnum):
    """Available ZUNA model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """ZUNA EEG Foundation Model loader."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="Zyphra/ZUNA",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    # Model dimensions from config.json
    _INPUT_DIM = 32
    _MAX_SEQLEN = 50
    _ROPE_DIM = 4

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="ZUNA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ZUNA EncoderDecoder model.

        Returns:
            torch.nn.Module: The ZUNA masked diffusion autoencoder.
        """
        _ensure_zuna_importable()
        from apps.AY2latent_bci.transformer import EncoderDecoder
        from lingua.args import dataclass_from_dict

        # Load config from HuggingFace
        config_path = hf_hub_download(repo_id=REPO_ID, filename=CONFIG_FILE)
        with open(config_path, "r") as f:
            config = json.load(f)

        # Import and construct args dataclass
        from apps.AY2latent_bci.transformer import DecoderTransformerArgs

        model_args = dataclass_from_dict(DecoderTransformerArgs, config["model"])

        # Load weights
        weights_path = hf_hub_download(
            repo_id=REPO_ID, filename=WEIGHTS_FILE, token=False
        )
        state_dict_raw = load_file(weights_path)
        state_dict = {k.removeprefix("model."): v for k, v in state_dict_raw.items()}

        # Construct and load model
        model = EncoderDecoder(model_args)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load synthetic inputs for the ZUNA model forward pass.

        Returns:
            dict: Input tensors for the EncoderDecoder forward method.
        """
        dtype = dtype_override or torch.float32
        seqlen = self._MAX_SEQLEN

        # encoder_input: EEG signal input [B, seqlen, input_dim]
        encoder_input = torch.randn(batch_size, seqlen, self._INPUT_DIM, dtype=dtype)

        # decoder_input: noisy version of signal [B, seqlen, input_dim]
        decoder_input = torch.randn(batch_size, seqlen, self._INPUT_DIM, dtype=dtype)

        # t: diffusion timestep [B]
        t = torch.full((batch_size,), 0.5, dtype=dtype)

        # chan_pos: 3D channel positions on scalp [B, seqlen, 3]
        chan_pos = torch.randn(batch_size, seqlen, 3, dtype=dtype)

        # chan_pos_discrete: discretized channel positions [B, seqlen, 3]
        chan_pos_discrete = torch.randint(0, 100, (batch_size, seqlen, 3))

        # chan_id: channel identifiers [B, seqlen]
        chan_id = torch.randint(0, 32, (batch_size, seqlen))

        # t_coarse: coarse time indices [B, seqlen]
        t_coarse = torch.arange(seqlen).unsqueeze(0).expand(batch_size, -1)

        # seq_lens: sequence lengths [B]
        seq_lens = torch.full((batch_size,), seqlen, dtype=torch.long)

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "t": t,
            "chan_pos": chan_pos,
            "chan_pos_discrete": chan_pos_discrete,
            "chan_id": chan_id,
            "t_coarse": t_coarse,
            "seq_lens": seq_lens,
        }
