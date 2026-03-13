# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Testers for Wan pipeline components (text encoder, transformer, VAE)."""

from typing import Any, Dict, Sequence

import torch
from infra import ComparisonConfig, Model, RunMode, TorchModelTester

from third_party.tt_forge_models.wan.pytorch import ModelLoader

# ---------------------------------------------------------------------------
# Wrapper modules — normalise each component's forward to return a single tensor
# ---------------------------------------------------------------------------


class WanTextEncoderWrapper(torch.nn.Module):
    """Wraps UMT5EncoderModel so forward() returns the last_hidden_state tensor."""

    def __init__(self, text_encoder: torch.nn.Module) -> None:
        super().__init__()
        self.text_encoder = text_encoder

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state


class WanTransformerWrapper(torch.nn.Module):
    """Wraps WanTransformer3DModel so forward() returns the denoised sample tensor."""

    def __init__(self, transformer: torch.nn.Module) -> None:
        super().__init__()
        self.transformer = transformer

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        output = self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )
        return output[0]


# ---------------------------------------------------------------------------
# Testers
# ---------------------------------------------------------------------------


class WanVAETester(TorchModelTester):
    """Tester for Wan VAE encoder/decoder."""

    def __init__(
        self,
        variant_name: str,
        vae_part: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
        **kwargs,
    ) -> None:
        if vae_part not in ["decoder", "encoder"]:
            raise ValueError(f"Invalid vae_part: {vae_part}")
        self._vae_part = vae_part
        self._model_loader = ModelLoader(variant_name, subfolder="vae")
        super().__init__(comparison_config, run_mode, **kwargs)

    def _get_model(self) -> Model:
        vae = self._model_loader.load_model()
        return vae.encoder if self._vae_part == "encoder" else vae.decoder

    def _get_input_activations(self) -> Dict | Sequence[Any]:
        return self._model_loader.load_inputs(vae_type=self._vae_part)


class WanTextEncoderTester(TorchModelTester):
    """Tester for Wan UMT5-XXL text encoder."""

    def __init__(
        self,
        variant_name: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
        **kwargs,
    ) -> None:
        self._model_loader = ModelLoader(variant_name, subfolder="text_encoder")
        super().__init__(comparison_config, run_mode, **kwargs)

    def _get_model(self) -> Model:
        text_encoder = self._model_loader.load_model()
        return WanTextEncoderWrapper(text_encoder)

    def _get_input_activations(self) -> Dict | Sequence[Any]:
        return self._model_loader.load_inputs()


class WanTransformerTester(TorchModelTester):
    """Tester for WanTransformer3DModel."""

    def __init__(
        self,
        variant_name: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
        **kwargs,
    ) -> None:
        self._model_loader = ModelLoader(variant_name, subfolder="transformer")
        super().__init__(comparison_config, run_mode, **kwargs)

    def _get_model(self) -> Model:
        transformer = self._model_loader.load_model()
        return WanTransformerWrapper(transformer)

    def _get_input_activations(self) -> Dict | Sequence[Any]:
        return self._model_loader.load_inputs()
