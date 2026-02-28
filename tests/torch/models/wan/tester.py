# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tester for Wan VAE model."""

from typing import Any, Dict, Sequence

from infra import ComparisonConfig, Model, RunMode, TorchModelTester

from third_party.tt_forge_models.wan.pytorch import ModelLoader


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
