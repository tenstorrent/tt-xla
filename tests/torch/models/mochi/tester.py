# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tester for Mochi VAE model."""

from typing import Any, Dict, Sequence

from infra import ComparisonConfig, Model, RunMode, TorchModelTester

from third_party.tt_forge_models.mochi.pytorch import ModelLoader

from .model_utils import MochiVAEWrapper


class MochiVAETester(TorchModelTester):
    """Tester for Mochi VAE decoder."""

    def __init__(
        self,
        variant_name,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
        **kwargs,
    ) -> None:
        self._model_loader = ModelLoader(variant_name, subfolder="vae")
        super().__init__(comparison_config, run_mode, **kwargs)
        # Disable perf measurement. The perf loop calls _run_on_tt_device
        # multiple times, which re-triggers Dynamo tracing and produces
        # rank-8 intermediate tensors that exceed TT-Metal's rank-5
        # broadcasting limit.
        self._disable_perf_measurement = True

    def _get_model(self) -> Model:
        vae_model = self._model_loader.load_model()
        enable_tiling = self._model_loader._variant_config.enable_tiling
        if not enable_tiling:
            vae_model = vae_model.decoder
        return MochiVAEWrapper(vae_model, enable_tiling)

    def _get_input_activations(self) -> Dict | Sequence[Any]:
        return self._model_loader.load_inputs(vae_type="decoder")
