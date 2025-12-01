# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Sequence

from infra import ComparisonConfig, Model, RunMode, TorchModelTester

from third_party.tt_forge_models.stable_diffusion_xl.pytorch import ModelLoader

from .model_utils import StableDiffusionXLWrapper


class StableDiffusionXLTester(TorchModelTester):
    """Tester for Stable Diffusion XL model."""

    def __init__(
        self,
        variant_name: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_loader = ModelLoader(variant_name)
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> Model:
        pipe = self._model_loader.load_model()
        model = pipe.unet
        model = StableDiffusionXLWrapper(
            model, self._model_loader.load_inputs()[3], cross_attention_kwargs=None
        )
        return model

    # @override
    def _get_input_activations(self) -> Dict | Sequence[Any]:
        inputs_list = self._model_loader.load_inputs()
        inputs = [inputs_list[0], inputs_list[1], inputs_list[2]]
        return inputs
