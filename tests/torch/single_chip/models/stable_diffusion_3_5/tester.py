# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Sequence

from infra import ComparisonConfig, Model, RunMode, TorchModelTester

from third_party.tt_forge_models.stable_diffusion.pytorch import ModelLoader

from .model_utils import StableDiffusion35Wrapper


class StableDiffusion35Tester(TorchModelTester):
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
        model = pipe.transformer
        model = StableDiffusion35Wrapper(
            model, joint_attention_kwargs=None, return_dict=False
        )
        return model

    # @override
    def _get_input_activations(self) -> Dict | Sequence[Any]:
        return self._model_loader.load_inputs()
