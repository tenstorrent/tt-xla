# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Sequence

from infra import ComparisonConfig, Model, RunMode, TorchModelTester

from third_party.tt_forge_models.oft.pytorch import ModelLoader


class OFTTester(TorchModelTester):
    """Tester for OFT model."""

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
        return self._model_loader.load_model()

    # @override
    def _get_input_activations(self) -> Dict | Sequence[Any]:
        return self._model_loader.load_inputs()

    # @override
    def _get_forward_method_args(self) -> Sequence[Any]:
        """
        Overrides to handle tuple of tensors.
        """
        if isinstance(self._input_activations, tuple):
            return list(self._input_activations)
        return []
