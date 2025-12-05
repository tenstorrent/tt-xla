# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Sequence

from infra import ComparisonConfig, Model, RunMode, TorchModelTester

from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.resnet.pytorch import ModelLoader


class ResnetTester(TorchModelTester):
    """Tester for resnet model."""

    def __init__(
        self,
        variant_name: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
        compiler_config: CompilerConfig = None,
        dtype_override=None,
    ) -> None:
        self._model_loader = ModelLoader(variant_name)
        super().__init__(
            comparison_config, run_mode, compiler_config, dtype_override=dtype_override
        )

    # @override
    def _get_model(self) -> Model:
        return self._model_loader.load_model()

    # @override
    def _get_input_activations(self) -> Dict | Sequence[Any]:
        return self._model_loader.load_inputs()
