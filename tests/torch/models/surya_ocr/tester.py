# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Sequence

from infra import ComparisonConfig, Model, RunMode, TorchModelTester


class SuryaOCRTester(TorchModelTester):
    """Tester for SuryaOCR model."""

    def __init__(
        self,
        variant: Any,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        from third_party.tt_forge_models.suryaocr.pytorch.loader import ModelLoader

        self._model_loader = ModelLoader(variant)
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> Model:
        from surya.detection import DetectionPredictor

        from .model_utils import _prepare_image

        DetectionPredictor.prepare_image = _prepare_image
        return self._model_loader.load_model()

    # @override
    def _get_input_activations(self) -> Dict | Sequence[Any]:
        return self._model_loader.load_inputs()
