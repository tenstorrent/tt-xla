# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Sequence
from infra import ComparisonConfig, Model, RunMode, TorchModelTester

class SuryaOCRDetectionTester(TorchModelTester):
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
        from surya.detection import (
            heatmap as _surya_heatmap,  # type: ignore[reportMissingImports]
        )
        from surya.detection import (
            DetectionPredictor,
        )
        from surya.detection import heatmap as _surya_heatmap2
        from surya.detection.processor import (
            SegformerImageProcessor,  # type: ignore[reportMissingImports]
        )

        from .utils import _get_dynamic_thresholds_torch, _detect_boxes_torch, _prepare_image, _segformer_preprocess
        DetectionPredictor.prepare_image = _prepare_image
        _surya_heatmap.get_dynamic_thresholds = _get_dynamic_thresholds_torch
        _surya_heatmap2.detect_boxes = _detect_boxes_torch
        SegformerImageProcessor._preprocess = _segformer_preprocess
        return self._model_loader.load_model()

    # @override
    def _get_input_activations(self) -> Dict | Sequence[Any]:
        return self._model_loader.load_inputs()
