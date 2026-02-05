# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Sequence

import torch
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
        from surya.common.surya.processor import (
            SuryaOCRProcessor,  # type: ignore[reportMissingImports]
        )
        from surya.detection import (
            DetectionPredictor,
        )
        from surya.detection import (
            heatmap as _surya_heatmap,  # type: ignore[reportMissingImports]
        )
        from surya.detection import heatmap as _surya_heatmap2
        from surya.detection.processor import (
            SegformerImageProcessor,  # type: ignore[reportMissingImports]
        )
        from surya.foundation.cache.dynamic_ops import (
            DynamicOpsCache,  # type: ignore[reportMissingImports]
        )
        from surya.foundation.cache.static_ops import (
            StaticOpsCache,  # type: ignore[reportMissingImports]
        )
        from surya.settings import settings  # type: ignore[reportMissingImports]

        # Force non-XLA behavior to avoid static padding-induced seq len mismatch
        settings.TORCH_DEVICE = "cpu"
        settings.COMPILE_FOUNDATION = False

        from .model_utils import (
            _detect_boxes_torch,
            _get_dynamic_thresholds_torch,
            _patched_dynamic_ops_cache_init,
            _patched_image_processor,
            _patched_process_and_tile_no_xla,
            _patched_static_ops_cache_init,
            _prepare_image,
            _segformer_preprocess,
        )

        DetectionPredictor.prepare_image = _prepare_image
        SegformerImageProcessor._preprocess = _segformer_preprocess
        _surya_heatmap.get_dynamic_thresholds = _get_dynamic_thresholds_torch
        _surya_heatmap2.detect_boxes = _detect_boxes_torch
        StaticOpsCache.__init__ = _patched_static_ops_cache_init
        DynamicOpsCache.__init__ = _patched_dynamic_ops_cache_init
        SuryaOCRProcessor._image_processor = _patched_image_processor
        SuryaOCRProcessor._process_and_tile = _patched_process_and_tile_no_xla
        return self._model_loader.load_model()

    # @override
    def _get_input_activations(self) -> Dict | Sequence[Any]:
        return self._model_loader.load_inputs()
