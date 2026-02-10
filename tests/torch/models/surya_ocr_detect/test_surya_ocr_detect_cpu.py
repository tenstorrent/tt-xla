from third_party.tt_forge_models.suryaocr.pytorch.loader import ModelLoader, ModelVariant
from tests.runner.requirements import RequirementsManager
import inspect

def test_surya_ocr_detect_cpu():
    model_loader = ModelLoader(ModelVariant.OCR_DETECTION)
    loader_path = inspect.getsourcefile(ModelLoader)
    with RequirementsManager.for_loader(loader_path):
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
        model = model_loader.load_model()
        inputs = model_loader.load_inputs()
        outputs = model(inputs)
        print(outputs)
