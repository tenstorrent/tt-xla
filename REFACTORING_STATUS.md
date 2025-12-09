# Model Refactoring Status for TT Inference Server

This document tracks the refactoring of models from `tt_forge_models` to support the TT Inference Server, following the ResNet refactoring pattern.

## Refactoring Pattern

Models are refactored to use:
- `VisionPreprocessor` and `VisionPostprocessor` from `tools.utils`
- Structured configs with `ModelConfig` dataclass
- `ModelVariant` StrEnum
- `_VARIANTS` dictionary
- Clean `input_preprocess()` and `output_postprocess()` methods

## Refactored Models

### ✅ Fully Refactored Models

1. **resnet** (pytorch) ✅
   - Location: `resnet/pytorch/loader.py`
   - Uses: VisionPreprocessor, VisionPostprocessor
   - Multiple sources: HuggingFace, TIMM, Torchvision
   - Status: Fully refactored

2. **vit** (pytorch) ✅
   - Location: `vit/pytorch/loader.py`
   - Uses: VisionPreprocessor, VisionPostprocessor
   - Sources: HuggingFace, Torchvision
   - Status: Fully refactored

3. **efficientnet** (pytorch) ✅
   - Location: `efficientnet/pytorch/loader.py`
   - Uses: VisionPreprocessor, VisionPostprocessor
   - Sources: Torchvision, TIMM
   - Status: Fully refactored

4. **mobilenetv2** (pytorch) ✅
   - Location: `mobilenetv2/pytorch/loader.py`
   - Uses: VisionPreprocessor, VisionPostprocessor
   - Sources: HuggingFace, TIMM, Torchvision
   - Status: Fully refactored

5. **vovnet** (pytorch) ✅
   - Location: `vovnet/pytorch/loader.py`
   - Uses: VisionPreprocessor, VisionPostprocessor
   - Status: Fully refactored

6. **segformer** (pytorch) ✅
   - Location: `segformer/pytorch/loader.py`
   - Uses: VisionPreprocessor, VisionPostprocessor
   - Task: Image Classification
   - Status: Fully refactored

7. **alexnet** (pytorch) ✅
   - Location: `alexnet/pytorch/loader.py`
   - Uses: VisionPreprocessor, VisionPostprocessor
   - Sources: Torchvision, OSMR
   - Status: Fully refactored
   - PR: #316
   - Test Command: `pytest -svv tests/runner/test_models.py::test_all_models_torch[alexnet/pytorch-alexnet-single_device-full-inference] 2>&1 | tee test_alexnet.log`

8. **vgg** (pytorch) ✅
   - Location: `vgg/pytorch/loader.py`
   - Uses: VisionPreprocessor, VisionPostprocessor
   - Status: Fully refactored
   - PR: #327
   - Test Command: `pytest -svv tests/runner/test_models.py::test_all_models_torch[vgg/pytorch-torchvision_vgg19_bn-single_device-full-inference] 2>&1 | tee test_vgg.log`

9. **googlenet** (pytorch) ✅
   - Location: `googlenet/pytorch/loader.py`
   - Uses: VisionPreprocessor, VisionPostprocessor
   - Status: Fully refactored
   - PR: #321
   - Test Command: `pytest -svv tests/runner/test_models.py::test_all_models_torch[googlenet/pytorch-googlenet-single_device-full-inference] 2>&1 | tee test_googlenet.log`

10. **inception** (pytorch) ✅
    - Location: `inception/pytorch/loader.py`
    - Uses: VisionPreprocessor, VisionPostprocessor
    - Status: Fully refactored
    - PR: #320
    - Test Command: `pytest -svv tests/runner/test_models.py::test_all_models_torch[inception/pytorch-inception_v3-single_device-full-inference] 2>&1 | tee test_inception.log`

11. **densenet** (pytorch) ✅
    - Location: `densenet/pytorch/loader.py`
    - Uses: VisionPreprocessor, VisionPostprocessor
    - Status: Fully refactored
    - PR: #299
    - Test Command: `pytest -svv tests/runner/test_models.py::test_all_models_torch[densenet/pytorch-densenet121-single_device-full-inference] 2>&1 | tee test_densenet.log`

12. **mobilenetv1** (pytorch) ✅
    - Location: `mobilenetv1/pytorch/loader.py`
    - Uses: VisionPreprocessor, VisionPostprocessor
    - Status: Fully refactored
    - PR: #328
    - Test Command: `pytest -svv tests/runner/test_models.py::test_all_models_torch[mobilenetv1/pytorch-mobilenet_v1-single_device-full-inference] 2>&1 | tee test_mobilenetv1.log`

13. **mobilenetv3** (pytorch) ✅
    - Location: `mobilenetv3/pytorch/loader.py`
    - Uses: VisionPreprocessor, VisionPostprocessor
    - Status: Fully refactored
    - PR: #306
    - Test Command: `pytest -svv tests/runner/test_models.py::test_all_models_torch[mobilenetv3/pytorch-mobilenet_v3_large-single_device-full-inference] 2>&1 | tee test_mobilenetv3.log`

14. **wide_resnet** (pytorch) ✅
    - Location: `wide_resnet/pytorch/loader.py`
    - Uses: VisionPreprocessor, VisionPostprocessor
    - Status: Fully refactored
    - PR: #300
    - Test Command: `pytest -svv tests/runner/test_models.py::test_all_models_torch[wide_resnet/pytorch-wide_resnet50_2-single_device-full-inference] 2>&1 | tee test_wide_resnet.log`

15. **resnext** (pytorch) ✅
    - Location: `resnext/pytorch/loader.py`
    - Uses: VisionPreprocessor, VisionPostprocessor
    - Status: Fully refactored
    - PR: #301
    - Test Command: `pytest -svv tests/runner/test_models.py::test_all_models_torch[resnext/pytorch-resnext50_32x4d-single_device-full-inference] 2>&1 | tee test_resnext.log`

16. **xception** (pytorch) ✅ **[NEW]**
    - Location: `xception/pytorch/loader.py`
    - Uses: VisionPreprocessor, VisionPostprocessor
    - Sources: TIMM
    - Status: Fully refactored
    - Branch: `lelanchelian/refactor_xception`
    - Test Command: `pytest -svv tests/runner/test_models.py::test_all_models_torch[xception/pytorch-xception65-single_device-full-inference] 2>&1 | tee test_xception.log`

### Partially Refactored Models

8. **unet** (pytorch) ✅
   - Location: `unet/pytorch/loader.py`
   - Uses: VisionPreprocessor
   - Task: Image Segmentation
   - Status: Partially refactored (uses VisionPreprocessor only)

9. **yolov3** (pytorch) ✅
   - Location: `yolov3/pytorch/loader.py`
   - Uses: VisionPreprocessor
   - Task: Object Detection
   - Status: Partially refactored (uses VisionPreprocessor only)

10. **yolov4** (pytorch) ✅
    - Location: `yolov4/pytorch/loader.py`
    - Uses: VisionPreprocessor
    - Task: Object Detection
    - Status: Partially refactored (uses VisionPreprocessor only)

---

## Next Easy Models to Refactor

These are image classification models with simple architectures similar to ResNet. They can be easily refactored to use `VisionPreprocessor` and `VisionPostprocessor`.

### Classic CNN Architectures


- [ ] **regnet** (pytorch, jax)
  - Location: `regnet/pytorch/loader.py`, `regnet/image_classification/jax/loader.py`
  - Task: CV_IMAGE_CLS
  - Refactorability: Easy
  - Test Command: `pytest -svv tests/runner/test_models.py::test_all_models_torch[regnet/pytorch-regnet_y_400mf-single_device-full-inference] 2>&1 | tee test_regnet.log`

### Vision Transformer Variants

- [ ] **beit** (pytorch, jax)
  - Location: `beit/pytorch/loader.py`, `beit/image_classification/jax/loader.py`
  - Task: CV_IMAGE_CLS
  - Refactorability: Easy - Similar to ViT
  - Test Command: `pytest -svv tests/runner/test_models.py::test_all_models_torch[beit/pytorch-beit_base_patch16_224-single_device-full-inference] 2>&1 | tee test_beit.log`

- [ ] **deit** (pytorch)
  - Location: `deit/pytorch/loader.py`
  - Task: CV_IMAGE_CLS
  - Refactorability: Easy
  - Test Command: `pytest -svv tests/runner/test_models.py::test_all_models_torch[deit/pytorch-deit_base_patch16_224-single_device-full-inference] 2>&1 | tee test_deit.log`

- [ ] **swin** (pytorch)
  - Location: `swin/image_classification/pytorch/loader.py`
  - Task: CV_IMAGE_CLS
  - Refactorability: Easy
  - Test Command: `pytest -svv tests/runner/test_models.py::test_all_models_torch[swin/pytorch-swin_tiny_patch4_window7_224-single_device-full-inference] 2>&1 | tee test_swin.log`

- [ ] **dinov2** (jax)
  - Location: `dinov2/image_classification/jax/loader.py`
  - Task: CV_IMAGE_CLS
  - Refactorability: Easy
  - Test Command: `pytest -svv tests/runner/test_models.py::test_all_models_jax[dinov2/image_classification/jax-dinov2_vitb14-single_device-full-inference] 2>&1 | tee test_dinov2.log`

- [ ] **perceiverio_vision** (pytorch)
  - Location: `perceiverio_vision/pytorch/loader.py`
  - Task: CV_IMAGE_CLS
  - Refactorability: Easy
  - Test Command: `pytest -svv tests/runner/test_models.py::test_all_models_torch[perceiverio_vision/pytorch-perceiver_io_vision-single_device-full-inference] 2>&1 | tee test_perceiverio_vision.log`

### Other Classification Architectures

- [ ] **mlp_mixer** (pytorch, jax)
  - Location: `mlp_mixer/pytorch/loader.py`, `mlp_mixer/image_classification/jax/loader.py`
  - Task: CV_IMAGE_CLS
  - Refactorability: Easy
  - Test Command: `pytest -svv tests/runner/test_models.py::test_all_models_torch[mlp_mixer/pytorch-mlp_mixer_b16_224-single_device-full-inference] 2>&1 | tee test_mlp_mixer.log`

- [ ] **ghostnet** (pytorch)
  - Location: `ghostnet/pytorch/loader.py`
  - Task: CV_IMAGE_CLS
  - Refactorability: Easy
  - Test Command: `pytest -svv tests/runner/test_models.py::test_all_models_torch[ghostnet/pytorch-ghostnet_100-single_device-full-inference] 2>&1 | tee test_ghostnet.log`

- [ ] **hardnet** (pytorch)
  - Location: `hardnet/pytorch/loader.py`
  - Task: CV_IMAGE_CLS
  - Refactorability: Easy
  - Test Command: `pytest -svv tests/runner/test_models.py::test_all_models_torch[hardnet/pytorch-hardnet_68-single_device-full-inference] 2>&1 | tee test_hardnet.log`

- [ ] **dla** (pytorch)
  - Location: `dla/pytorch/loader.py`
  - Task: CV_IMAGE_CLS
  - Refactorability: Easy
  - Test Command: `pytest -svv tests/runner/test_models.py::test_all_models_torch[dla/pytorch-dla34-single_device-full-inference] 2>&1 | tee test_dla.log`

- [ ] **hrnet** (pytorch)
  - Location: `hrnet/pytorch/loader.py`
  - Task: CV_IMAGE_CLS
  - Refactorability: Easy
  - Test Command: `pytest -svv tests/runner/test_models.py::test_all_models_torch[hrnet/pytorch-hrnet_w18-single_device-full-inference] 2>&1 | tee test_hrnet.log`

- [ ] **efficientnet_lite** (pytorch)
  - Location: `efficientnet_lite/pytorch/loader.py`
  - Task: CV_IMAGE_CLS
  - Refactorability: Easy
  - Test Command: `pytest -svv tests/runner/test_models.py::test_all_models_torch[efficientnet_lite/pytorch-efficientnet_lite0-single_device-full-inference] 2>&1 | tee test_efficientnet_lite.log`

- [ ] **nbeats** (pytorch)
  - Location: `nbeats/pytorch/loader.py`
  - Task: CV_IMAGE_CLS
  - Refactorability: Easy
  - Test Command: `pytest -svv tests/runner/test_models.py::test_all_models_torch[nbeats/pytorch-nbeats-single_device-full-inference] 2>&1 | tee test_nbeats.log`

- [ ] **suryaocr** (pytorch)
  - Location: `suryaocr/pytorch/loader.py`
  - Task: CV_IMAGE_CLS
  - Refactorability: Easy (though OCR-specific, classification-like)
  - Test Command: `pytest -svv tests/runner/test_models.py::test_all_models_torch[suryaocr/pytorch-suryaocr-single_device-full-inference] 2>&1 | tee test_suryaocr.log`

- [ ] **mnist** (pytorch, jax)
  - Location: `mnist/image_classification/pytorch/loader.py`, `mnist/image_classification/jax/loader.py`
  - Task: CV_IMAGE_CLS
  - Refactorability: Easy
  - Test Command: `pytest -svv tests/runner/test_models.py::test_all_models_torch[mnist/image_classification/pytorch-mnist-single_device-full-inference] 2>&1 | tee test_mnist.log`

---

## Refactoring Instructions

1. **Create a branch**: `git checkout -b lelanchelian/refactor_<model_name>`
2. **Refactor the loader** following the ResNet pattern:
   - Import `VisionPreprocessor` and `VisionPostprocessor` from `...tools.utils`
   - Add `input_preprocess()` method that uses `VisionPreprocessor`
   - Add `output_postprocess()` method that uses `VisionPostprocessor`
   - Keep `load_inputs()` as a backward compatibility wrapper
   - Update `load_model()` to store model instance for postprocessor
3. **Test the refactored model** using the test command format:
   ```bash
   pytest -svv tests/runner/test_models.py::test_all_models_torch[<model_path>/pytorch-<variant>-single_device-full-inference] 2>&1 | tee test_<model_name>.log
   ```
4. **Commit with message**: `refactor <model_name> to support tt_inference server`
5. **Update this document** to mark the model as refactored

---

## Notes

- **Refactored Pattern**: Models using `VisionPreprocessor` and `VisionPostprocessor` with structured configs
- **Easy Refactor**: Standard image classification models that can directly use the vision utilities
- **Difficult Refactor**: Models requiring custom preprocessing/postprocessing, multi-modal inputs, or specialized architectures

For more details on difficult models, see `third_party/tt_forge_models/VISION_MODELS_REFACTORING_STATUS.md`.
