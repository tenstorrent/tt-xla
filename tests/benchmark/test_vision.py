# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json

import torch
from benchmarks.vision_benchmark import benchmark_vision_torch_xla
from utils import aggregate_ttnn_perf_metrics, resolve_display_name

# Defaults for all vision models
DEFAULT_OPTIMIZATION_LEVEL = 2
DEFAULT_TRACE_ENABLED = True
DEFAULT_BATCH_SIZE = 1
DEFAULT_LOOP_COUNT = 128
DEFAULT_INPUT_SIZE = (3, 224, 224)  # (channels, height, width)
DEFAULT_DATA_FORMAT = torch.bfloat16
DEFAULT_REQUIRED_PCC = 0.97


def test_vision(
    model,
    model_info_name,
    output_file,
    load_inputs_fn,
    extract_output_tensor_fn,
    request=None,
    optimization_level=DEFAULT_OPTIMIZATION_LEVEL,
    trace_enabled=DEFAULT_TRACE_ENABLED,
    batch_size=DEFAULT_BATCH_SIZE,
    loop_count=DEFAULT_LOOP_COUNT,
    input_size=DEFAULT_INPUT_SIZE,
    data_format=DEFAULT_DATA_FORMAT,
    required_pcc=DEFAULT_REQUIRED_PCC,
):
    """Test vision model with the given configuration.

    Args:
        model: Loaded model instance in eval mode
        model_info_name: Model name for identification and reporting
        output_file: Path to save benchmark results as JSON
        load_inputs_fn: Function to load a single batch of preprocessed inputs.
            Signature: fn(batch_size, data_format) -> Tensor
        extract_output_tensor_fn: Function to extract tensor from model outputs (e.g. get .logits).
        optimization_level: Optimization level (0, 1, or 2)
        trace_enabled: Enable trace
        batch_size: Batch size
        loop_count: Number of benchmark iterations
        input_size: Input size tuple (channels, height, width) - channel-first format
        data_format: Data format
        required_pcc: Required PCC threshold
    """
    resolved_display_name = resolve_display_name(
        request=request, fallback=model_info_name
    )
    ttnn_perf_metrics_output_file = f"tt_xla_{resolved_display_name}_perf_metrics"

    print(f"Running vision benchmark for model: {model_info_name}")
    print(
        f"""Configuration:
    optimization_level={optimization_level}
    trace_enabled={trace_enabled}
    batch_size={batch_size}
    loop_count={loop_count}
    input_size={input_size}
    data_format={data_format}
    required_pcc={required_pcc}
    ttnn_perf_metrics_output_file={ttnn_perf_metrics_output_file}
    """
    )

    results = benchmark_vision_torch_xla(
        model=model,
        model_info_name=model_info_name,
        display_name=resolved_display_name,
        optimization_level=optimization_level,
        trace_enabled=trace_enabled,
        batch_size=batch_size,
        loop_count=loop_count,
        input_size=input_size,
        data_format=data_format,
        ttnn_perf_metrics_output_file=ttnn_perf_metrics_output_file,
        load_inputs_fn=load_inputs_fn,
        extract_output_tensor_fn=extract_output_tensor_fn,
        required_pcc=required_pcc,
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_info_name

        aggregate_ttnn_perf_metrics(ttnn_perf_metrics_output_file, results)

        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)


def test_efficientnet(output_file, request):
    from third_party.tt_forge_models.efficientnet.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    # Configuration
    data_format = torch.bfloat16
    batch_size = 8

    # Load model
    variant = ModelVariant.TIMM_EFFICIENTNET_B0
    loader = ModelLoader(variant=variant)
    model_info_name = loader.get_model_info(variant=variant).name
    model = loader.load_model(dtype_override=data_format)
    model = model.eval()

    def load_inputs_fn(batch_size, dtype):
        return loader.load_inputs(dtype_override=dtype, batch_size=batch_size)

    def extract_output_tensor_fn(output):
        return output

    test_vision(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        request=request,
        load_inputs_fn=load_inputs_fn,
        extract_output_tensor_fn=extract_output_tensor_fn,
        batch_size=batch_size,
        data_format=data_format,
    )


# Trace disabled: PCC degradation (https://github.com/tenstorrent/tt-xla/issues/3931)
def test_mnist(output_file, request):
    from third_party.tt_forge_models.mnist.image_classification.pytorch.loader import (
        ModelLoader,
    )

    # Configuration
    data_format = torch.bfloat16
    batch_size = 32
    input_size = (1, 28, 28)

    # Load model
    loader = ModelLoader()
    model_info_name = loader.get_model_info().name
    model = loader.load_model(dtype_override=data_format)
    model = model.eval()

    # MNIST doesn't have load_inputs in tt_forge_models, use random tensor
    def load_inputs_fn(batch_size, dtype):
        return torch.randn(batch_size, *input_size, dtype=dtype)

    def extract_output_tensor_fn(output):
        return output

    test_vision(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        request=request,
        load_inputs_fn=load_inputs_fn,
        extract_output_tensor_fn=extract_output_tensor_fn,
        batch_size=batch_size,
        input_size=input_size,
        data_format=data_format,
        trace_enabled=False,
    )


def test_mobilenetv2(output_file, request):
    from third_party.tt_forge_models.mobilenetv2.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    # Configuration
    data_format = torch.bfloat16
    batch_size = 12

    # Load model
    variant = ModelVariant.MOBILENET_V2_TORCH_HUB
    loader = ModelLoader(variant=variant)
    model_info_name = loader.get_model_info(variant=variant).name
    model = loader.load_model(dtype_override=data_format)
    model = model.eval()

    def load_inputs_fn(batch_size, dtype):
        return loader.load_inputs(dtype_override=dtype, batch_size=batch_size)

    def extract_output_tensor_fn(output):
        return output

    test_vision(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        request=request,
        load_inputs_fn=load_inputs_fn,
        extract_output_tensor_fn=extract_output_tensor_fn,
        batch_size=batch_size,
        data_format=data_format,
    )


# Kokoro is a text-to-speech model: a single tensor-only forward (phoneme ids +
# style vector + host-precomputed alignment) producing one waveform tensor, which
# matches the vision harness's "single forward, PCC on the output tensor" shape.
# It runs in float32 (the loader documents that lower precision yields mixed-dtype
# matmuls in the LSTM/decoder) and takes three named inputs, so load_inputs_fn
# returns a dict (handled generically by the harness). Bringup safe defaults:
# optimization_level=0, trace_enabled=False.
def test_kokoro(output_file, request):
    from third_party.tt_forge_models.kokoro.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    # Configuration: model is float32-only, single sequence (batch 1).
    data_format = torch.float32
    batch_size = 1

    # Load model
    variant = ModelVariant.BASE
    loader = ModelLoader(variant=variant)
    model_info_name = loader.get_model_info(variant=variant).name
    model = loader.load_model()
    model = model.eval()

    # The inputs are deterministic and the loader recomputes the duration->frame
    # alignment (a forward through the predictor) on each call, so compute the
    # input dict once and reuse it for every benchmark iteration.
    inputs = loader.load_inputs()

    def load_inputs_fn(batch_size, dtype):
        return inputs

    def extract_output_tensor_fn(output):
        return output

    test_vision(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        request=request,
        load_inputs_fn=load_inputs_fn,
        extract_output_tensor_fn=extract_output_tensor_fn,
        batch_size=batch_size,
        # Not an image; the harness uses input_size only for the reporting string.
        # Pass (batch, phonemes, frames) so the 3-element formatting is satisfied.
        input_size=(1, loader.DEFAULT_SEQ_LEN, loader.MAX_FRAMES),
        data_format=data_format,
        # The time-unrolled (32-step, bidirectional) LSTMs make a single forward
        # far heavier than a vision classifier, so use a smaller loop count than
        # the vision default of 128 to keep the run within a practical wall clock.
        loop_count=8,
        optimization_level=0,  # safe default for bringup; model-perf-tuning will ramp
        trace_enabled=False,  # safe default for bringup; model-perf-tuning will ramp
    )


def test_resnet50(output_file, request):
    from third_party.tt_forge_models.resnet.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    # Configuration
    data_format = torch.bfloat16
    batch_size = 8

    # Load model
    variant = ModelVariant.RESNET_50_HF
    loader = ModelLoader(variant=variant)
    model_info_name = loader.get_model_info(variant=variant).name
    model = loader.load_model(dtype_override=data_format)
    model = model.eval()

    def load_inputs_fn(batch_size, dtype):
        return loader.load_inputs(dtype_override=dtype, batch_size=batch_size)

    def extract_output_tensor_fn(output):
        return output.logits

    test_vision(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        request=request,
        load_inputs_fn=load_inputs_fn,
        extract_output_tensor_fn=extract_output_tensor_fn,
        batch_size=batch_size,
        data_format=data_format,
        required_pcc=0.90,
    )


def test_segformer(output_file, request):
    from third_party.tt_forge_models.segformer.semantic_segmentation.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    # Configuration
    data_format = torch.bfloat16
    batch_size = 1
    input_size = (3, 512, 512)

    # Load model
    variant = ModelVariant.B0_FINETUNED
    loader = ModelLoader(variant=variant)
    model_info_name = loader.get_model_info(variant=variant).name
    model = loader.load_model(dtype_override=data_format)
    model = model.eval()

    # Segformer doesn't have separate input_preprocess in tt_forge_models
    def load_inputs_fn(batch_size, dtype):
        return torch.randn(batch_size, *input_size, dtype=dtype)

    def extract_output_tensor_fn(output):
        return output.logits

    test_vision(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        request=request,
        load_inputs_fn=load_inputs_fn,
        extract_output_tensor_fn=extract_output_tensor_fn,
        batch_size=batch_size,
        input_size=input_size,
        data_format=data_format,
    )


# Trace disabled: host/device tensor shape mismatch (https://github.com/tenstorrent/tt-xla/issues/3933)
def test_swin(output_file, request):
    from third_party.tt_forge_models.swin.image_classification.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    # Configuration
    data_format = torch.bfloat16
    batch_size = 1
    input_size = (3, 512, 512)

    # Load model
    variant = ModelVariant.SWIN_S
    loader = ModelLoader(variant=variant)
    model_info_name = loader.get_model_info(variant=variant).name
    model = loader.load_model(dtype_override=data_format)
    model = model.eval()

    def load_inputs_fn(batch_size, dtype):
        return torch.randn(batch_size, *input_size, dtype=dtype)

    def extract_output_tensor_fn(output):
        return output

    test_vision(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        request=request,
        load_inputs_fn=load_inputs_fn,
        extract_output_tensor_fn=extract_output_tensor_fn,
        batch_size=batch_size,
        input_size=input_size,
        data_format=data_format,
        required_pcc=0.90,
        trace_enabled=False,
    )


def test_ufld(output_file, request):
    from third_party.tt_forge_models.ultra_fast_lane_detection.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    # Configuration
    data_format = torch.bfloat16
    batch_size = 1

    # Load model
    variant = ModelVariant.TUSIMPLE_RESNET34
    loader = ModelLoader(variant=variant)
    model_info_name = loader.get_model_info(variant=variant).name
    input_size = (3, *loader.config.input_size)
    model = loader.load_model(dtype_override=data_format)
    model = model.eval()

    def load_inputs_fn(batch_size, dtype):
        return torch.randn(batch_size, *input_size, dtype=dtype)

    def extract_output_tensor_fn(output):
        return output

    test_vision(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        request=request,
        load_inputs_fn=load_inputs_fn,
        extract_output_tensor_fn=extract_output_tensor_fn,
        batch_size=batch_size,
        input_size=input_size,
        data_format=data_format,
    )


def test_ufld_v2(output_file, request):
    from third_party.tt_forge_models.ultra_fast_lane_detection_v2.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    # Configuration
    data_format = torch.bfloat16
    batch_size = 1

    # Load model
    variant = ModelVariant.TUSIMPLE_RESNET34
    loader = ModelLoader(variant=variant)
    model_info_name = loader.get_model_info(variant=variant).name
    input_size = (3, loader.config.input_height, loader.config.input_width)
    model = loader.load_model(dtype_override=data_format)
    model = model.eval()

    def load_inputs_fn(batch_size, dtype):
        return torch.randn(batch_size, *input_size, dtype=dtype)

    def extract_output_tensor_fn(output):
        return output[0]

    test_vision(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        request=request,
        load_inputs_fn=load_inputs_fn,
        extract_output_tensor_fn=extract_output_tensor_fn,
        batch_size=batch_size,
        input_size=input_size,
        data_format=data_format,
    )


# Trace disabled: PCC degradation (https://github.com/tenstorrent/tt-xla/issues/3932)
def test_unet(output_file, request):
    from third_party.tt_forge_models.vgg19_unet.pytorch.loader import ModelLoader

    # Configuration
    data_format = torch.bfloat16
    batch_size = 1
    input_size = (3, 256, 256)

    # Load model
    loader = ModelLoader()
    model_info_name = loader.get_model_info().name
    model = loader.load_model(dtype_override=data_format)
    model = model.eval()

    def load_inputs_fn(batch_size, dtype):
        return torch.randn(batch_size, *input_size, dtype=dtype)

    def extract_output_tensor_fn(output):
        return output

    test_vision(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        request=request,
        load_inputs_fn=load_inputs_fn,
        extract_output_tensor_fn=extract_output_tensor_fn,
        batch_size=batch_size,
        input_size=input_size,
        data_format=data_format,
        trace_enabled=False,
    )


def test_vit(output_file, request):
    from third_party.tt_forge_models.vit.pytorch.loader import ModelLoader, ModelVariant

    # Configuration
    data_format = torch.bfloat16
    batch_size = 8

    # Load model
    variant = ModelVariant.BASE
    loader = ModelLoader(variant=variant)
    model_info_name = loader.get_model_info(variant=variant).name
    model = loader.load_model(dtype_override=data_format)
    model = model.eval()

    def load_inputs_fn(batch_size, dtype):
        return loader.load_inputs(dtype_override=dtype, batch_size=batch_size)

    def extract_output_tensor_fn(output):
        return output.logits

    test_vision(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        request=request,
        load_inputs_fn=load_inputs_fn,
        extract_output_tensor_fn=extract_output_tensor_fn,
        batch_size=batch_size,
        data_format=data_format,
    )


def test_vovnet(output_file, request):
    from third_party.tt_forge_models.vovnet.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    # Configuration
    data_format = torch.bfloat16
    batch_size = 8

    # Load model
    variant = ModelVariant.TIMM_VOVNET19B_DW_RAIN1K
    loader = ModelLoader(variant=variant)
    model_info_name = loader.get_model_info(variant=variant).name
    model = loader.load_model(dtype_override=data_format)
    model = model.eval()

    def load_inputs_fn(batch_size, dtype):
        return loader.load_inputs(dtype_override=dtype, batch_size=batch_size)

    def extract_output_tensor_fn(output):
        return output

    test_vision(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        request=request,
        load_inputs_fn=load_inputs_fn,
        extract_output_tensor_fn=extract_output_tensor_fn,
        batch_size=batch_size,
        data_format=data_format,
    )
