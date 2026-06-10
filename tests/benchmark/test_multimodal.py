# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Multimodal (vision-language) benchmarks.

Config-driven entry points (one ``test_<model>`` per model) that drive a
single multimodal forward pass through the shared harness in
``benchmarks/multimodal_benchmark.py``. This mirrors the ``test_vision.py`` /
``vision_benchmark.py`` split: model-specific config lives here, the reusable
measurement logic lives in ``benchmarks/``.
"""

import json

import torch
from benchmarks.multimodal_benchmark import benchmark_multimodal_torch_xla
from utils import aggregate_ttnn_perf_metrics, resolve_display_name

# Defaults for all multimodal models
DEFAULT_OPTIMIZATION_LEVEL = 2
DEFAULT_TRACE_ENABLED = True
DEFAULT_LOOP_COUNT = 32
DEFAULT_DATA_FORMAT = torch.bfloat16
DEFAULT_REQUIRED_PCC = 0.97


def test_multimodal(
    model,
    model_info_name,
    output_file,
    load_inputs_fn,
    extract_output_tensor_fn,
    request=None,
    optimization_level=DEFAULT_OPTIMIZATION_LEVEL,
    trace_enabled=DEFAULT_TRACE_ENABLED,
    loop_count=DEFAULT_LOOP_COUNT,
    data_format=DEFAULT_DATA_FORMAT,
    required_pcc=DEFAULT_REQUIRED_PCC,
):
    """Benchmark a multimodal model with the given configuration.

    Args:
        model: Loaded model instance in eval mode
        model_info_name: Model name for identification and reporting
        output_file: Path to save benchmark results as JSON
        load_inputs_fn: Function returning a single dict of model inputs.
            Signature: fn(dtype: torch.dtype) -> dict[str, Tensor]
        extract_output_tensor_fn: Function to extract tensor from model outputs.
        optimization_level: Optimization level (0, 1, or 2)
        trace_enabled: Enable trace
        loop_count: Number of benchmark iterations
        data_format: Data format
        required_pcc: Required PCC threshold
    """
    resolved_display_name = resolve_display_name(
        request=request, fallback=model_info_name
    )
    ttnn_perf_metrics_output_file = f"tt_xla_{resolved_display_name}_perf_metrics"

    print(f"Running multimodal benchmark for model: {model_info_name}")
    print(
        f"""Configuration:
    optimization_level={optimization_level}
    trace_enabled={trace_enabled}
    loop_count={loop_count}
    data_format={data_format}
    required_pcc={required_pcc}
    ttnn_perf_metrics_output_file={ttnn_perf_metrics_output_file}
    """
    )

    results = benchmark_multimodal_torch_xla(
        model=model,
        model_info_name=model_info_name,
        display_name=resolved_display_name,
        optimization_level=optimization_level,
        trace_enabled=trace_enabled,
        loop_count=loop_count,
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


def test_internvl3(output_file, request):
    from third_party.tt_forge_models.internvl3.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    # Configuration
    data_format = torch.bfloat16

    # Load model
    variant = ModelVariant.INTERNVL3_1B_HF
    loader = ModelLoader(variant=variant)
    model_info_name = loader.get_model_info(variant=variant).name
    model = loader.load_model(dtype_override=data_format)
    model = model.eval()

    def load_inputs_fn(dtype):
        return loader.load_inputs(dtype_override=dtype)

    # Wrapper.forward already returns the logits tensor.
    def extract_output_tensor_fn(output):
        return output

    test_multimodal(
        model=model,
        model_info_name=model_info_name,
        output_file=output_file,
        request=request,
        load_inputs_fn=load_inputs_fn,
        extract_output_tensor_fn=extract_output_tensor_fn,
        data_format=data_format,
        required_pcc=0.90,
    )
