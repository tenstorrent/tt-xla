# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import os

import numpy as np
import pytest
import torch_xla.distributed.spmd as xs
from benchmarks.llm_benchmark import benchmark_llm_torch_xla
from llm_utils.token_accuracy import TokenAccuracy
from loguru import logger
from utils import create_model_loader, resolve_display_name

# Defaults for all llms
DEFAULT_OPTIMIZATION_LEVEL = 2
DEFAULT_TP_OPTIMIZATION_LEVEL = 1
DEFAULT_MEMORY_LAYOUT_ANALYSIS = False
DEFAULT_TRACE_ENABLED = True
DEFAULT_BATCH_SIZE = 32
DEFAULT_LOOP_COUNT = 1
# WARNING: Changing this value will affect accuracy metrics due to context length differences.
# If changed, ALL reference outputs (*.refpt files) must be regenerated with the same total_length
# using scripts/generate_reference_outputs.py --total_length <value>
DEFAULT_INPUT_SEQUENCE_LENGTH = 128
DEFAULT_DATA_FORMAT = "bfloat16"
DEFAULT_TASK = "text-generation"
DEFAULT_EXPERIMENTAL_WEIGHT_DTYPE = "bfp_bf8"
DEFAULT_EXPERIMENTAL_ENABLE_PERMUTE_MATMUL_FUSION = False
DEFAULT_REQUIRED_PCC = 0.95


def default_read_logits_fn(output):
    return output.logits


def test_llm(
    ModelLoaderModule,
    variant,
    output_file,
    optimization_level=DEFAULT_OPTIMIZATION_LEVEL,
    trace_enabled=DEFAULT_TRACE_ENABLED,
    batch_size=DEFAULT_BATCH_SIZE,
    loop_count=DEFAULT_LOOP_COUNT,
    input_sequence_length=DEFAULT_INPUT_SEQUENCE_LENGTH,
    data_format=DEFAULT_DATA_FORMAT,
    task=DEFAULT_TASK,
    experimental_weight_dtype=DEFAULT_EXPERIMENTAL_WEIGHT_DTYPE,
    experimental_enable_permute_matmul_fusion=DEFAULT_EXPERIMENTAL_ENABLE_PERMUTE_MATMUL_FUSION,
    read_logits_fn=default_read_logits_fn,
    mesh_config_fn=None,
    shard_spec_fn=None,
    arch=None,
    required_pcc=DEFAULT_REQUIRED_PCC,
    fp32_dest_acc_en=None,
    num_layers=None,
    request=None,
    accuracy_testing: bool = False,
    max_output_tokens=None,
    input_sharding_fn=None,
):
    """Test LLM model with the given variant and optional configuration overrides.

    Args:
        variant: Model variant identifier
        output_file: Path to save benchmark results as JSON
        optimization_level: Optimization level (0, 1, or 2)
        trace_enabled: Enable trace
        batch_size: Batch size
        loop_count: Number of benchmark iterations
        input_sequence_length: Input sequence length
        data_format: Data format
        task: Task type
        experimental_weight_dtype: Weight dtype for block format conversion (e.g. "bfp_bf8", "bfp_bf4", or "" for none)
        experimental_enable_permute_matmul_fusion: Enable permute matmul fusion optimization
        read_logits_fn: Function to extract logits from model output
        required_pcc: Required PCC threshold
        num_layers: Number of layers to override
        accuracy_testing: Enable token accuracy testing with reference data
    """
    # Set default batch size if None
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE

    model_loader = create_model_loader(
        ModelLoaderModule, num_layers=num_layers, variant=variant
    )
    if num_layers is not None and model_loader is None:
        pytest.fail(
            "num_layers override requested but ModelLoader does not support it."
        )
    model_info_name = model_loader.get_model_info(variant=variant).name
    display_name = resolve_display_name(request=request, fallback=model_info_name)

    ttnn_perf_metrics_output_file = f"tt_xla_{display_name}_perf_metrics"

    print(f"Running LLM benchmark for variant: {variant}")
    print(
        f"""Configuration:
    optimization_level={optimization_level}
    trace_enabled={trace_enabled}
    batch_size={batch_size}
    loop_count={loop_count}
    input_sequence_length={input_sequence_length}
    data_format={data_format}
    task={task}
    experimental_weight_dtype={experimental_weight_dtype}
    experimental_enable_permute_matmul_fusion={experimental_enable_permute_matmul_fusion}
    required_pcc={required_pcc}
    num_layers={num_layers}
    ttnn_perf_metrics_output_file={ttnn_perf_metrics_output_file}
    """
    )

    # Resolve model name for accuracy testing
    model_name_for_accuracy = None
    hf_model_name = None
    if accuracy_testing:
        model_name_for_accuracy = TokenAccuracy.get_model_name_from_variant(
            model_loader, variant
        )
        hf_model_name = TokenAccuracy.get_hf_model_name_from_variant(
            model_loader, variant
        )

    results = benchmark_llm_torch_xla(
        optimization_level=optimization_level,
        trace_enabled=trace_enabled,
        model_loader=model_loader,
        model_variant=variant,
        display_name=display_name,
        batch_size=batch_size,
        loop_count=loop_count,
        task=task,
        data_format=data_format,
        input_sequence_length=input_sequence_length,
        experimental_weight_dtype=experimental_weight_dtype,
        experimental_enable_permute_matmul_fusion=experimental_enable_permute_matmul_fusion,
        ttnn_perf_metrics_output_file=ttnn_perf_metrics_output_file,
        read_logits_fn=read_logits_fn,
        mesh_config_fn=mesh_config_fn,
        shard_spec_fn=shard_spec_fn,
        arch=arch,
        required_pcc=required_pcc,
        fp32_dest_acc_en=fp32_dest_acc_en,
        accuracy_testing=accuracy_testing,
        model_name_for_accuracy=model_name_for_accuracy,
        hf_model_name_for_accuracy=hf_model_name,
        max_output_tokens=max_output_tokens,
        input_sharding_fn=input_sharding_fn,
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_info_name

        # LLM-specific perf metrics handling: Use only decode graph (second file)
        # LLMs split into 2 graphs: prefill (index 0) and decode (index 1)
        # Only decode is relevant for throughput
        base_name = os.path.basename(ttnn_perf_metrics_output_file)
        perf_files = [
            f
            for f in os.listdir(".")
            if f.startswith(base_name) and f.endswith(".json")
        ]
        perf_files = sorted(perf_files)

        if len(perf_files) == 2:
            # Use only the decode graph (second file)
            decode_perf_file = perf_files[1]
            print(f"Using decode graph perf metrics from: {decode_perf_file}")

            with open(decode_perf_file, "r") as f:
                perf_metrics_data = json.load(f)

            if "summary" in perf_metrics_data and isinstance(
                perf_metrics_data["summary"], dict
            ):
                summary = perf_metrics_data["summary"]
                results["config"]["ttnn_total_ops"] = summary.get("total_ops", 0)
                results["config"]["ttnn_total_shardable_ops"] = summary.get(
                    "total_shardable_ops", 0
                )
                results["config"]["ttnn_effectively_sharded_ops"] = summary.get(
                    "effectively_sharded_ops", 0
                )
                results["config"]["ttnn_system_memory_ops"] = summary.get(
                    "system_memory_ops", 0
                )
                results["config"]["ttnn_effectively_sharded_percentage"] = summary.get(
                    "effectively_sharded_percentage", 0.0
                )
                results["config"]["ttnn_num_graphs"] = 2  # prefill + decode
        else:
            logger.warning(
                f"Expected 2 perf metrics files (prefill + decode) for LLM, but found {len(perf_files)}: {perf_files}. "
                f"Skipping perf metrics."
            )
            results["config"]["ttnn_num_graphs"] = len(perf_files)

        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)


def test_llm_tp(
    ModelLoaderModule,
    variant,
    output_file,
    num_layers=None,
    request=None,
    arch="wormhole_llmbox",
    **kwargs,
):
    mesh_config_fn = kwargs.pop(
        "mesh_config_fn", getattr(ModelLoaderModule, "get_mesh_config", None)
    )
    shard_spec_fn = kwargs.pop(
        "shard_spec_fn", getattr(ModelLoaderModule, "load_shard_spec", None)
    )

    if "optimization_level" in kwargs:
        optimization_level = kwargs.pop("optimization_level")
    else:
        optimization_level = DEFAULT_TP_OPTIMIZATION_LEVEL

    test_llm(
        ModelLoaderModule=ModelLoaderModule,
        variant=variant,
        output_file=output_file,
        optimization_level=optimization_level,
        mesh_config_fn=mesh_config_fn,
        shard_spec_fn=shard_spec_fn,
        arch=arch,
        num_layers=num_layers,
        request=request,
        **kwargs,
    )


def test_llama_3_2_1b(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.LLAMA_3_2_1B_INSTRUCT
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_llama_3_2_3b(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.LLAMA_3_2_3B_INSTRUCT
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_gemma_1_1_2b(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.gemma.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.GEMMA_1_1_2B_IT
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_gemma_2_2b(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.gemma.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.GEMMA_2_2B_IT
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_phi1(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.phi1.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.PHI1
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_phi1_5(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.phi1_5.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.PHI1_5
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_phi2(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.phi2.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.PHI2
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_falcon3_1b(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.falcon.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.FALCON_1B
    # Tuple format: (logits, past_key_values, ...)
    read_logits_fn = lambda output: output[0]
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        read_logits_fn=read_logits_fn,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_falcon3_3b(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.falcon.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.FALCON_3B
    # Tuple format: (logits, past_key_values, ...)
    read_logits_fn = lambda output: output[0]
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        read_logits_fn=read_logits_fn,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_qwen_2_5_0_5b(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.QWEN_2_5_0_5B_INSTRUCT
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        required_pcc=0.94,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_qwen_3_0_6b(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.QWEN_3_0_6B
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_qwen_3_1_7b(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.QWEN_3_1_7B
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_qwen_3_4b(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.QWEN_3_4B
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_qwen_2_5_1_5b(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.QWEN_2_5_1_5B_INSTRUCT
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_qwen_2_5_3b(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.QWEN_2_5_3B_INSTRUCT
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_qwen_3_8b(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.QWEN_3_8B
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_qwen_2_5_7b(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.QWEN_2_5_7B_INSTRUCT
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


# FAILED: KeyError: "L['self'].model.lifted_tensor_0"
def test_gemma_1_1_7b(output_file, num_layers, request, max_output_tokens):
    from third_party.tt_forge_models.gemma.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.GEMMA_1_1_7B_IT
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
        max_output_tokens=max_output_tokens,
    )


# FAILED: TypeError: Phi3ForCausalLM.forward() got an unexpected keyword argument 'cache_position'
def test_phi3_mini(output_file, num_layers, request, max_output_tokens):
    from third_party.tt_forge_models.phi3.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.MINI_4K
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
        max_output_tokens=max_output_tokens,
    )


# FAILED: KeyError: 'lifted_tensor_0'
def test_phi3_5_mini(output_file, num_layers, request, max_output_tokens):
    from third_party.tt_forge_models.phi3.phi_3_5.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.MINI_INSTRUCT
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
        max_output_tokens=max_output_tokens,
    )


# FAILED: AttributeError: 'MambaConfig' object has no attribute 'num_attention_heads'
def test_mamba_2_8b(output_file, num_layers, request, max_output_tokens):
    from third_party.tt_forge_models.mamba.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.MAMBA_2_8B
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
        max_output_tokens=max_output_tokens,
    )


def test_falcon3_7b(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.falcon.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.FALCON_7B
    # Tuple format: (logits, past_key_values, ...)
    read_logits_fn = lambda output: output[0]
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        read_logits_fn=read_logits_fn,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_mistral_7b(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.mistral.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.MISTRAL_7B_INSTRUCT_V03
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


# Trace disabled: host/device tensor shape mismatch (https://github.com/tenstorrent/tt-xla/issues/3934)
def test_ministral_8b(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.mistral.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.MINISTRAL_8B
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
        fp32_dest_acc_en=False,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
        trace_enabled=False,
    )


def test_llama_3_1_8b(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.LLAMA_3_1_8B_INSTRUCT
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
        fp32_dest_acc_en=False,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_falcon3_7b_tp(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.falcon.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.FALCON_7B
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_falcon3_10b_tp(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.falcon.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.FALCON_10B
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_llama_3_1_8b_instruct_tp(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.LLAMA_3_1_8B_INSTRUCT
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_mistral_7b_tp(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.mistral.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.MISTRAL_7B_INSTRUCT_V03
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


# Trace disabled: host/device tensor shape mismatch (https://github.com/tenstorrent/tt-xla/issues/3935)
def test_ministral_8b_tp(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.mistral.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.MINISTRAL_8B
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
        trace_enabled=False,
    )


def test_mistral_nemo_instruct_2407_tp(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.mistral.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.MISTRAL_NEMO_INSTRUCT_2407
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_mistral_small_24b_instruct_2501_tp(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.mistral.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.MISTRAL_SMALL_24B_INSTRUCT_2501
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_qwen_2_5_14b_instruct_tp(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.QWEN_2_5_14B_INSTRUCT
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_qwen_2_5_32b_instruct_tp(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.QWEN_2_5_32B_INSTRUCT
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_qwen_2_5_coder_32b_instruct_tp(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.qwen_2_5_coder.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.QWEN_2_5_CODER_32B_INSTRUCT
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_qwen_3_0_6b_tp(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.QWEN_3_0_6B
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_qwen_3_1_7b_tp(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.QWEN_3_1_7B
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_qwen_3_8b_tp(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.QWEN_3_8B
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_qwen_3_14b_tp(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.QWEN_3_14B
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_qwen_3_32b_tp(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.QWEN_3_32B
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_llama_3_8b_instruct_tp(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.LLAMA_3_8B_INSTRUCT
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_llama_3_1_8b_tp(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.LLAMA_3_1_8B
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_llama_3_8b_tp(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.LLAMA_3_8B
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


def test_llama_3_1_70b_tp(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.LLAMA_3_1_70B_INSTRUCT
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )


# Use 1x8 shard specs for gpt-oss-20b until https://github.com/tenstorrent/tt-xla/issues/3490 is resolved.
def _gpt_oss_20b_mesh_config_fn(model_loader, num_devices):
    return (1, num_devices), ("batch", "model")


def _gpt_oss_20b_shard_spec_fn(model_loader, model):
    shard_specs = {}
    for layer in model.model.layers:
        shard_specs[layer.self_attn.q_proj.weight] = ("model", None)
        shard_specs[layer.self_attn.k_proj.weight] = ("model", None)
        shard_specs[layer.self_attn.v_proj.weight] = ("model", None)
        shard_specs[layer.self_attn.o_proj.weight] = (None, "model")
        shard_specs[layer.self_attn.sinks] = (None,)
        shard_specs[layer.mlp.router.weight] = (None, None)
        shard_specs[layer.mlp.experts.gate_up_proj] = ("model", None, None)
        shard_specs[layer.mlp.experts.gate_up_proj_bias] = ("model", None)
        shard_specs[layer.mlp.experts.down_proj] = ("model", None, None)
        shard_specs[layer.mlp.experts.down_proj_bias] = ("model", None)
    return shard_specs


def _batch_parallel_input_sharding_fn(mesh, input_args):
    """Data-parallel style batch sharding (activations only).

    Matches non-xfail DP patterns (vLLM plugin / examples): shard only
    activation inputs on the batch axis and keep parameter sharding policy
    independent.
    """
    xs.mark_sharding(input_args["input_ids"], mesh, ("batch", None))


def _gpt_oss_galaxy_mesh_config_fn(model_loader, num_devices):
    """4x8 wormhole_galaxy mesh (benchmark-only; forge loader stays upstream main)."""

    if num_devices != 32:
        raise ValueError(
            "GPT-OSS wormhole_galaxy benchmarks expect 32 devices (4x8 mesh)."
        )
    return (4, 8), ("batch", "model")


def _gpt_oss_galaxy_shard_spec_fn(model_loader, model):
    """Galaxy 4x8 shard layout aligned with repo ``tt-metal_galaxy_parallelism`` (tt-metal HF tensors).

    Mesh axes: ``("batch", "model")`` — TP on the 8-wide ``model`` columns; weights use ``None``
    on ``batch`` (replicated across rows). Activations: batch-sharded via ``input_sharding_fn``.

    MoE / sinks / lm_head axes match the "Closest tt-metal-style shard specs" and shape notes
    there (vocab-parallel ``lm_head`` on HF ``[vocab, hidden]``).
    """

    batch_axis = None

    shard_specs = {}

    shard_specs[model.model.embed_tokens.weight] = (None, batch_axis)
    shard_specs[model.model.norm.weight] = (batch_axis,)
    # HF [vocab, hidden]: TP shard vocab (first dim); tt-metal transposes/pads on device — see tt-metal_galaxy_parallelism
    shard_specs[model.lm_head.weight] = ("model", batch_axis)

    for layer in model.model.layers:
        shard_specs[layer.self_attn.q_proj.weight] = ("model", batch_axis)
        shard_specs[layer.self_attn.q_proj.bias] = ("model",)
        shard_specs[layer.self_attn.k_proj.weight] = ("model", batch_axis)
        shard_specs[layer.self_attn.k_proj.bias] = ("model",)
        shard_specs[layer.self_attn.v_proj.weight] = ("model", batch_axis)
        shard_specs[layer.self_attn.v_proj.bias] = ("model",)
        shard_specs[layer.self_attn.o_proj.weight] = (batch_axis, "model")
        shard_specs[layer.self_attn.o_proj.bias] = (batch_axis,)
        shard_specs[layer.self_attn.sinks] = ("model",)
        shard_specs[layer.mlp.router.weight] = (None, batch_axis)
        shard_specs[layer.mlp.experts.gate_up_proj] = (None, batch_axis, "model")
        shard_specs[layer.mlp.experts.gate_up_proj_bias] = (None, "model")
        shard_specs[layer.mlp.experts.down_proj] = (None, "model", batch_axis)
        shard_specs[layer.mlp.experts.down_proj_bias] = (None, batch_axis)
        shard_specs[layer.input_layernorm.weight] = (batch_axis,)
        shard_specs[layer.post_attention_layernorm.weight] = (batch_axis,)

    return shard_specs


# Trace disabled: host/device tensor shape mismatch (https://github.com/tenstorrent/tt-xla/issues/3929)
def test_gpt_oss_20b_tp(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.gpt_oss.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.GPT_OSS_20B
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
        mesh_config_fn=_gpt_oss_20b_mesh_config_fn,
        shard_spec_fn=_gpt_oss_20b_shard_spec_fn,
        trace_enabled=False,
    )


# Trace disabled: host/device tensor shape mismatch (https://github.com/tenstorrent/tt-xla/issues/3929)
def test_gpt_oss_20b_tp_batch_size_1(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.gpt_oss.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.GPT_OSS_20B
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        max_output_tokens=max_output_tokens,
        mesh_config_fn=_gpt_oss_20b_mesh_config_fn,
        shard_spec_fn=_gpt_oss_20b_shard_spec_fn,
        batch_size=batch_size if batch_size is not None else 1,
        trace_enabled=False,
    )


def test_llama_3_1_70b_tp_galaxy(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.LLAMA_3_1_70B_INSTRUCT
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
        arch="wormhole_galaxy",
    )


# Trace disabled: host/device tensor shape mismatch (https://github.com/tenstorrent/tt-xla/issues/3929)
def test_gpt_oss_20b_tp_galaxy_batch_size_64(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.gpt_oss.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.GPT_OSS_20B
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        max_output_tokens=max_output_tokens,
        batch_size=(
            batch_size if batch_size is not None else 64
        ),  # 128 fails to compile - https://github.com/tenstorrent/tt-xla/issues/3907
        input_sharding_fn=_batch_parallel_input_sharding_fn,
        arch="wormhole_galaxy",
        optimization_level=1,
        mesh_config_fn=_gpt_oss_galaxy_mesh_config_fn,
        shard_spec_fn=_gpt_oss_galaxy_shard_spec_fn,
        trace_enabled=False,
    )


# Trace disabled: host/device tensor shape mismatch (https://github.com/tenstorrent/tt-xla/issues/3929)
def test_gpt_oss_120b_tp_galaxy_batch_size_64(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    """Same Galaxy mesh, shard spec, batch size, and input sharding as ``test_gpt_oss_20b_tp_galaxy_batch_size_64``."""

    from third_party.tt_forge_models.gpt_oss.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.GPT_OSS_120B
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        max_output_tokens=max_output_tokens,
        batch_size=(
            batch_size if batch_size is not None else 64
        ),  # 128 fails to compile - https://github.com/tenstorrent/tt-xla/issues/3907
        input_sharding_fn=_batch_parallel_input_sharding_fn,
        arch="wormhole_galaxy",
        optimization_level=1,
        mesh_config_fn=_gpt_oss_galaxy_mesh_config_fn,
        shard_spec_fn=_gpt_oss_galaxy_shard_spec_fn,
        trace_enabled=False,
    )
