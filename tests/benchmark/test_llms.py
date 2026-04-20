# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import os

import numpy as np
import pytest
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
DEFAULT_REQUIRED_PCC = 0.94


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
    decode_only: bool = False,
    weight_dtype_overrides: dict = None,
    input_output_sharding_spec=None,
    kv_cache_sharding_spec=None,
    per_user_prompts=None,
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
        decode_only=decode_only,
        weight_dtype_overrides=weight_dtype_overrides,
        input_output_sharding_spec=input_output_sharding_spec,
        kv_cache_sharding_spec=kv_cache_sharding_spec,
        per_user_prompts=per_user_prompts,
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
    decode_only=False,
    required_pcc=DEFAULT_REQUIRED_PCC,
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
        decode_only=decode_only,
        required_pcc=required_pcc,
        **kwargs,
    )


def test_llama_3_2_1b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_llama_3_2_3b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_gemma_1_1_2b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_gemma_2_2b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_phi1(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_phi1_5(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_phi2(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_falcon3_1b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_falcon3_3b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_qwen_2_5_0_5b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_qwen_3_0_6b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_qwen_3_1_7b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_qwen_3_4b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_qwen_2_5_1_5b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_qwen_2_5_3b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_qwen_3_8b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_qwen_2_5_7b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
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
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_mistral_7b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


# Trace disabled: host/device tensor shape mismatch (https://github.com/tenstorrent/tt-xla/issues/3934)
def test_ministral_8b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
        trace_enabled=False,
    )


def test_llama_3_1_8b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_falcon3_7b_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_falcon3_10b_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_llama_3_1_8b_instruct_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_mistral_7b_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


# Trace disabled: host/device tensor shape mismatch (https://github.com/tenstorrent/tt-xla/issues/3935)
def test_ministral_8b_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
        trace_enabled=False,
    )


def test_mistral_nemo_instruct_2407_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_mistral_small_24b_instruct_2501_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_qwen_2_5_14b_instruct_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_qwen_2_5_32b_instruct_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_qwen_2_5_coder_32b_instruct_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_qwen_3_0_6b_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_qwen_3_1_7b_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_qwen_3_8b_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_qwen_3_14b_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_qwen_3_32b_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_llama_3_8b_instruct_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_llama_3_1_8b_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_llama_3_8b_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
    )


def test_llama_3_1_70b_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
        weight_dtype_overrides={
            "model.layers.*.mlp.gate_proj.weight": "bfp_bf4",
            "model.layers.*.mlp.up_proj.weight": "bfp_bf4",
        },
    )


# Distinct per-user prompts for GPT-OSS (MoE) benchmarks. Replicating the same
# input across users collapses expert routing; these are tokenized and padded
# to PER_USER_PROMPT_LEN inside the benchmark.
_GPT_OSS_PER_USER_QUESTIONS = [
    "What is the capital of France, and why has it remained the political center for centuries? I'd love an answer that goes beyond the basics, covers the main ideas, and includes a few concrete examples.",
    "Can you explain how a transformer decides which tokens to attend to during self-attention? I'd love an answer that goes beyond the basics, covers all the main ideas, and includes a few concrete examples.",
    "What are the main differences between classical and quantum mechanics in terms of predictability? Please explain it step by step in a way a curious reader can follow, with clear, concrete, and helpful examples.",
    "How does photosynthesis convert sunlight into chemical energy inside a plant cell at the molecular level? I'd love an answer that goes beyond the basics, covers the main ideas, and includes a few concrete examples.",
    "Why do onions make people cry when chopped, and what is a reliable way to prevent it? Please explain it step by step in a way a curious reader can follow, with clear and concrete examples.",
    "What caused the fall of the Western Roman Empire, and which factors do historians weight most heavily? Please explain it step by step in a way a curious reader can follow, with clear and concrete examples.",
    "How do modern GPUs achieve such high floating-point throughput compared to general-purpose CPUs today? I'd love an answer that goes beyond the basics, covers all the main ideas, and includes a few concrete examples.",
    "What is the Monty Hall problem, and why does switching doors actually improve the chance of winning? Please explain it step by step in a way a curious reader can really follow, with concrete examples.",
    "How does sourdough bread get its sour flavor, and what role do wild yeasts play in the process? I'd love an answer that goes beyond the surface and really includes a few illustrative examples.",
    "What is the difference between TCP and UDP, and when would you pick one over the other? Please explain it step by step in a way a curious reader can follow, with clear and concrete examples.",
    "Why is the sky blue during the day but often red or orange at sunset, optically speaking? Please explain it step by step in a way a curious reader can really follow, with concrete examples.",
    "How do black holes form, and what happens to matter that crosses the event horizon from outside? Please explain it step by step in a way a curious reader can follow, with clear and concrete examples.",
    "What is the role of mitochondria in eukaryotic cells, and why are they called the powerhouse of the cell? Please explain it step by step in a way a curious reader can follow.",
    "Can you walk me through how HTTPS establishes a secure connection between a browser and a server? I'd love an answer that goes beyond the basics, covers the main ideas, and includes a few concrete examples.",
    "What are the main reasons Japan developed such a distinctive tea ceremony tradition over the centuries? I'd love an answer that goes beyond the basics, covers all the main ideas, and includes a few concrete examples.",
    "How do vaccines train the immune system, and why do some vaccines require booster shots to stay effective? Please explain it step by step in a way a curious reader can really follow, with concrete examples.",
    "What distinguishes a compiler from an interpreter, and are there languages that blur the line between them? Please explain it step by step in a way a curious reader can follow, with clear and concrete examples.",
    "Why do some bridges collapse under resonance, and what engineering practices help prevent this from happening? I'd love an answer that goes beyond the basics, covers the main ideas, and includes a few concrete examples.",
    "How does caffeine affect the brain, and why does its stimulating effect differ so much from person to person? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "What is the significance of the Rosetta Stone in the history of decoding ancient Egyptian hieroglyphs? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "How do modern recommender systems balance exploration and exploitation when suggesting items to their users? I'd love an answer that goes beyond the basics, covers all the main ideas, and includes a few concrete examples.",
    "What causes rainbows, and why do they always appear opposite the sun with a specific angular radius? Please explain it step by step in a way a curious reader can really follow, with concrete examples.",
    "Why did the Industrial Revolution start in Britain rather than in other parts of Europe at the time? Please explain it step by step in a way a curious reader can follow, with clear and concrete examples.",
    "How do refrigerators actually move heat from inside to outside, and what role does the refrigerant play? Please explain it step by step in a way a curious reader can really follow, with concrete examples.",
    "What is dynamic programming, and which kinds of problems is it particularly well suited to solve efficiently? Please explain it step by step in a way a curious reader can follow, with clear and concrete examples.",
    "How does the human eye perceive color, and why do some people see colors differently from one another? Please explain it step by step in a way a curious reader can really follow, with concrete examples.",
    "What is the difference between a virus, a worm, and a Trojan horse in computer security terms? Please explain it step by step in a way a curious reader can really follow, with concrete examples.",
    "How did Charles Darwin arrive at the theory of evolution by natural selection, and who influenced him most? Please explain it step by step in a way a curious reader can really follow, with concrete examples.",
    "What makes a chess engine like Stockfish so strong, and how does it evaluate positions at great depth? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "Why are some materials superconductors at low temperatures, and what practical applications does this enable? I'd love an answer that goes beyond the basics, covers the main ideas, and includes a few concrete examples.",
    "How do hurricanes form over warm ocean water, and what determines how intense they eventually become? I'd love an answer that goes beyond the basics, covers the main ideas, and includes a few concrete examples.",
    "What is the traveling salesman problem, and why is it such a famously hard problem in computer science? Please explain it step by step in a way a curious reader can really follow, with concrete examples.",
    "How did the printing press change European society in the 15th and 16th centuries, culturally and politically? I'd love an answer that goes beyond the surface and really includes a few illustrative examples.",
    "What is the reason many birds fly in a V-formation, and how does it help with long-distance travel? I'd love an answer that goes beyond the surface and really includes a few illustrative examples.",
    "How do antibiotics work at the cellular level, and why is antibiotic resistance such a growing concern? Please explain it step by step in a way a curious reader can follow, with clear and concrete examples.",
    "What is the Fourier transform, and why does it show up in so many different engineering and physics problems? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "How do tides work, and why does the moon have a larger effect than the more massive but distant sun? I'd love an answer that goes beyond the basics and includes a few concrete, illustrative examples.",
    "What is the purpose of dark matter in cosmology, and what evidence do scientists have for its existence? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "How does a combustion engine convert the chemical energy in gasoline into mechanical motion for the wheels? I'd love an answer that goes beyond the basics, covers the main ideas, and includes a few concrete examples.",
    "What is the Turing test, and why do many researchers now consider it a less useful benchmark than it once was? I'd love an answer that goes beyond the surface and includes a few illustrative examples.",
    "How do bees communicate the location of food sources to other bees inside the hive with their waggle dance? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "What factors led to the decline of the Ottoman Empire in the 19th and early 20th centuries over time? I'd love an answer that goes beyond the surface and includes a few illustrative examples.",
    "How does a solid-state drive differ internally from a traditional spinning hard disk drive in architecture? I'd love an answer that goes beyond the basics, covers the main ideas, and includes a few concrete examples.",
    "What is the difference between machine learning, deep learning, and classical statistics in applied practice? I'd love an answer that goes beyond the basics, covers the main ideas, and includes a few concrete examples.",
    "How does the human body regulate its internal temperature when external conditions change dramatically? I'd love an answer that goes beyond the basics, covers all the main ideas clearly, and includes several concrete and illustrative examples.",
    "What is the origin of the English language, and how has it absorbed vocabulary from so many other languages? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "How do satellites maintain a stable orbit around Earth, and what makes geostationary orbit so special? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "What is the role of enzymes in biological reactions, and why are they so specific to particular substrates? Please explain it step by step in a way a curious reader can really follow, with concrete examples.",
    "How do modern databases implement transactions, and what do the ACID guarantees actually mean in practice? Please explain it step by step in a way a curious reader can follow, with clear and concrete examples.",
    "Why do earthquakes happen along certain faults, and how do seismologists estimate magnitudes after an event? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "What is the philosophical debate between free will and determinism, and how do neuroscientists weigh in today? I'd love an answer that goes beyond the basics and includes a few concrete, illustrative examples.",
    "How does a cryptocurrency like Bitcoin reach consensus about the true state of its distributed ledger? I'd love an answer that goes beyond the basics, covers all the main ideas, and includes a few concrete examples.",
    "What is the reason leaves change color in autumn, and why does the timing vary so much by region? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "How does the immune system distinguish between the body's own cells and foreign pathogens most of the time? Please explain it step by step in a way a curious reader can follow, with clear and concrete examples.",
    "What is the Doppler effect, and how do astronomers use it to measure the motion of distant stars and galaxies? I'd love an answer that goes beyond the surface and includes a few illustrative examples.",
    "Why did the Silk Road become such an important trade route, and which goods and ideas traveled along it? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "How does a modern jet engine generate thrust, and what are the thermodynamic stages it uses internally? Please explain it step by step in a way a curious reader can really follow, with concrete examples.",
    "What is the P versus NP problem, and why is it still unsolved after so many decades of intense research? I'd love an answer that goes beyond the surface and really includes a few illustrative examples.",
    "How does mindfulness meditation affect the brain over time, and what does current neuroimaging research show? Please explain it step by step in a way a curious reader can really follow, with concrete examples.",
    "What causes thunderstorms, and why is lightning sometimes seen without the accompanying sound of thunder? I'd love an answer that goes beyond the basics, covers all the main ideas, and includes a few concrete examples.",
    "How do magicians use misdirection to create the illusion of impossible events even for attentive audiences? Please explain it step by step in a way a curious reader can follow, with clear and concrete examples.",
    "What is the reason that most of the observable universe is expanding away from us at accelerating rates? Please explain it step by step in a way a curious reader can follow, with clear and concrete examples.",
    "How do musical instruments like violins produce such rich, complex tones from a vibrating string and bow? Please explain it step by step in a way a curious reader can really follow, with concrete examples.",
    "What is reinforcement learning, and how does an agent balance short-term reward with long-term strategy? Please explain it step by step in a way a curious reader can follow, with clear and concrete examples.",
    "How did the Great Library of Alexandria come to be, and what happened to the knowledge it once contained? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "Why do some countries drive on the left side of the road while others drive on the right, historically? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "How does a lithium-ion battery store and release energy, and what causes it to slowly degrade over cycles? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "What is the difference between correlation and causation, and why do people confuse them so often in practice? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "How do octopuses solve puzzles, and what does their nervous system tell us about the evolution of intelligence? I'd love an answer that goes beyond the basics and includes a few concrete, illustrative examples.",
    "What is the theory of plate tectonics, and how did it revolutionize our understanding of the surface of Earth? I'd love an answer that goes beyond the surface and really includes a few illustrative examples.",
    "How do search engines rank web pages, and how has the approach evolved from early keyword matching to today? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "Why does bread rise when yeast is added, and what happens chemically during the baking process in the oven? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "What is the history of the Olympic Games, and how have they changed from their ancient Greek origins to today? I'd love an answer that goes beyond the basics and includes a few concrete, illustrative examples.",
    "How do noise-cancelling headphones cancel sound using tiny microphones and real-time signal processing? I'd love an answer that goes beyond the basics, covers the main ideas, and includes a few concrete examples.",
    "What is the significance of the discovery of the Higgs boson, and how does it relate to the masses of particles? Please explain it step by step in a way a curious reader can follow.",
    "How do coral reefs form, and why are they such important ecosystems despite occupying a small fraction of ocean? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "What is the Byzantine Generals Problem, and why is it a useful abstraction for understanding distributed systems? Please explain it step by step in a way a curious reader can really follow, with concrete examples.",
    "How do cats communicate with humans, and which behaviors appear to have evolved specifically for that purpose? Please explain it step by step in a way a curious reader can follow, with clear and concrete examples.",
    "What led to the start of World War I, and how did alliances turn a regional crisis into a global conflict? I'd love an answer that goes beyond the surface and really includes a few illustrative examples.",
    "How does a quantum computer differ from a classical computer, and what kinds of problems can it solve faster? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "Why is pi considered such an important mathematical constant, and where does it show up outside of geometry? Please explain it step by step in a way a curious reader can really follow, with concrete examples.",
    "How do vaccines get designed and tested, from initial discovery through clinical trials and eventual approval? I'd love an answer that goes beyond the basics, covers the main ideas, and includes a few concrete examples.",
    "What is the Great Filter hypothesis, and how does it relate to the Fermi paradox about extraterrestrial life? I'd love an answer that goes beyond the basics and includes a few concrete, illustrative examples.",
    "How do sound waves travel through air, water, and solids differently, and what governs their speed in each? I'd love an answer that goes beyond the basics and includes a few concrete, illustrative examples.",
    "What is the role of the hippocampus in forming long-term memories, and what happens when it is damaged? I'd love an answer that goes beyond the basics and includes a few concrete, illustrative examples.",
    "How did ancient Egyptians build the great pyramids at Giza, and which theories are currently best supported? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "What is the difference between static and dynamic typing in programming languages, and what are the trade-offs? Please explain it step by step in a way a curious reader can really follow, with concrete examples.",
    "How do gears in a transmission multiply torque, and why do electric vehicles usually need so few of them? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "What is the reason some perfumes last much longer on the skin than others, in terms of chemistry and evaporation? I'd love an answer that goes beyond the basics and includes a few concrete, illustrative examples.",
    "How do sports like tennis and basketball use advanced analytics now, and how have coaches adapted strategies? Please explain it step by step in a way a curious reader can follow, with clear and concrete examples.",
    "What causes the northern lights, and why are they more commonly seen near the magnetic poles of the Earth? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "How does a microphone convert sound into an electrical signal, and how do different designs change the output? Please explain it step by step in a way a curious reader can really follow, with concrete examples.",
    "What is the history of paper money, and why did it eventually replace commodity-based forms of currency? Please explain it step by step in a way a curious reader can really follow, with concrete examples.",
    "How do antidepressant medications work, and why do they often take several weeks to show a therapeutic effect? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "What is the difference between supervised and unsupervised learning, and when is each approach more appropriate? Please explain it step by step in a way a curious reader can really follow, with concrete examples.",
    "How do astronauts train for life on the International Space Station, including handling emergencies in microgravity? Please explain it step by step in a way a curious reader can follow, with clear and concrete examples.",
    "What is the reason some languages are read right to left while most European languages are read left to right? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "How does the stock market price of a company reflect the collective beliefs of investors about its future? Please explain it step by step in a way a curious reader can follow, with clear and concrete examples.",
    "What is the origin of the universe according to the Big Bang model, and how do we know what happened long ago? I'd love an answer that goes beyond the surface and includes a few illustrative examples.",
    "How does yeast-based brewing of beer differ from wine fermentation in terms of inputs, microbes, and conditions? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "What is the philosophy behind test-driven development, and in what contexts is it most effective for engineers? Please explain it step by step in a way a curious reader can really follow, with concrete examples.",
    "How do dogs interpret human emotions, and how has their co-evolution with humans shaped their social behavior? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "What caused the Great Depression of the 1930s, and how did government responses shape modern economic policy? I'd love an answer that goes beyond the basics and includes a few concrete, illustrative examples.",
    "How do compilers perform optimizations like inlining and loop unrolling, and when do these actually help performance? I'd love an answer that goes beyond the surface and really includes a few illustrative examples.",
    "What is the structure and function of DNA, and how did Watson and Crick combine ideas to propose the double helix? Please explain it step by step in a way a curious reader can follow.",
    "How do telescopes observe objects billions of light-years away, and what does that mean about looking back in time? I'd love an answer that goes beyond the surface and really includes a few illustrative examples.",
    "What is the reason some people are left-handed, and what does research suggest about possible genetic influences? Please explain it step by step in a way a curious reader can really follow, with concrete examples.",
    "How do ride-sharing apps match drivers to riders in real time, and what algorithms make the overall system efficient? I'd love an answer that goes beyond the basics and includes a few concrete, illustrative examples.",
    "What is the story of Joan of Arc, and why does she remain such a powerful figure in French national memory today? I'd love an answer that goes beyond the surface and includes a few illustrative examples.",
    "How do hormones coordinate activity between distant parts of the body, using the bloodstream as a delivery system? Please explain it step by step in a way a curious reader can really follow, with concrete examples.",
    "What is the history of the calendar, and why do we have exactly twelve months of varying lengths in the Gregorian one? I'd love an answer that goes beyond the surface and includes a few illustrative examples.",
    "How does cryptography use large prime numbers to make certain problems practically impossible to solve by brute force? Please explain it step by step in a way a curious reader can really follow, with concrete examples.",
    "What is the reason some animals hibernate in winter, and how does their metabolism change to conserve energy then? I'd love an answer that goes beyond the basics and includes a few concrete, illustrative examples.",
    "How do wind turbines convert moving air into electricity, and what factors limit their efficiency in practice today? Please explain it step by step in a way a curious reader can really follow, with concrete examples.",
    "What is the difference between weather and climate, and why is distinguishing between them important for public policy? Please explain it step by step in a way a curious reader can really follow, with concrete examples.",
    "How do video game engines render three-dimensional worlds in real time, and what tricks do they use to save work? I'd love an answer that goes beyond the surface and really includes a few illustrative examples.",
    "What is the Cambrian explosion, and why is that period so important in the overall history of life on Earth? I'd love an answer that goes beyond the surface and really includes a few illustrative examples.",
    "How do modern airplanes generate lift, and what role do the flaps, slats, and ailerons play during different phases? I'd love an answer that goes beyond the surface and includes examples.",
    "What is the reason traditional Japanese architecture often uses wood rather than stone, historically and culturally? I'd love an answer that goes beyond the basics, covers the main ideas, and includes a few concrete examples.",
    "How do allergies work, and why does the immune system sometimes overreact to harmless substances like pollen? Please explain it step by step in a way a curious reader can really follow, with concrete examples.",
    "What is the concept of opportunity cost in economics, and how can it change decisions that seem purely financial? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "How do spiders produce silk, and why is spider silk often compared favorably to steel by weight in material science? I'd love an answer that goes beyond the surface and really includes a few illustrative examples.",
    "What led to the European Enlightenment, and how did it shape modern ideas about government and individual rights? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "How does a microwave oven heat food, and why does it sometimes warm things unevenly or struggle with frozen items? I'd love an answer that goes beyond the surface and really includes a few illustrative examples.",
    "What is the role of sleep in memory consolidation, and what happens cognitively when people are chronically deprived? I'd love an answer that goes beyond the basics and includes a few concrete, illustrative examples.",
    "How do self-driving cars combine cameras, lidar, and radar to build a model of their surroundings while moving? I'd love an answer that goes beyond the basics and includes a few concrete, illustrative examples.",
    "What is the mathematical concept of infinity, and why are there different sizes of infinity according to set theory? I'd love an answer that goes beyond the basics and includes a few concrete and clearly illustrative examples.",
    "How does a vaccine mRNA platform like the one used for COVID-19 differ from older vaccine technologies fundamentally? I'd love an answer that goes beyond the basics and includes a few concrete, illustrative examples.",
    "What is the history of the internet, and how did early research networks evolve into the global system we use today? I'd love an answer that goes beyond the surface and really includes a few illustrative examples.",
]


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


# Trace disabled: ~23% slower with trace on bs=32 (https://github.com/tenstorrent/tt-xla/issues/4192)
def test_gpt_oss_20b_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
        mesh_config_fn=_gpt_oss_20b_mesh_config_fn,
        shard_spec_fn=_gpt_oss_20b_shard_spec_fn,
        trace_enabled=False,
        per_user_prompts=None if accuracy_testing else _GPT_OSS_PER_USER_QUESTIONS,
    )


def test_gpt_oss_20b_tp_batch_size_1(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
        mesh_config_fn=_gpt_oss_20b_mesh_config_fn,
        shard_spec_fn=_gpt_oss_20b_shard_spec_fn,
        batch_size=batch_size if batch_size is not None else 1,
        per_user_prompts=None if accuracy_testing else _GPT_OSS_PER_USER_QUESTIONS,
    )


def test_llama_3_1_70b_tp_galaxy(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
        arch="wormhole_galaxy",
    )


def test_gpt_oss_20b_tp_galaxy_batch_size_64(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
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
        decode_only=decode_only,
        batch_size=(
            batch_size if batch_size is not None else 64
        ),  # 128 fails to compile - https://github.com/tenstorrent/tt-xla/issues/3907
        arch="wormhole_galaxy",
        optimization_level=1,
        per_user_prompts=None if accuracy_testing else _GPT_OSS_PER_USER_QUESTIONS,
    )


def test_gpt_oss_120b_tp_galaxy_batch_size_64(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
):
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
        decode_only=decode_only,
        batch_size=(
            batch_size if batch_size is not None else 64
        ),  # 128 fails to compile - https://github.com/tenstorrent/tt-xla/issues/3907
        arch="wormhole_galaxy",
        optimization_level=1,
        weight_dtype_overrides={
            "model.layers.*.mlp.router.weight": "bfp_bf4",
            "model.layers.*.mlp.experts.gate_up_proj": "bfp_bf4",
            "model.layers.*.mlp.experts.down_proj": "bfp_bf4",
        },
        required_pcc=0.93,
        per_user_prompts=None if accuracy_testing else _GPT_OSS_PER_USER_QUESTIONS,
    )


def _galaxy_mesh_config_fn(model_loader, num_devices):
    """4x8 wormhole_galaxy mesh"""

    if num_devices != 32:
        raise ValueError("wormhole_galaxy benchmarks expect 32 devices (4x8 mesh).")
    return (4, 8), ("batch", "model")


def _moe_throughput_galaxy_shard_spec_fn(model_loader, model):
    """Sharding specs for MoE models optimized for throughput on 4x8 galaxy mesh.
    TP - 8 : DP - 4 : EP - 32
    Inputs are sharded on the batch axis DP - 4. One tile per device so batch 128 should be used.
    Attention weights are sharded on model axis TP - 8 and replicated along the batch axis.
    Expert weights are sharded across both model and batch axes EP - 32.
    """

    shard_specs = {}

    shard_specs[model.model.embed_tokens.weight] = (None, None)
    shard_specs[model.model.norm.weight] = (None,)
    # HF [vocab, hidden]: TP shard vocab (first dim); tt-metal transposes/pads on device — see tt-metal_galaxy_parallelism
    shard_specs[model.lm_head.weight] = (None, None)

    for layer in model.model.layers:
        shard_specs[layer.self_attn.q_proj.weight] = ("model", None)
        shard_specs[layer.self_attn.k_proj.weight] = ("model", None)
        shard_specs[layer.self_attn.v_proj.weight] = ("model", None)
        shard_specs[layer.self_attn.o_proj.weight] = (None, "model")
        shard_specs[layer.self_attn.sinks] = ("model",)
        shard_specs[layer.mlp.router.weight] = (None, None)
        # This is a temporary sharding spec to enable gpt oss to not get OOM on galaxy.
        # Once the MoE module is refactored, this should be changed to EP 32.
        shard_specs[layer.mlp.experts.gate_up_proj] = ("model", "batch", None)
        shard_specs[layer.mlp.experts.gate_up_proj_bias] = ("model", None)
        shard_specs[layer.mlp.experts.down_proj] = ("model", None, "batch")
        shard_specs[layer.mlp.experts.down_proj_bias] = ("model", "batch")
        shard_specs[layer.input_layernorm.weight] = (None,)
        shard_specs[layer.post_attention_layernorm.weight] = (None,)

    return shard_specs


def test_gpt_oss_120b_tp_dp_galaxy_batch_size_128(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
):
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
        decode_only=decode_only,
        batch_size=128,
        arch="wormhole_galaxy",
        optimization_level=1,
        mesh_config_fn=_galaxy_mesh_config_fn,
        shard_spec_fn=_moe_throughput_galaxy_shard_spec_fn,
        input_output_sharding_spec=("batch", None),
        kv_cache_sharding_spec=("batch", "model", None, None),
        trace_enabled=True,
        per_user_prompts=None if accuracy_testing else _GPT_OSS_PER_USER_QUESTIONS,
    )


def _gpt_oss_120b_qb2_mesh_config_fn(model_loader, num_devices):
    return (1, 4), ("batch", "model")


def _gpt_oss_120b_qb2_shard_spec_fn(model_loader, model):
    """QB2 (1,4) mesh shard specs — model-axis-only, no batch sharding."""
    shard_specs = {}
    shard_specs[model.model.embed_tokens.weight] = (None, None)
    shard_specs[model.model.norm.weight] = (None,)

    for layer in model.model.layers:
        shard_specs[layer.self_attn.q_proj.weight] = ("model", None)
        shard_specs[layer.self_attn.k_proj.weight] = ("model", None)
        shard_specs[layer.self_attn.v_proj.weight] = ("model", None)
        shard_specs[layer.self_attn.o_proj.weight] = (None, "model")
        shard_specs[layer.self_attn.sinks] = (None,)
        shard_specs[layer.mlp.experts.gate_up_proj] = ("model", None, None)
        shard_specs[layer.mlp.experts.gate_up_proj_bias] = ("model", None)
        shard_specs[layer.mlp.experts.down_proj] = ("model", None, None)
        shard_specs[layer.mlp.experts.down_proj_bias] = ("model", None)
    return shard_specs


def test_gpt_oss_120b_tp_qb2(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
):
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
        decode_only=decode_only,
        batch_size=batch_size if batch_size is not None else 8,
        arch="qb2-blackhole",
        optimization_level=1,
        trace_enabled=True,
        experimental_weight_dtype="bfp_bf8",
        weight_dtype_overrides={
            "default": "bfp_bf8",
            "model.layers.*.mlp.experts.gate_up_proj": "bfp_bf4",
            "model.layers.*.mlp.experts.down_proj": "bfp_bf4",
        },
        required_pcc=0.93,  # set for now as it's ~0.93 on test runs locally
        mesh_config_fn=_gpt_oss_120b_qb2_mesh_config_fn,
        # shard_spec_fn=_gpt_oss_120b_qb2_shard_spec_fn,
        per_user_prompts=None if accuracy_testing else _GPT_OSS_PER_USER_QUESTIONS,
    )
