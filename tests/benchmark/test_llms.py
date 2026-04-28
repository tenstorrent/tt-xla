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
    inject_custom_moe: bool = False,
    custom_moe_cluster_axis: int = 0,
    gpt_oss_fused_decode: bool = False,
    preprocess_fused_weights: bool = False,
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
        inject_custom_moe: Replace MoE blocks with tt_torch sparse_mlp adapters
        custom_moe_cluster_axis: Mesh axis used for custom MoE dispatch/combine
        gpt_oss_fused_decode: Enable the GPT-OSS fused decode composite path
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
    inject_custom_moe={inject_custom_moe}
    custom_moe_cluster_axis={custom_moe_cluster_axis}
    gpt_oss_fused_decode={gpt_oss_fused_decode}
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
        inject_custom_moe=inject_custom_moe,
        custom_moe_cluster_axis=custom_moe_cluster_axis,
        gpt_oss_fused_decode=gpt_oss_fused_decode,
        preprocess_fused_weights=preprocess_fused_weights,
        per_user_prompts=per_user_prompts,
    )

    if isinstance(results.get("config"), dict):
        results["config"]["inject_custom_moe"] = inject_custom_moe
        results["config"]["custom_moe_cluster_axis"] = custom_moe_cluster_axis
        results["config"]["gpt_oss_fused_decode"] = gpt_oss_fused_decode
        results["config"]["preprocess_fused_weights"] = preprocess_fused_weights

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
    "What is the capital of France, and why has it remained the political center for centuries, for the curious reader?",
    "How does photosynthesis convert sunlight into chemical energy inside a plant cell at the molecular level for engineers and scientists today?",
    "Why do onions make people cry when chopped, and what is reliable way to prevent it, in modern engineering practice?",
    "What caused the fall of the Western Roman Empire, and which factors do historians weight most heavily in modern biological research?",
    "What is the Monty Hall problem, and why does switching doors actually improve the chance of winning, concretely?",
    "How does sourdough bread get its sour flavor, and what role do wild yeasts play in process really?",
    "What is the difference between TCP and UDP, and when would you pick one over the other for engineers and scientists?",
    "Why is sky blue during the day but often red or orange at sunset, optically speaking in modern biological research?",
    "How do black holes form, and what happens to matter that crosses the event horizon from outside in modern engineering practice?",
    "Can you walk me through how HTTPS establishes a secure connection between a browser and a server for the curious reader today?",
    "How do vaccines train immune system, and why do some vaccines require booster shots to stay effective in modern computer architecture?",
    "What distinguishes a compiler from an interpreter, and are there languages that blur the line between them for everyday curious readers?",
    "Why do some bridges collapse under resonance, and what engineering practices help prevent this from happening for the curious reader today?",
    "How does caffeine affect the brain, and why does its stimulating effect differ so much from person to person in practice?",
    "What is significance of the Rosetta Stone in the history of decoding ancient Egyptian hieroglyphs in everyday science?",
    "What causes rainbows, and why do they always appear opposite sun with a specific angular radius, in everyday life?",
    "Why did Industrial Revolution start in Britain rather than in other parts of Europe at the time for the curious reader today?",
    "How do refrigerators actually move heat from inside to outside, and what role does the refrigerant play, in practice?",
    "What is dynamic programming, and which kinds of problems is it particularly well suited to solve efficiently, for curious reader?",
    "How does human eye perceive color, and why do some people see colors differently from one another for everyday curious readers?",
    "What is difference between a virus, a worm, and a Trojan horse in computer security terms in modern biological research?",
    "How did Charles Darwin arrive at theory of evolution by natural selection, and who influenced him most in modern engineering practice?",
    "What makes chess engine like Stockfish so strong, and how does it evaluate positions at great depth, concretely?",
    "Why are some materials superconductors at low temperatures, and what practical applications does this enable, in real-world settings?",
    "How do hurricanes form over warm ocean water, and what determines how intense they eventually become, in modern engineering practice?",
    "What is traveling salesman problem, and why is it such a famously hard problem in computer science for engineers in practice?",
    "How did printing press change European society in the 15th and 16th centuries for the curious reader today?",
    "What is reason many birds fly in a V-formation, and how does it help with long-distance travel today?",
    "How do antibiotics work at cellular level, and why is antibiotic resistance such a growing concern for engineers and scientists today?",
    "What is Fourier transform, and why does it show up in so many different engineering and physics problems in everyday science?",
    "How do tides work, and why does the moon have a larger effect than the more massive but distant sun really?",
    "What is purpose of dark matter in cosmology, and what evidence do scientists have for its existence in everyday science?",
    "How does a combustion engine convert the chemical energy in gasoline into mechanical motion for the wheels, for engineers and scientists?",
    "What is the Turing test, and why do many researchers now consider it less useful benchmark than it once was?",
    "How do bees communicate location of food sources to other bees inside the hive with their waggle dance, concretely?",
    "What factors led to decline of the Ottoman Empire in the 19th and early 20th centuries over time?",
    "How does a solid-state drive differ internally from a traditional spinning hard disk drive in architecture, in real-world settings?",
    "What is the difference between machine learning, deep learning, and classical statistics in applied practice, for the curious reader?",
    "What is origin of the English language, and how has it absorbed vocabulary from so many other languages in everyday life?",
    "How do satellites maintain stable orbit around Earth, and what makes geostationary orbit so special, in practice?",
    "What is role of enzymes in biological reactions, and why are they so specific to particular substrates in modern computer architecture?",
    "How do modern databases implement transactions, and what do the ACID guarantees actually mean in practice in modern engineering practice?",
    "Why do earthquakes happen along certain faults, and how do seismologists estimate magnitudes after event in everyday science?",
    "What is the philosophical debate between free will and determinism, and how do neuroscientists weigh in today now?",
    "What is reason leaves change color in autumn, and why does the timing vary so much by region in everyday life?",
    "How does the immune system distinguish between the body's own cells and foreign pathogens most of the time for engineers in practice?",
    "What is Doppler effect, and how do astronomers use it to measure the motion of distant stars and galaxies?",
    "Why did the Silk Road become such important trade route, and which goods and ideas traveled along it in everyday life?",
    "How does modern jet engine generate thrust, and what are the thermodynamic stages it uses internally, in modern science?",
    "What is the P versus NP problem, and why is it still unsolved after so many decades of intense research?",
    "How does mindfulness meditation affect the brain over time, and what does current neuroimaging research show for everyday users?",
    "How do magicians use misdirection to create the illusion of impossible events even for attentive audiences, in modern engineering?",
    "What is reason that most of the observable universe is expanding away from us at accelerating rates, for engineers and scientists?",
    "How do musical instruments like violins produce such rich, complex tones from vibrating string and bow, in modern science?",
    "What is reinforcement learning, and how does an agent balance short-term reward with long-term strategy in modern computer architecture?",
    "How did Great Library of Alexandria come to be, and what happened to the knowledge it once contained, in practice?",
    "Why do some countries drive on the left side of the road while others drive on the right, historically in practice?",
    "How does lithium-ion battery store and release energy, and what causes it to slowly degrade over cycles in everyday science?",
    "What is difference between correlation and causation, and why do people confuse them so often in practice for everyday users?",
    "How do octopuses solve puzzles, and what does their nervous system tell us about evolution of intelligence, briefly?",
    "What is theory of plate tectonics, and how did it revolutionize our understanding of the surface of Earth today?",
    "How do search engines rank web pages, and how has approach evolved from early keyword matching to today, concretely?",
    "Why does bread rise when yeast is added, and what happens chemically during baking process in the oven in everyday science?",
    "What is history of the Olympic Games, and how have they changed from their ancient Greek origins to today, briefly?",
    "How do noise-cancelling headphones cancel sound using tiny microphones and real-time signal processing for engineers and scientists today?",
    "How do coral reefs form, and why are they such important ecosystems despite occupying small fraction of ocean, concretely?",
    "What is the Byzantine Generals Problem, and why is it useful abstraction for understanding distributed systems in modern computer architecture?",
    "How do cats communicate with humans, and which behaviors appear to have evolved specifically for that purpose, in everyday life?",
    "What led to the start of World War I, and how did alliances turn regional crisis into a global conflict today?",
    "How does quantum computer differ from a classical computer, and what kinds of problems can it solve faster in everyday life?",
    "Why is pi considered such important mathematical constant, and where does it show up outside of geometry, in modern science?",
    "How do vaccines get designed and tested, from initial discovery through clinical trials and eventual approval, in modern engineering practice?",
    "What is the Great Filter hypothesis, and how does it relate to the Fermi paradox about extraterrestrial life today?",
    "How do sound waves travel through air, water, and solids differently, and what governs their speed in each really?",
    "What is the role of the hippocampus in forming long-term memories, and what happens when it is damaged really?",
    "How did ancient Egyptians build the great pyramids at Giza, and which theories are currently best supported in practice?",
    "What is the difference between static and dynamic typing in programming languages, and what are the trade-offs in everyday science?",
    "How do gears in transmission multiply torque, and why do electric vehicles usually need so few of them for everyday users?",
    "What is the reason some perfumes last much longer on the skin than others, in terms of chemistry and evaporation now?",
    "How do sports like tennis and basketball use advanced analytics now, and how have coaches adapted strategies for curious reader today?",
    "What causes the northern lights, and why are they more commonly seen near the magnetic poles of the Earth, briefly?",
    "How does a microphone convert sound into an electrical signal, and how do different designs change the output for everyday users?",
    "What is history of paper money, and why did it eventually replace commodity-based forms of currency for engineers and scientists?",
    "How do antidepressant medications work, and why do they often take several weeks to show therapeutic effect in everyday life?",
    "What is difference between supervised and unsupervised learning, and when is each approach more appropriate for engineers in practice?",
    "How do astronauts train for life on International Space Station, including handling emergencies in microgravity, in modern engineering practice?",
    "What is reason some languages are read right to left while most European languages are read left to right in everyday science?",
    "How does the stock market price of a company reflect the collective beliefs of investors about its future for everyday curious readers?",
    "What is origin of the universe according to the Big Bang model, and how do we know what happened long ago?",
    "How does yeast-based brewing of beer differ from wine fermentation in terms of inputs, microbes, for engineers and scientists?",
    "What is the philosophy behind test-driven development, and in what contexts is it most effective for engineers, in practice?",
    "How do dogs interpret human emotions, and how has their co-evolution with humans shaped their social behavior in practice?",
    "What caused the Great Depression of the 1930s, and how did government responses shape modern economic policy really?",
    "How do compilers perform optimizations like inlining and loop unrolling, and when do these actually help performance?",
    "How do telescopes observe objects billions of light-years away, and what does that mean about looking back in time?",
    "What is reason some people are left-handed, and what does research suggest about possible genetic influences for engineers and scientists?",
    "How do ride-sharing apps match drivers to riders in real time, and what algorithms make overall system efficient in practice?",
    "What is story of Joan of Arc, and why does she remain such a powerful figure in French national memory today?",
    "How do hormones coordinate activity between distant parts of body, using the bloodstream as a delivery system in modern computer architecture?",
    "What is history of the calendar, and why do we have exactly twelve months of varying lengths in the Gregorian one?",
    "How does cryptography use large prime numbers to make certain problems practically impossible to solve by brute force for everyday users?",
    "What is the reason some animals hibernate in winter, and how does their metabolism change to conserve energy then now?",
    "How do wind turbines convert moving air into electricity, and what factors limit their efficiency in practice today for everyday users?",
    "What is the difference between weather and climate, and why is distinguishing between them important for public policy in everyday life?",
    "How do video game engines render three-dimensional worlds in real time, and what tricks do they use to save work?",
    "What is Cambrian explosion, and why is that period so important in the overall history of life on Earth now?",
    "How do modern airplanes generate lift, and what role do the flaps, slats for engineers and scientists today?",
    "What is the reason traditional Japanese architecture often uses wood rather than stone, historically and culturally, for engineers and scientists?",
    "How do allergies work, and why does the immune system sometimes overreact to harmless substances like pollen, in practice?",
    "What is concept of opportunity cost in economics, and how can it change decisions that seem purely financial, concretely?",
    "How do spiders produce silk, and why is spider silk often compared favorably to steel by weight in material science?",
    "What led to the European Enlightenment, and how did it shape modern ideas about government and individual rights in practice?",
    "How does a microwave oven heat food, and why does it sometimes warm things unevenly or struggle with frozen items?",
    "What is the role of sleep in memory consolidation, and what happens cognitively when people are chronically deprived really?",
    "How do self-driving cars combine cameras, lidar, and radar to build model of their surroundings while moving, briefly?",
    "What is mathematical concept of infinity, and why are there different sizes of infinity according to set theory in everyday science?",
    "How does vaccine mRNA platform like the one used for COVID-19 differ from older vaccine technologies fundamentally in practice?",
    "What is the history of the internet, and how did early research networks evolve into the global system we use today?",
    "What is role of the cell membrane in regulating the molecular traffic across living cells, in everyday life, concretely?",
    "How does load balancer distribute traffic across multiple backend servers, and what algorithms are most commonly used today in practice?",
    "Why does aluminum form a protective oxide layer that prevents further corrosion, even in moist outdoor environments, in practice really?",
    "How do migratory butterflies like monarchs navigate over thousands of miles to a single forest, for the curious reader?",
    "How does public-key infrastructure rely on certificate authorities to establish trust online, and what limitations does this approach have really?",
    "How does hash function differ from encryption, and why are collisions so important to avoid in modern cryptographic systems today?",
    "How does heat pump heat a home efficiently, even when outside temperatures are well below freezing, in everyday engineering practice?",
    "What is role of the appendix in modern human physiology, and what current evidence suggests about its function in the body?",
    "How do mosquitoes find hosts using carbon dioxide, body heat, and odor cues during flight, in modern biological research?",
    "How do whales coordinate group hunting techniques like bubble net feeding, and what does this say about cooperation in marine mammals?",
    "What is role of testosterone during human development, and why does its level matter so differently at various life stages now?",
]


# Use 1x8 shard specs for gpt-oss-20b until https://github.com/tenstorrent/tt-xla/issues/3490 is resolved.
# These specs match the fused decode setup: attention is TP-sharded on the model
# axis, while expert tensors are sharded only on the expert dimension.
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
        if getattr(layer.mlp.router, "bias", None) is not None:
            shard_specs[layer.mlp.router.bias] = (None,)
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


def test_gpt_oss_20b_tp_fused_decode(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    gpt_oss_fused_decode,
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
        num_layers=1,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
        decode_only=decode_only,
        mesh_config_fn=_gpt_oss_20b_mesh_config_fn,
        shard_spec_fn=_gpt_oss_20b_shard_spec_fn,
        trace_enabled=False,
        inject_custom_moe=True,
        custom_moe_cluster_axis=1,
        gpt_oss_fused_decode=gpt_oss_fused_decode,
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
        shard_specs[layer.mlp.router.weight] = (None, "batch")
        # This is a temporary sharding spec to enable gpt oss to not get OOM on galaxy.
        # Once the MoE module is refactored, this should be changed to EP 32.
        shard_specs[layer.mlp.experts.gate_up_proj] = ("model", "batch", None)
        shard_specs[layer.mlp.experts.gate_up_proj_bias] = ("model", None)
        shard_specs[layer.mlp.experts.down_proj] = ("model", None, "batch")
        shard_specs[layer.mlp.experts.down_proj_bias] = ("model", "batch")
        shard_specs[layer.input_layernorm.weight] = (None,)
        shard_specs[layer.post_attention_layernorm.weight] = (None,)

        # Fused MoE decode weights (SparseMOEGPT with preprocess_fused_weights=True).
        # Global shape: (ring_devices*12, mesh_cols, E_per_dev, groups, K+32, 128).
        # ring_devices is the "batch" axis (rows=4 on 4x8 galaxy) and mesh_cols is
        # the "model" axis (cols=8). Per-device shape becomes (12, 1, ...), which
        # matches what MoeGpt kernel expects (dim 0 == num DRAM-bank matmul cores).
        mlp = layer.mlp
        if hasattr(mlp, "fused_w0_w1") and hasattr(mlp, "fused_w2"):
            shard_specs[mlp.fused_w0_w1] = ("batch", "model", None, None, None, None)
            shard_specs[mlp.fused_w2] = ("batch", "model", None, None, None, None)

    return shard_specs


def test_gpt_oss_120b_tp_dp_galaxy_fused_decode_batch_size_128(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    gpt_oss_fused_decode,
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
        inject_custom_moe=True,
        custom_moe_cluster_axis=0,
        gpt_oss_fused_decode=gpt_oss_fused_decode,
        preprocess_fused_weights=True,
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
            "model.layers.*.mlp.router.weight": "bfp_bf4",
            "model.layers.*.gate_up_proj": "bfp_bf4",
            "model.layers.*.down_proj": "bfp_bf4",
        },
        required_pcc=0.93,  # set for now as it's ~0.93 on test runs locally
        mesh_config_fn=_gpt_oss_120b_qb2_mesh_config_fn,
        shard_spec_fn=_gpt_oss_120b_qb2_shard_spec_fn,
        per_user_prompts=None if accuracy_testing else _GPT_OSS_PER_USER_QUESTIONS,
    )
