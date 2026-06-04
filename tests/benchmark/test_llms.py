# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import time

import numpy as np
import pytest
from benchmarks.llm_benchmark import (
    benchmark_llm_torch_xla,
    benchmark_llm_torch_xla_prefill,
)
from llm_utils.token_accuracy import TokenAccuracy
from loguru import logger
from utils import create_model_loader, resolve_display_name

# Defaults for all llms
DEFAULT_OPTIMIZATION_LEVEL = 2
DEFAULT_TP_OPTIMIZATION_LEVEL = 2
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


def _apply_ttnn_perf_summary(results, perf_file, num_graphs):
    """Copy a TTNN perf-metrics JSON's summary into results["config"].

    Reads the per-graph metrics file produced by the compile backend and maps its
    "summary" block onto the ttnn_* config fields. Always records ttnn_num_graphs.
    """
    with open(perf_file, "r") as f:
        perf_metrics_data = json.load(f)

    summary = perf_metrics_data.get("summary")
    if isinstance(summary, dict):
        config = results["config"]
        config["ttnn_total_ops"] = summary.get("total_ops", 0)
        config["ttnn_total_shardable_ops"] = summary.get("total_shardable_ops", 0)
        config["ttnn_effectively_sharded_ops"] = summary.get(
            "effectively_sharded_ops", 0
        )
        config["ttnn_system_memory_ops"] = summary.get("system_memory_ops", 0)
        config["ttnn_effectively_sharded_percentage"] = summary.get(
            "effectively_sharded_percentage", 0.0
        )
    results["config"]["ttnn_num_graphs"] = num_graphs


def _make_perf_metrics_dir(request):
    """Return a fresh, per-run directory for the backend's TTNN perf-metrics JSON files.

    The compile backend writes one ``<base>_<i>.json`` per graph. Pointing each run at its
    own directory keeps the later glob from picking up leftovers from a previous run in the
    same working directory, which would otherwise inflate ttnn_num_graphs or feed stale
    metrics into the results. Prefers pytest's tmp_path (unique per test node, auto-cleaned);
    falls back to a pid/time-stamped dir when no request fixture is available.
    """
    if request is not None:
        base = str(request.getfixturevalue("tmp_path"))
    else:
        base = os.path.join(
            ".", f"perf_metrics_{os.getpid()}_{int(time.time() * 1000)}"
        )
    perf_dir = os.path.join(base, "ttnn_perf_metrics")
    os.makedirs(perf_dir, exist_ok=True)
    return perf_dir


def _list_perf_files(perf_dir, base_name):
    """Sorted full paths of this run's perf-metrics JSONs in perf_dir."""
    files = [
        os.path.join(perf_dir, f)
        for f in os.listdir(perf_dir)
        if f.startswith(base_name) and f.endswith(".json")
    ]
    return sorted(files)


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
    required_pcc=DEFAULT_REQUIRED_PCC,
    fp32_dest_acc_en=None,
    experimental_kv_cache_dtype=None,
    num_layers=None,
    request=None,
    accuracy_testing: bool = False,
    max_output_tokens=None,
    decode_only: bool = False,
    weight_dtype_overrides: dict = None,
    input_output_sharding_spec=None,
    kv_cache_sharding_spec=None,
    use_mla_cache: bool = False,
    expected_ops: list = None,
    check_fusions: bool = False,
    use_indexer_cache: bool = False,
    enable_create_d2m_subgraphs: bool = False,
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

    perf_dir = _make_perf_metrics_dir(request)
    ttnn_perf_metrics_output_file = os.path.join(
        perf_dir, f"tt_xla_{display_name}_perf_metrics"
    )

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
    experimental_kv_cache_dtype={experimental_kv_cache_dtype}
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
        required_pcc=required_pcc,
        fp32_dest_acc_en=fp32_dest_acc_en,
        experimental_kv_cache_dtype=experimental_kv_cache_dtype,
        accuracy_testing=accuracy_testing,
        model_name_for_accuracy=model_name_for_accuracy,
        hf_model_name_for_accuracy=hf_model_name,
        max_output_tokens=max_output_tokens,
        decode_only=decode_only,
        weight_dtype_overrides=weight_dtype_overrides,
        input_output_sharding_spec=input_output_sharding_spec,
        kv_cache_sharding_spec=kv_cache_sharding_spec,
        use_mla_cache=use_mla_cache,
        expected_ops=expected_ops,
        check_fusions_enabled=check_fusions,
        use_indexer_cache=use_indexer_cache,
        enable_create_d2m_subgraphs=enable_create_d2m_subgraphs,
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_info_name

        # LLM-specific perf metrics handling: Use only decode graph (second file)
        # LLMs split into 2 graphs: prefill (index 0) and decode (index 1)
        # Only decode is relevant for throughput. perf_dir is unique to this run, so the
        # glob only ever sees this run's files (no pollution from previous runs).
        base_name = os.path.basename(ttnn_perf_metrics_output_file)
        perf_files = _list_perf_files(perf_dir, base_name)

        if len(perf_files) == 2:
            # Use only the decode graph (second file)
            decode_perf_file = perf_files[1]
            print(f"Using decode graph perf metrics from: {decode_perf_file}")
            _apply_ttnn_perf_summary(results, decode_perf_file, num_graphs=2)
        else:
            logger.warning(
                f"Expected 2 perf metrics files (prefill + decode) for LLM, but found {len(perf_files)}: {perf_files}. "
                f"Skipping perf metrics."
            )
            results["config"]["ttnn_num_graphs"] = len(perf_files)

        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)


def test_llm_prefill(
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
    experimental_weight_dtype=None,
    experimental_enable_permute_matmul_fusion=DEFAULT_EXPERIMENTAL_ENABLE_PERMUTE_MATMUL_FUSION,
    read_logits_fn=default_read_logits_fn,
    mesh_config_fn=None,
    shard_spec_fn=None,
    required_pcc=DEFAULT_REQUIRED_PCC,
    fp32_dest_acc_en=None,
    experimental_kv_cache_dtype=None,
    num_layers=None,
    request=None,
    weight_dtype_overrides: dict = None,
    input_output_sharding_spec=None,
    expected_ops: list = None,
    check_fusions: bool = False,
    use_indexer_cache: bool = False,
    enable_create_d2m_subgraphs: bool = False,
    do_apply_weight_dtype_overrides: bool = False,
    skip_pcc: bool = False,
):
    """Test the PREFILL of an LLM with the given variant and optional config overrides.

    Args:
        ModelLoaderModule: Model loader module/class used to instantiate the model.
        variant: Model variant identifier.
        output_file: Path to save benchmark results as JSON.
        optimization_level: tt-mlir optimization level (0, 1, or 2).
        trace_enabled: Enable device trace.
        batch_size: Batch size (rows the prefill prompt is broadcast across).
        loop_count: Number of benchmark iterations (must be 1).
        input_sequence_length: Input sequence length (number of prefill tokens).
        data_format: Data format (must be "bfloat16").
        task: Task type (must be "text-generation").
        experimental_weight_dtype: Weight dtype for block format conversion
            (e.g. "bfp_bf8", "bfp_bf4", or "" for none).
        experimental_enable_permute_matmul_fusion: Enable permute matmul fusion optimization.
        read_logits_fn: Function to extract logits from model output.
        mesh_config_fn: Optional callback returning the (mesh_shape, name) for multi-chip.
        shard_spec_fn: Optional callback returning model weight sharding specs for multi-chip.
        required_pcc: Required PCC threshold for prefill-logits validation.
        fp32_dest_acc_en: Optional fp32 dest accumulation compile flag.
        experimental_kv_cache_dtype: Optional KV-cache dtype compile flag.
        num_layers: Number of layers to override (None = full model).
        request: pytest request fixture, used to resolve the display name.
        weight_dtype_overrides: Explicit per-tensor weight dtype override dict.
        input_output_sharding_spec: Optional sharding spec for input_ids / output logits.
        expected_ops: Optional list of ops to assert on when check_fusions is set.
        check_fusions: Whether to run the fusion check against expected_ops.
        use_indexer_cache: Initialize per-layer Indexer caches before device transfer.
        enable_create_d2m_subgraphs: Enable D2M subgraph creation (requires opt level >= 1).
        do_apply_weight_dtype_overrides: Apply weight dtype overrides; left False for
            prefill since prefill runs in bf16.
        skip_pcc: Log the prefill PCC instead of asserting it (lets the run finish and
            still write its results JSON when PCC is below threshold).
    """
    # Resolve fixture values that arrive as None (unset CLI options) to their defaults.
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE
    if optimization_level is None:
        optimization_level = DEFAULT_OPTIMIZATION_LEVEL
    if input_sequence_length is None:
        input_sequence_length = DEFAULT_INPUT_SEQUENCE_LENGTH

    model_loader = create_model_loader(
        ModelLoaderModule, num_layers=num_layers, variant=variant
    )
    if num_layers is not None and model_loader is None:
        pytest.fail(
            "num_layers override requested but ModelLoader does not support it."
        )
    model_info_name = model_loader.get_model_info(variant=variant).name
    display_name = resolve_display_name(request=request, fallback=model_info_name)

    perf_dir = _make_perf_metrics_dir(request)
    ttnn_perf_metrics_output_file = os.path.join(
        perf_dir, f"tt_xla_{display_name}_perf_metrics"
    )

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
    experimental_kv_cache_dtype={experimental_kv_cache_dtype}
    required_pcc={required_pcc}
    num_layers={num_layers}
    ttnn_perf_metrics_output_file={ttnn_perf_metrics_output_file}
    """
    )

    results = benchmark_llm_torch_xla_prefill(
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
        required_pcc=required_pcc,
        fp32_dest_acc_en=fp32_dest_acc_en,
        experimental_kv_cache_dtype=experimental_kv_cache_dtype,
        weight_dtype_overrides=weight_dtype_overrides,
        input_output_sharding_spec=input_output_sharding_spec,
        expected_ops=expected_ops,
        check_fusions_enabled=check_fusions,
        use_indexer_cache=use_indexer_cache,
        enable_create_d2m_subgraphs=enable_create_d2m_subgraphs,
        do_apply_weight_dtype_overrides=do_apply_weight_dtype_overrides,  # prefill needs to be bf16
        skip_pcc=skip_pcc,
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_info_name

        # Prefill-only: there is no decode graph. One perf-metrics file is produced under
        # skip_pcc (perf graph only); otherwise two (perf graph + PCC logits graph). The
        # perf graph compiles first, so sorted()[0] is the timed graph — use it for the
        # TTNN op metrics. ttnn_num_graphs records how many were found. perf_dir is unique
        # to this run, so the glob only ever sees this run's files (no pollution from
        # previous runs).
        base_name = os.path.basename(ttnn_perf_metrics_output_file)
        perf_files = _list_perf_files(perf_dir, base_name)

        if perf_files:
            prefill_perf_file = perf_files[0]
            print(f"Using prefill graph perf metrics from: {prefill_perf_file}")
            _apply_ttnn_perf_summary(
                results, prefill_perf_file, num_graphs=len(perf_files)
            )
        else:
            logger.warning(
                "Expected at least 1 prefill perf metrics file for LLM, but found none. "
                "Skipping perf metrics."
            )
            results["config"]["ttnn_num_graphs"] = 0

        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)


# Generic driver invoked by the per-model prefill tests, not a test itself.
test_llm_prefill.__test__ = False


def test_llm_tp(
    ModelLoaderModule,
    variant,
    output_file,
    num_layers=None,
    request=None,
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
        num_layers=num_layers,
        request=request,
        decode_only=decode_only,
        required_pcc=required_pcc,
        **kwargs,
    )


def test_llm_prefill_tp(
    ModelLoaderModule,
    variant,
    output_file,
    num_layers=None,
    request=None,
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

    test_llm_prefill(
        ModelLoaderModule=ModelLoaderModule,
        variant=variant,
        output_file=output_file,
        optimization_level=optimization_level,
        mesh_config_fn=mesh_config_fn,
        shard_spec_fn=shard_spec_fn,
        num_layers=num_layers,
        request=request,
        required_pcc=required_pcc,
        **kwargs,
    )


# Generic driver invoked by the per-model TP prefill tests, not a test itself.
test_llm_prefill_tp.__test__ = False


def test_llama_3_2_1b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
    check_fusions,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
        expected_ops=[
            "ttnn.scaled_dot_product_attention",
            "ttnn.rms_norm",
        ],
        check_fusions=check_fusions,
    )


def test_llama_3_2_1b_prefill(
    output_file,
    num_layers,
    request,
    batch_size,
    optimization_level,
    input_sequence_length,
    skip_pcc,
):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.LLAMA_3_2_1B_INSTRUCT
    test_llm_prefill(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
        fp32_dest_acc_en=False,
        batch_size=batch_size,
        input_sequence_length=(
            input_sequence_length
            if input_sequence_length is not None
            else DEFAULT_INPUT_SEQUENCE_LENGTH
        ),
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
        experimental_weight_dtype=None,  # prefill needs to be bf16
        do_apply_weight_dtype_overrides=False,  # prefill needs to be bf16
        trace_enabled=False,
        skip_pcc=skip_pcc,
    )


def test_llama_3_2_3b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
    )


def test_gemma_1_1_2b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
    )


def test_gemma_2_2b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
    )


def test_phi1(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
    )


def test_phi1_prefill(
    output_file,
    num_layers,
    request,
    batch_size,
    optimization_level,
    input_sequence_length,
    skip_pcc,
):
    from third_party.tt_forge_models.phi1.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.PHI1
    test_llm_prefill(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
        fp32_dest_acc_en=False,
        batch_size=batch_size,
        input_sequence_length=(
            input_sequence_length
            if input_sequence_length is not None
            else DEFAULT_INPUT_SEQUENCE_LENGTH
        ),
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
        experimental_weight_dtype=None,  # prefill needs to be bf16
        do_apply_weight_dtype_overrides=False,  # prefill needs to be bf16
        trace_enabled=False,
        skip_pcc=skip_pcc,
    )


def test_phi1_5(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
    )


def test_phi2(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
    )


def test_falcon3_1b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
    )


def test_falcon3_3b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
    )


def test_qwen_2_5_0_5b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
    )


def test_qwen_3_0_6b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
    )


def test_qwen_3_1_7b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
    )


def test_qwen_3_4b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
    )


def test_qwen_2_5_1_5b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level if optimization_level is not None else 1
        ),
        required_pcc=0.90,
    )


def test_qwen_2_5_3b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
    )


def test_qwen_3_8b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
    )


def test_qwen_3_8b_prefill(
    output_file,
    num_layers,
    request,
    batch_size,
    optimization_level,
    input_sequence_length,
    skip_pcc,
):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.QWEN_3_8B
    test_llm_prefill(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
        fp32_dest_acc_en=False,
        batch_size=batch_size,
        input_sequence_length=(
            input_sequence_length
            if input_sequence_length is not None
            else DEFAULT_INPUT_SEQUENCE_LENGTH
        ),
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
        experimental_weight_dtype=None,  # prefill needs to be bf16
        do_apply_weight_dtype_overrides=False,  # prefill needs to be bf16
        trace_enabled=False,
        skip_pcc=skip_pcc,
    )


def test_qwen_2_5_7b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level if optimization_level is not None else 1
        ),
        required_pcc=0.90,
    )


# FAILED: KeyError: "L['self'].model.lifted_tensor_0"
def test_gemma_1_1_7b(
    output_file, num_layers, request, max_output_tokens, optimization_level
):
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
    )


# FAILED: TypeError: Phi3ForCausalLM.forward() got an unexpected keyword argument 'cache_position'
def test_phi3_mini(
    output_file, num_layers, request, max_output_tokens, optimization_level
):
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
    )


# FAILED: KeyError: 'lifted_tensor_0'
def test_phi3_5_mini(
    output_file, num_layers, request, max_output_tokens, optimization_level
):
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
    )


# FAILED: AttributeError: 'MambaConfig' object has no attribute 'num_attention_heads'
def test_mamba_2_8b(
    output_file, num_layers, request, max_output_tokens, optimization_level
):
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
    )


def test_falcon3_7b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
    )


def test_mistral_7b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
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
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
    )


def test_llama_3_1_8b(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
        required_pcc=0.90,
    )


def test_llama_3_1_8b_prefill(
    output_file,
    num_layers,
    request,
    batch_size,
    optimization_level,
    input_sequence_length,
    skip_pcc,
):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.LLAMA_3_1_8B_INSTRUCT
    test_llm_prefill(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
        fp32_dest_acc_en=False,
        batch_size=batch_size,
        input_sequence_length=(
            input_sequence_length
            if input_sequence_length is not None
            else DEFAULT_INPUT_SEQUENCE_LENGTH
        ),
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_OPTIMIZATION_LEVEL
        ),
        required_pcc=0.90,
        experimental_weight_dtype=None,  # prefill needs to be bf16
        do_apply_weight_dtype_overrides=False,  # prefill needs to be bf16
        trace_enabled=False,
        skip_pcc=skip_pcc,
    )


def test_falcon3_7b_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_TP_OPTIMIZATION_LEVEL
        ),
    )


def test_falcon3_10b_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_TP_OPTIMIZATION_LEVEL
        ),
    )


def test_llama_3_1_8b_instruct_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_TP_OPTIMIZATION_LEVEL
        ),
    )


def test_mistral_7b_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_TP_OPTIMIZATION_LEVEL
        ),
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
    optimization_level,
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
        optimization_level=1,
    )


def test_mistral_nemo_instruct_2407_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=1,
    )


def test_mistral_small_24b_instruct_2501_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=1,  # flaky: occasionally hangs in CI with optimization_level=2
    )


def test_qwen_2_5_14b_instruct_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=1,
    )


def test_qwen_2_5_32b_instruct_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_TP_OPTIMIZATION_LEVEL
        ),
    )


def test_qwen_2_5_coder_32b_instruct_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=1,
    )


def test_qwen_3_0_6b_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_TP_OPTIMIZATION_LEVEL
        ),
    )


def test_qwen_3_1_7b_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_TP_OPTIMIZATION_LEVEL
        ),
    )


def test_qwen_3_8b_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=1,  # flaky: occasionally hangs in CI with optimization_level=2
    )


def test_qwen_3_14b_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=1,
    )


def test_qwen_3_32b_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_TP_OPTIMIZATION_LEVEL
        ),
    )


def test_llama_3_8b_instruct_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_TP_OPTIMIZATION_LEVEL
        ),
    )


def test_llama_3_1_8b_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_TP_OPTIMIZATION_LEVEL
        ),
    )


def test_llama_3_8b_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=(
            optimization_level
            if optimization_level is not None
            else DEFAULT_TP_OPTIMIZATION_LEVEL
        ),
    )


def test_llama_3_1_70b_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=1,  # flaky: occasionally hangs in CI with optimization_level=2
    )


def test_llama_3_1_70b_tp_prefill(
    output_file,
    num_layers,
    request,
    batch_size,
    optimization_level,
    input_sequence_length,
    skip_pcc,
):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.LLAMA_3_1_70B_INSTRUCT
    test_llm_prefill_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=num_layers,
        request=request,
        fp32_dest_acc_en=False,
        batch_size=batch_size,
        input_sequence_length=(
            input_sequence_length
            if input_sequence_length is not None
            else DEFAULT_INPUT_SEQUENCE_LENGTH
        ),
        optimization_level=(
            optimization_level if optimization_level is not None else 1
        ),  # flaky: occasionally hangs in CI with optimization_level=2
        required_pcc=0.90,
        experimental_weight_dtype=None,  # prefill needs to be bf16
        do_apply_weight_dtype_overrides=False,  # prefill needs to be bf16
        trace_enabled=False,
        skip_pcc=skip_pcc,
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


# Trace disabled: ~23% slower with trace on bs=32 (https://github.com/tenstorrent/tt-xla/issues/4192)
def test_gpt_oss_20b_tp(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=1,
    )


# Test with D2M fusion enabled (enable-create-d2m-subgraphs=true).
def test_gpt_oss_20b_tp_d2m(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=1,
        enable_create_d2m_subgraphs=True,
    )


def test_gpt_oss_20b_tp_batch_size_1(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=1,
    )


def test_llama_3_1_70b_tp_galaxy(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=1,
    )


def test_gpt_oss_20b_tp_galaxy_batch_size_64(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=1,
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
    optimization_level,
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
        optimization_level=1,
        mesh_config_fn=_galaxy_mesh_config_fn,
        shard_spec_fn=_moe_throughput_galaxy_shard_spec_fn,
        input_output_sharding_spec=("batch", None),
        kv_cache_sharding_spec=("batch", "model", None, None),
        trace_enabled=True,
    )


def test_gpt_oss_120b_tp_galaxy_batch_size_64(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        batch_size=batch_size if batch_size is not None else 64,
        optimization_level=1,
        mesh_config_fn=_galaxy_mesh_config_fn,
        shard_spec_fn=_moe_throughput_galaxy_shard_spec_fn,
        input_output_sharding_spec=("batch", None),
        kv_cache_sharding_spec=("batch", "model", None, None),
        trace_enabled=True,
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
    optimization_level,
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
    )


# Trace disabled: topk i64 indices can't reside in device DRAM inside capture_or_execute_trace
# This test only runs 2 layers so we expect to see incoherent output
def test_kimi_k2_tp_galaxy_2_layers(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
):
    from third_party.tt_forge_models.kimi_k2.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.KIMI_K2_INSTRUCT_MODIFIED
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=2,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=64,  # Test hangs for a batch size of 128 - Issue: https://github.com/tenstorrent/tt-xla/issues/4565
        max_output_tokens=max_output_tokens,
        decode_only=decode_only,
        input_output_sharding_spec=("batch", None),
        use_mla_cache=True,
        optimization_level=0,
        trace_enabled=False,
    )


# Trace disabled: topk i64 indices can't reside in device DRAM inside capture_or_execute_trace
# This test only runs 2 layers so we expect to see incoherent output
def test_kimi_k2_5_tp_galaxy_2_layers(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
):
    from third_party.tt_forge_models.kimi_k2.k2_5.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.KIMI_K2_5_MODIFIED
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=2,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=64,  # Test hangs for a batch size of 128 - Issue: https://github.com/tenstorrent/tt-xla/issues/4565
        max_output_tokens=max_output_tokens,
        decode_only=decode_only,
        input_output_sharding_spec=("batch", None),
        use_mla_cache=True,
        optimization_level=0,
        trace_enabled=False,
    )


# This test only runs 2 layers so we expect to see incoherent output
def test_deepseek_v3_2_exp_tp_galaxy_2_layers(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
):
    from third_party.tt_forge_models.deepseek.deepseek_v3_2_exp.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.DEEPSEEK_V3_2_EXP_MODIFIED
    test_llm_tp(
        ModelLoader,
        variant,
        output_file,
        num_layers=2,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=128,
        max_output_tokens=max_output_tokens,
        decode_only=decode_only,
        input_output_sharding_spec=("batch", None),
        use_mla_cache=True,
        use_indexer_cache=True,
        optimization_level=0,
        trace_enabled=False,
        required_pcc=-0.92,
    )


def test_falcon3_7b_tp_qb2(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=2,
    )


def test_falcon3_10b_tp_qb2(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=2,
    )


def test_llama_3_1_8b_instruct_tp_qb2(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=2,
    )


def test_ministral_8b_tp_qb2(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=2,
    )


def test_mistral_nemo_instruct_2407_tp_qb2(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=2,
    )


def test_mistral_small_24b_instruct_2501_tp_qb2(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=2,
    )


def test_qwen_2_5_14b_instruct_tp_qb2(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=2,
    )


def test_qwen_2_5_coder_32b_instruct_tp_qb2(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=2,
    )


def test_qwen_3_8b_tp_qb2(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=2,
    )


def test_qwen_3_14b_tp_qb2(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=2,
    )


def test_qwen_3_32b_tp_qb2(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=2,
    )


def test_gpt_oss_20b_tp_qb2(
    output_file,
    num_layers,
    request,
    accuracy_testing,
    batch_size,
    max_output_tokens,
    decode_only,
    optimization_level,
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
        optimization_level=2,
    )
