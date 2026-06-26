# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import pytest
from benchmarks.llm_benchmark import (
    AccuracyConfig,
    CompileConfig,
    PccMode,
    ShardingConfig,
    benchmark_llm_torch_xla,
)
from llm_utils.token_accuracy import TokenAccuracy
from model_utils import create_model_loader
from naming import perf_metrics_filename, resolve_display_name
from reporting import aggregate_llm_decode_perf, write_benchmark_json

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
DEFAULT_EXPERIMENTAL_KV_CACHE_DTYPE = "bfp_bf8"
DEFAULT_EXPERIMENTAL_ENABLE_PERMUTE_MATMUL_FUSION = False
DEFAULT_REQUIRED_PCC = 0.94


def default_read_logits_fn(output):
    return output.logits


# Sentinel for "argument not supplied" so helpers can distinguish a forced
# value (which ignores the corresponding CLI option) from "fall back to the
# CLI option, else a per-test default".
_UNSET = object()


def _resolve_opt(optimization_level, default=DEFAULT_OPTIMIZATION_LEVEL):
    """CLI --optimization-level if given, else the per-test default."""
    return optimization_level if optimization_level is not None else default


def _run_llm(
    ModelLoaderModule,
    variant,
    cli,
    request,
    *,
    optimization_level=_UNSET,
    default_optimization_level=DEFAULT_OPTIMIZATION_LEVEL,
    num_layers=_UNSET,
    default_num_layers=None,
    batch_size=_UNSET,
    default_batch_size=None,
    **overrides,
):
    """Run a single-device LLM benchmark from the bundled CLI options.

    ``optimization_level`` / ``num_layers`` / ``batch_size`` may be passed
    explicitly to *force* a value (ignoring the corresponding CLI option), as
    some models require a fixed config. Otherwise the CLI option wins, falling
    back to the matching ``default_*``. Everything else flows through
    ``overrides`` to :func:`_benchmark_llm`.
    """
    opt = (
        optimization_level
        if optimization_level is not _UNSET
        else _resolve_opt(cli.optimization_level, default_optimization_level)
    )
    nl = (
        num_layers
        if num_layers is not _UNSET
        else (cli.num_layers if cli.num_layers is not None else default_num_layers)
    )
    bs = (
        batch_size
        if batch_size is not _UNSET
        else (cli.batch_size if cli.batch_size is not None else default_batch_size)
    )
    _benchmark_llm(
        ModelLoaderModule=ModelLoaderModule,
        variant=variant,
        output_file=cli.output_file,
        num_layers=nl,
        request=request,
        accuracy_testing=cli.accuracy_testing,
        batch_size=bs,
        max_output_tokens=cli.max_output_tokens,
        decode_only=cli.decode_only,
        optimization_level=opt,
        pcc_mode=PccMode.from_options(
            pcc_only=cli.pcc_only,
            pcc_prefill=cli.pcc_prefill,
            pcc_decode=cli.pcc_decode,
        ),
        **overrides,
    )


def _run_llm_tp(
    ModelLoaderModule,
    variant,
    cli,
    request,
    *,
    mesh_config_fn=_UNSET,
    shard_spec_fn=_UNSET,
    **kwargs,
):
    """Tensor-parallel variant of :func:`_run_llm`.

    Resolves the mesh / shard-spec functions from the loader class (or the
    per-test overrides) and defaults the optimization level to the TP default.
    """
    resolved_mesh = (
        mesh_config_fn
        if (mesh_config_fn is not _UNSET and mesh_config_fn is not None)
        else getattr(ModelLoaderModule, "get_mesh_config", None)
    )
    resolved_shard = (
        shard_spec_fn
        if shard_spec_fn is not _UNSET
        else getattr(ModelLoaderModule, "load_shard_spec", None)
    )
    kwargs.setdefault("default_optimization_level", DEFAULT_TP_OPTIMIZATION_LEVEL)
    _run_llm(
        ModelLoaderModule,
        variant,
        cli,
        request,
        mesh_config_fn=resolved_mesh,
        shard_spec_fn=resolved_shard,
        **kwargs,
    )


# =========================================================================== #
# Tensor-parallel mesh + weight-sharding specs                                #
# --------------------------------------------------------------------------- #
# TRANSIENT SCAFFOLDING - move these to forge-models once they are stable.
#
# A shard spec is a ``{parameter_tensor: axis_tuple}`` map consumed by
# ``xs.mark_sharding`` (see ``benchmarks/llm_benchmark.py``); a mesh fn returns
# ``((rows, cols), (axis_names))``. These live here, inline next to the TP tests,
# precisely as a reminder that they are temporary: once a model's sharding lands
# in its tt-forge-models loader as ``load_shard_spec`` (and ``get_mesh_config``),
# the per-model spec below is deleted and the TP test falls back to the loader's.
#
# Each per-model spec is a short, self-contained function composing the reusable
# *primitives* below (the only durable part) - so adding a model is "copy one and
# tweak the axes / module paths" and removing one is "delete the function".
# Axis names (``"model"`` / ``"batch"``) refer to the mesh axes from the mesh fn.
# =========================================================================== #


# --- reusable parallelism primitives --------------------------------------- #
def megatron_attention(shard_specs, attn, axis="model", *, bias=False, qk_norm=False):
    """Megatron-style column->row tensor parallelism for an attention block.

    q/k/v projections are column-parallel (shard output features along ``axis``)
    and the output projection is row-parallel, so the only collective is a single
    all-reduce after o_proj. Optionally shards q/k/v biases and replicates the
    q/k RMS norms when present.
    """
    shard_specs[attn.q_proj.weight] = (axis, None)
    shard_specs[attn.k_proj.weight] = (axis, None)
    shard_specs[attn.v_proj.weight] = (axis, None)
    shard_specs[attn.o_proj.weight] = (None, axis)
    if bias and attn.q_proj.bias is not None:
        shard_specs[attn.q_proj.bias] = (axis,)
        shard_specs[attn.k_proj.bias] = (axis,)
        shard_specs[attn.v_proj.bias] = (axis,)
    if qk_norm and hasattr(attn, "q_norm"):
        shard_specs[attn.q_norm.weight] = (None,)
        shard_specs[attn.k_norm.weight] = (None,)


def mla_attention(shard_specs, attn, axis="model"):
    """Tensor parallelism for multi-head latent attention (DeepSeek / Kimi).

    The low-rank down-projections (``q_a`` / ``kv_a``) are row-parallel and the
    up-projections (``q_b`` / ``kv_b``) are column-parallel along ``axis``, with a
    row-parallel output projection - the MLA analogue of column->row attention.
    """
    shard_specs[attn.q_a_proj.weight] = (None, axis)
    shard_specs[attn.q_b_proj.weight] = (axis, None)
    shard_specs[attn.kv_a_proj_with_mqa.weight] = (None, axis)
    shard_specs[attn.kv_b_proj.weight] = (axis, None)
    shard_specs[attn.o_proj.weight] = (None, axis)


def fused_gate_up_experts(
    shard_specs,
    experts,
    *,
    gate_up=("model", None, None),
    down=("model", None, None),
    gate_up_bias=("model", None),
    down_bias=("model", None),
):
    """Shard a MoE block whose experts fuse gate+up into one ``gate_up_proj``.

    Defaults give plain expert-parallelism along the model axis; pass ``gate_up``
    / ``down`` / ``down_bias`` to additionally shard experts along the batch axis
    (the galaxy throughput layout).
    """
    shard_specs[experts.gate_up_proj] = gate_up
    shard_specs[experts.gate_up_proj_bias] = gate_up_bias
    shard_specs[experts.down_proj] = down
    shard_specs[experts.down_proj_bias] = down_bias


def routed_experts(shard_specs, experts, *, expert_axes=("batch", "model"), bias=False):
    """Shard routed (all-to-all) experts with separate gate/up/down projections.

    Used by DeepSeek V3.x / GLM / Kimi-style MoE: each of gate/up/down is sharded
    across both mesh axes (EP = rows*cols), i.e. ``(expert_axes, None, None)``.
    Optionally shards the matching expert biases.
    """
    spec = (expert_axes, None, None)
    shard_specs[experts.gate_proj] = spec
    shard_specs[experts.up_proj] = spec
    shard_specs[experts.down_proj] = spec
    if bias:
        for name in ("gate_proj_bias", "up_proj_bias", "down_proj_bias"):
            b = getattr(experts, name, None)
            if b is not None:
                shard_specs[b] = (expert_axes, None)


# --- mesh topologies ------------------------------------------------------- #
def single_row_mesh(model_loader, num_devices):
    """1xN mesh: DP=1, every device on the model (TP) axis - i.e. pure TP.

    Used for gpt-oss-20b at 1x8 until
    https://github.com/tenstorrent/tt-xla/issues/3490 is resolved.
    """
    return (1, num_devices), ("batch", "model")


def galaxy_4x8_mesh(model_loader, num_devices):
    """4x8 wormhole_galaxy mesh (DP=4, TP=8)."""
    if num_devices != 32:
        raise ValueError("wormhole_galaxy benchmarks expect 32 devices (4x8 mesh).")
    return (4, 8), ("batch", "model")


def qb2_1x4_mesh(model_loader, num_devices):
    """1x4 QB2 mesh: DP=1, TP=4."""
    return (1, 4), ("batch", "model")


# --- per-model shard specs ------------------------------------------------- #
def gpt_oss_20b_shard_spec(model_loader, model):
    shard_specs = {}
    for layer in model.model.layers:
        megatron_attention(shard_specs, layer.self_attn)
        shard_specs[layer.self_attn.sinks] = (None,)
        shard_specs[layer.mlp.router.weight] = (None, None)
        fused_gate_up_experts(shard_specs, layer.mlp.experts)
    return shard_specs


def gpt_oss_120b_galaxy_shard_spec(model_loader, model):
    """gpt-oss-120b throughput layout on the 4x8 galaxy mesh.
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
        megatron_attention(shard_specs, layer.self_attn)
        shard_specs[layer.self_attn.sinks] = ("model",)
        shard_specs[layer.mlp.router.weight] = (None, None)
        # This is a temporary sharding spec to enable gpt oss to not get OOM on galaxy.
        # Once the MoE module is refactored, this should be changed to EP 32.
        fused_gate_up_experts(
            shard_specs,
            layer.mlp.experts,
            gate_up=("model", "batch", None),
            down=("model", None, "batch"),
            down_bias=("model", "batch"),
        )
        shard_specs[layer.input_layernorm.weight] = (None,)
        shard_specs[layer.post_attention_layernorm.weight] = (None,)

    return shard_specs


def gpt_oss_120b_qb2_shard_spec(model_loader, model):
    """gpt-oss-120b on the 1x4 QB2 mesh — model-axis-only, no batch"""
    shard_specs = {}
    shard_specs[model.model.embed_tokens.weight] = (None, None)
    shard_specs[model.model.norm.weight] = (None,)

    for layer in model.model.layers:
        megatron_attention(shard_specs, layer.self_attn)
        shard_specs[layer.self_attn.sinks] = (None,)
        fused_gate_up_experts(shard_specs, layer.mlp.experts)
    return shard_specs


def deepseek_v3_1_shard_spec(model_loader, model):
    """DeepSeek V3.1 on the 4x8 galaxy mesh (TP 8, DP 4, EP 32).

    Hidden dim sharded along the model axis: MLA attention, model-sharded norms,
    vocab-parallel embed/lm_head, routed (+ shared) experts across both axes. The
    leading dense layers fall through to the dense branch.
    """
    from tt_torch.sparse_mlp import A2aSparseMLPWithSharedExperts

    shard_specs = {}
    shard_specs[model.model.embed_tokens.weight] = (None, "model")
    shard_specs[model.model.norm.weight] = ("model",)
    shard_specs[model.lm_head.weight] = (None, "model")

    for layer in model.model.layers:
        mla_attention(shard_specs, layer.self_attn)
        shard_specs[layer.input_layernorm.weight] = ("model",)
        shard_specs[layer.post_attention_layernorm.weight] = ("model",)

        mlp = layer.mlp
        if isinstance(mlp, A2aSparseMLPWithSharedExperts):
            inner = mlp.mlp if hasattr(mlp, "mlp") else mlp
            shard_specs[inner.router.gate.weight] = (None, "model")
            routed_experts(shard_specs, inner.experts, bias=True)

            shared = getattr(mlp, "shared_experts", None)
            if shared is not None:
                shard_specs[shared.gate_proj.weight] = (None, "model")
                shard_specs[shared.up_proj.weight] = (None, "model")
                shard_specs[shared.down_proj.weight] = ("model", None)
        else:
            shard_specs[mlp.gate_proj.weight] = ("batch", "model")
            shard_specs[mlp.up_proj.weight] = ("batch", "model")
            shard_specs[mlp.down_proj.weight] = ("model", "batch")

    return shard_specs


def glm_4_7_shard_spec(model_loader, model):
    """GLM-4 on the 4x8 galaxy mesh (TP 8, DP 4, EP 32), hidden replicated.

    Hidden dim kept replicated so RMS norms reduce locally instead of lowering to
    a distributed all_gather. Embedding replicated, lm_head vocab-parallel,
    attention / dense MLP / shared experts col->row along the model axis; routed
    experts across both axes (EP 32).
    """
    from tt_torch.sparse_mlp import A2aSparseMLPWithSharedExperts

    shard_specs = {}
    shard_specs[model.model.embed_tokens.weight] = (None, None)
    shard_specs[model.model.norm.weight] = (None,)
    shard_specs[model.lm_head.weight] = ("model", None)

    for layer in model.model.layers:
        shard_specs[layer.input_layernorm.weight] = (None,)
        shard_specs[layer.post_attention_layernorm.weight] = (None,)

        megatron_attention(shard_specs, layer.self_attn, bias=True, qk_norm=True)

        mlp = layer.mlp
        if isinstance(mlp, A2aSparseMLPWithSharedExperts):
            inner = mlp.mlp
            shard_specs[inner.router.gate.weight] = (None, None)
            routed_experts(shard_specs, inner.experts)

            shared = getattr(mlp, "shared_experts", None)
            if shared is not None:
                shard_specs[shared.gate_proj.weight] = ("model", None)
                shard_specs[shared.up_proj.weight] = ("model", None)
                shard_specs[shared.down_proj.weight] = (None, "model")
        else:
            shard_specs[mlp.gate_proj.weight] = ("model", None)
            shard_specs[mlp.up_proj.weight] = ("model", None)
            shard_specs[mlp.down_proj.weight] = (None, "model")

    return shard_specs


def _benchmark_llm(
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
    experimental_kv_cache_dtype=DEFAULT_EXPERIMENTAL_KV_CACHE_DTYPE,
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
    experts_implementation: Optional[str] = None,
    pcc_mode: Optional[PccMode] = None,
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
        expert_implementation: Expert implementation type
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

    ttnn_perf_metrics_output_file = perf_metrics_filename(display_name)

    print(f"Running LLM benchmark for variant: {variant}")
    print(f"""Configuration:
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
    """)

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
        model_loader=model_loader,
        model_variant=variant,
        display_name=display_name,
        batch_size=batch_size,
        loop_count=loop_count,
        task=task,
        data_format=data_format,
        input_sequence_length=input_sequence_length,
        ttnn_perf_metrics_output_file=ttnn_perf_metrics_output_file,
        read_logits_fn=read_logits_fn,
        required_pcc=required_pcc,
        compile_config=CompileConfig(
            optimization_level=optimization_level,
            trace_enabled=trace_enabled,
            experimental_weight_dtype=experimental_weight_dtype,
            experimental_enable_permute_matmul_fusion=experimental_enable_permute_matmul_fusion,
            fp32_dest_acc_en=fp32_dest_acc_en,
            experimental_kv_cache_dtype=experimental_kv_cache_dtype,
            enable_create_d2m_subgraphs=enable_create_d2m_subgraphs,
        ),
        sharding_config=ShardingConfig(
            mesh_config_fn=mesh_config_fn,
            shard_spec_fn=shard_spec_fn,
            input_output_sharding_spec=input_output_sharding_spec,
            kv_cache_sharding_spec=kv_cache_sharding_spec,
        ),
        accuracy_config=AccuracyConfig(
            enabled=accuracy_testing,
            model_name_for_accuracy=model_name_for_accuracy,
            hf_model_name_for_accuracy=hf_model_name,
        ),
        max_output_tokens=max_output_tokens,
        decode_only=decode_only,
        weight_dtype_overrides=weight_dtype_overrides,
        use_mla_cache=use_mla_cache,
        expected_ops=expected_ops,
        check_fusions_enabled=check_fusions,
        use_indexer_cache=use_indexer_cache,
        experts_implementation=experts_implementation,
        pcc_mode=pcc_mode,
    )

    if output_file:
        # LLMs emit a prefill + decode graph; only the decode graph drives
        # steady-state throughput, so fold just that one into the result.
        aggregate_llm_decode_perf(ttnn_perf_metrics_output_file, results)
        write_benchmark_json(results, output_file, model_rawname=model_info_name)


def test_llama_3_2_1b(cli, request):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm(
        ModelLoader,
        ModelVariant.LLAMA_3_2_1B_INSTRUCT,
        cli,
        request,
        expected_ops=[
            "ttnn.scaled_dot_product_attention",
            "ttnn.rms_norm",
        ],
        check_fusions=cli.check_fusions,
    )


def test_llama_3_2_3b(cli, request):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm(
        ModelLoader,
        ModelVariant.LLAMA_3_2_3B_INSTRUCT,
        cli,
        request,
        experimental_kv_cache_dtype=None,
    )


def test_gemma_1_1_2b(cli, request):
    from third_party.tt_forge_models.gemma.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm(ModelLoader, ModelVariant.GEMMA_1_1_2B_IT, cli, request)


def test_gemma_2_2b(cli, request):
    from third_party.tt_forge_models.gemma.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm(
        ModelLoader,
        ModelVariant.GEMMA_2_2B_IT,
        cli,
        request,
        experimental_kv_cache_dtype=None,
    )


def test_phi1(cli, request):
    from third_party.tt_forge_models.phi1.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm(
        ModelLoader, ModelVariant.PHI1, cli, request, experimental_kv_cache_dtype=None
    )


def test_phi1_5(cli, request):
    from third_party.tt_forge_models.phi1_5.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm(
        ModelLoader, ModelVariant.PHI1_5, cli, request, experimental_kv_cache_dtype=None
    )


def test_phi2(cli, request):
    from third_party.tt_forge_models.phi2.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm(
        ModelLoader, ModelVariant.PHI2, cli, request, experimental_kv_cache_dtype=None
    )


def test_falcon3_1b(cli, request):
    from third_party.tt_forge_models.falcon.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    # Tuple format: (logits, past_key_values, ...)
    _run_llm(
        ModelLoader,
        ModelVariant.FALCON_1B,
        cli,
        request,
        read_logits_fn=lambda output: output[0],
    )


def test_falcon3_3b(cli, request):
    from third_party.tt_forge_models.falcon.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    # Tuple format: (logits, past_key_values, ...)
    _run_llm(
        ModelLoader,
        ModelVariant.FALCON_3B,
        cli,
        request,
        read_logits_fn=lambda output: output[0],
    )


def test_qwen_2_5_0_5b(cli, request):
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm(
        ModelLoader,
        ModelVariant.QWEN_2_5_0_5B_INSTRUCT,
        cli,
        request,
        required_pcc=0.94,
        experimental_kv_cache_dtype=None,
    )


def test_qwen_3_0_6b(cli, request):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm(ModelLoader, ModelVariant.QWEN_3_0_6B, cli, request)


def test_qwen_3_1_7b(cli, request):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm(
        ModelLoader,
        ModelVariant.QWEN_3_1_7B,
        cli,
        request,
        experimental_kv_cache_dtype=None,
    )


def test_qwen_3_4b(cli, request):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm(
        ModelLoader,
        ModelVariant.QWEN_3_4B,
        cli,
        request,
        experimental_kv_cache_dtype=None,
    )


def test_qwen_2_5_1_5b(cli, request):
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm(
        ModelLoader,
        ModelVariant.QWEN_2_5_1_5B_INSTRUCT,
        cli,
        request,
        default_optimization_level=1,
        required_pcc=0.90,
        experimental_kv_cache_dtype=None,
    )


def test_qwen_2_5_3b(cli, request):
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm(
        ModelLoader,
        ModelVariant.QWEN_2_5_3B_INSTRUCT,
        cli,
        request,
        experimental_kv_cache_dtype=None,
    )


def test_qwen_3_8b(cli, request):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm(ModelLoader, ModelVariant.QWEN_3_8B, cli, request)


def test_qwen_2_5_7b(cli, request):
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm(
        ModelLoader,
        ModelVariant.QWEN_2_5_7B_INSTRUCT,
        cli,
        request,
        default_optimization_level=1,
        required_pcc=0.90,
    )


# FAILED: KeyError: "L['self'].model.lifted_tensor_0"
def test_gemma_1_1_7b(cli, request):
    from third_party.tt_forge_models.gemma.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm(ModelLoader, ModelVariant.GEMMA_1_1_7B_IT, cli, request)


# FAILED: TypeError: Phi3ForCausalLM.forward() got an unexpected keyword argument 'cache_position'
def test_phi3_mini(cli, request):
    from third_party.tt_forge_models.phi3.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm(ModelLoader, ModelVariant.MINI_4K, cli, request)


# FAILED: KeyError: 'lifted_tensor_0'
def test_phi3_5_mini(cli, request):
    from third_party.tt_forge_models.phi3.phi_3_5.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm(ModelLoader, ModelVariant.MINI_INSTRUCT, cli, request)


# FAILED: AttributeError: 'MambaConfig' object has no attribute 'num_attention_heads'
def test_mamba_2_8b(cli, request):
    from third_party.tt_forge_models.mamba.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm(ModelLoader, ModelVariant.MAMBA_2_8B, cli, request)


def test_falcon3_7b(cli, request):
    from third_party.tt_forge_models.falcon.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    # Tuple format: (logits, past_key_values, ...)
    _run_llm(
        ModelLoader,
        ModelVariant.FALCON_7B,
        cli,
        request,
        read_logits_fn=lambda output: output[0],
    )


def test_mistral_7b(cli, request):
    from third_party.tt_forge_models.mistral.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm(ModelLoader, ModelVariant.MISTRAL_7B_INSTRUCT_V03, cli, request)


# Trace disabled: host/device tensor shape mismatch (https://github.com/tenstorrent/tt-xla/issues/3934)
def test_ministral_8b(cli, request):
    from third_party.tt_forge_models.mistral.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm(
        ModelLoader,
        ModelVariant.MINISTRAL_8B,
        cli,
        request,
        fp32_dest_acc_en=False,
        trace_enabled=False,
    )


# The n150 perf entry (llama_3_1_8b_instruct) is excluded from the onPR perf filter
# (still runs in nightly): device hang during uplift
# (https://github.com/tenstorrent/tt-xla/issues/5282, fix in
# https://github.com/tenstorrent/tt-metal/pull/47221). The accuracy entry still runs.
def test_llama_3_1_8b(cli, request):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm(
        ModelLoader,
        ModelVariant.LLAMA_3_1_8B_INSTRUCT,
        cli,
        request,
        fp32_dest_acc_en=False,
        required_pcc=0.90,
    )


def test_falcon3_7b_tp(cli, request):
    from third_party.tt_forge_models.falcon.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.FALCON_7B,
        cli,
        request,
        experimental_kv_cache_dtype=None,
    )


def test_falcon3_10b_tp(cli, request):
    from third_party.tt_forge_models.falcon.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.FALCON_10B,
        cli,
        request,
        experimental_kv_cache_dtype=None,
    )


def test_llama_3_1_8b_instruct_tp(cli, request):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(ModelLoader, ModelVariant.LLAMA_3_1_8B_INSTRUCT, cli, request)


def test_mistral_7b_tp(cli, request):
    from third_party.tt_forge_models.mistral.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(ModelLoader, ModelVariant.MISTRAL_7B_INSTRUCT_V03, cli, request)


# Trace disabled: host/device tensor shape mismatch (https://github.com/tenstorrent/tt-xla/issues/3935)
def test_ministral_8b_tp(cli, request):
    from third_party.tt_forge_models.mistral.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.MINISTRAL_8B,
        cli,
        request,
        trace_enabled=False,
        optimization_level=1,
    )


def test_mistral_nemo_instruct_2407_tp(cli, request):
    from third_party.tt_forge_models.mistral.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.MISTRAL_NEMO_INSTRUCT_2407,
        cli,
        request,
        optimization_level=1,
    )


def test_mistral_small_24b_instruct_2501_tp(cli, request):
    from third_party.tt_forge_models.mistral.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.MISTRAL_SMALL_24B_INSTRUCT_2501,
        cli,
        request,
        optimization_level=1,  # flaky: occasionally hangs in CI with optimization_level=2
    )


def test_qwen_2_5_14b_instruct_tp(cli, request):
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.QWEN_2_5_14B_INSTRUCT,
        cli,
        request,
        optimization_level=1,
    )


def test_qwen_2_5_32b_instruct_tp(cli, request):
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(ModelLoader, ModelVariant.QWEN_2_5_32B_INSTRUCT, cli, request)


def test_qwen_2_5_coder_32b_instruct_tp(cli, request):
    from third_party.tt_forge_models.qwen_2_5_coder.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.QWEN_2_5_CODER_32B_INSTRUCT,
        cli,
        request,
        optimization_level=1,
    )


def test_qwen_3_0_6b_tp(cli, request):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(ModelLoader, ModelVariant.QWEN_3_0_6B, cli, request)


def test_qwen_3_1_7b_tp(cli, request):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(ModelLoader, ModelVariant.QWEN_3_1_7B, cli, request)


def test_qwen_3_8b_tp(cli, request):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.QWEN_3_8B,
        cli,
        request,
        optimization_level=1,  # flaky: occasionally hangs in CI with optimization_level=2
    )


def test_qwen_3_14b_tp(cli, request):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader, ModelVariant.QWEN_3_14B, cli, request, optimization_level=1
    )


def test_qwen_3_32b_tp(cli, request):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(ModelLoader, ModelVariant.QWEN_3_32B, cli, request)


def test_llama_3_8b_instruct_tp(cli, request):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(ModelLoader, ModelVariant.LLAMA_3_8B_INSTRUCT, cli, request)


def test_llama_3_1_8b_tp(cli, request):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(ModelLoader, ModelVariant.LLAMA_3_1_8B, cli, request)


def test_llama_3_8b_tp(cli, request):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(ModelLoader, ModelVariant.LLAMA_3_8B, cli, request)


def test_llama_3_1_70b_tp(cli, request):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.LLAMA_3_1_70B_INSTRUCT,
        cli,
        request,
        weight_dtype_overrides={
            "model.layers.*.mlp.gate_proj.weight": "bfp_bf4",
            "model.layers.*.mlp.up_proj.weight": "bfp_bf4",
        },
        optimization_level=1,  # flaky: occasionally hangs in CI with optimization_level=2
    )


# Trace disabled: ~23% slower with trace on bs=32 (https://github.com/tenstorrent/tt-xla/issues/4192)
# The n300-llmbox perf entry (gpt_oss_20b_tp) is excluded from the onPR perf filter
# (still runs in nightly): hangs on n300-llmbox (https://github.com/tenstorrent/tt-xla/issues/5151).
def test_gpt_oss_20b_tp(cli, request):
    from tt_torch import TT_DENSE_EXPERTS_BACKEND_NAME

    from third_party.tt_forge_models.gpt_oss.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.GPT_OSS_20B,
        cli,
        request,
        mesh_config_fn=single_row_mesh,
        shard_spec_fn=gpt_oss_20b_shard_spec,
        trace_enabled=False,
        optimization_level=1,
        experts_implementation=TT_DENSE_EXPERTS_BACKEND_NAME,
        required_pcc=0.94,
    )


# Test with D2M fusion enabled (enable-create-d2m-subgraphs=true).
# FAILED: SIGSEGV in TTNNRowMajorLayoutPropagation (https://github.com/tenstorrent/tt-xla/issues/5121)
def test_gpt_oss_20b_tp_d2m(cli, request):
    from tt_torch import TT_DENSE_EXPERTS_BACKEND_NAME

    from third_party.tt_forge_models.gpt_oss.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.GPT_OSS_20B,
        cli,
        request,
        mesh_config_fn=single_row_mesh,
        shard_spec_fn=gpt_oss_20b_shard_spec,
        trace_enabled=False,
        optimization_level=1,
        enable_create_d2m_subgraphs=True,
        experts_implementation=TT_DENSE_EXPERTS_BACKEND_NAME,
    )


# Excluded from the onPR perf filter (still runs in nightly): slice op requires
# tile-aligned height (https://github.com/tenstorrent/tt-xla/issues/5207).
def test_gpt_oss_20b_tp_batch_size_1(cli, request):
    from tt_torch import TT_DENSE_EXPERTS_BACKEND_NAME

    from third_party.tt_forge_models.gpt_oss.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.GPT_OSS_20B,
        cli,
        request,
        mesh_config_fn=single_row_mesh,
        shard_spec_fn=gpt_oss_20b_shard_spec,
        default_batch_size=1,
        optimization_level=1,
        experts_implementation=TT_DENSE_EXPERTS_BACKEND_NAME,
    )


# Excluded from the onPR perf filter (still runs in nightly): galaxy fabric "Failed
# to add pinning constraints" (https://github.com/tenstorrent/tt-xla/issues/5210).
def test_llama_3_1_70b_tp_galaxy(cli, request):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.LLAMA_3_1_70B_INSTRUCT,
        cli,
        request,
        optimization_level=1,
    )


def test_gpt_oss_20b_tp_galaxy_batch_size_64(cli, request):
    from tt_torch import TT_DENSE_EXPERTS_BACKEND_NAME

    from third_party.tt_forge_models.gpt_oss.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.GPT_OSS_20B,
        cli,
        request,
        # 128 fails to compile - https://github.com/tenstorrent/tt-xla/issues/3907
        default_batch_size=64,
        optimization_level=1,
        experimental_kv_cache_dtype=None,
        experts_implementation=TT_DENSE_EXPERTS_BACKEND_NAME,
    )


def test_gpt_oss_120b_tp_dp_galaxy_batch_size_128(cli, request):
    from tt_torch import TT_DENSE_EXPERTS_BACKEND_NAME

    from third_party.tt_forge_models.gpt_oss.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.GPT_OSS_120B,
        cli,
        request,
        batch_size=128,
        optimization_level=1,
        mesh_config_fn=galaxy_4x8_mesh,
        shard_spec_fn=gpt_oss_120b_galaxy_shard_spec,
        input_output_sharding_spec=("batch", None),
        kv_cache_sharding_spec=("batch", "model", None, None),
        trace_enabled=True,
        experimental_kv_cache_dtype=None,
        experts_implementation=TT_DENSE_EXPERTS_BACKEND_NAME,
    )


def test_gpt_oss_120b_tp_galaxy_batch_size_64(cli, request):
    from tt_torch import TT_DENSE_EXPERTS_BACKEND_NAME

    from third_party.tt_forge_models.gpt_oss.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.GPT_OSS_120B,
        cli,
        request,
        default_batch_size=64,
        optimization_level=1,
        mesh_config_fn=galaxy_4x8_mesh,
        shard_spec_fn=gpt_oss_120b_galaxy_shard_spec,
        input_output_sharding_spec=("batch", None),
        kv_cache_sharding_spec=("batch", "model", None, None),
        trace_enabled=True,
        experimental_kv_cache_dtype=None,
        experts_implementation=TT_DENSE_EXPERTS_BACKEND_NAME,
    )


def test_gpt_oss_120b_tp_qb2(cli, request):
    from tt_torch import TT_DENSE_EXPERTS_BACKEND_NAME

    from third_party.tt_forge_models.gpt_oss.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.GPT_OSS_120B,
        cli,
        request,
        default_batch_size=8,
        optimization_level=1,
        trace_enabled=True,
        experimental_weight_dtype="bfp_bf8",
        experimental_kv_cache_dtype=None,
        weight_dtype_overrides={
            "default": "bfp_bf8",
            "model.layers.*.mlp.experts.gate_up_proj": "bfp_bf4",
            "model.layers.*.mlp.experts.down_proj": "bfp_bf4",
        },
        required_pcc=0.93,  # set for now as it's ~0.93 on test runs locally
        mesh_config_fn=qb2_1x4_mesh,
        # shard_spec_fn=gpt_oss_120b_qb2_shard_spec,
        experts_implementation=TT_DENSE_EXPERTS_BACKEND_NAME,
    )


# Trace disabled: topk i64 indices can't reside in device DRAM inside capture_or_execute_trace
# This test only runs 2 layers so we expect to see incoherent output
def test_kimi_k2_tp_galaxy_2_layers(cli, request):
    from third_party.tt_forge_models.kimi_k2.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.KIMI_K2_INSTRUCT_MODIFIED,
        cli,
        request,
        num_layers=2,
        batch_size=64,  # Test hangs for a batch size of 128 - Issue: https://github.com/tenstorrent/tt-xla/issues/4565
        input_output_sharding_spec=("batch", None),
        use_mla_cache=True,
        experimental_kv_cache_dtype=None,
        optimization_level=0,
        trace_enabled=False,
    )


# Trace disabled: topk i64 indices can't reside in device DRAM inside capture_or_execute_trace
# This test only runs 2 layers so we expect to see incoherent output
def test_kimi_k2_5_tp_galaxy_2_layers(cli, request):
    from third_party.tt_forge_models.kimi_k2.k2_5.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.KIMI_K2_5_MODIFIED,
        cli,
        request,
        num_layers=2,
        batch_size=64,  # Test hangs for a batch size of 128 - Issue: https://github.com/tenstorrent/tt-xla/issues/4565
        input_output_sharding_spec=("batch", None),
        use_mla_cache=True,
        experimental_kv_cache_dtype=None,
        optimization_level=0,
        trace_enabled=False,
    )


# This test only runs 2 layers so we expect to see incoherent output
def test_deepseek_v3_2_exp_tp_galaxy_2_layers(cli, request):
    from third_party.tt_forge_models.deepseek.deepseek_v3_2_exp.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.DEEPSEEK_V3_2_EXP_MODIFIED,
        cli,
        request,
        num_layers=2,
        batch_size=128,
        input_output_sharding_spec=("batch", None),
        use_mla_cache=True,
        use_indexer_cache=True,
        experimental_kv_cache_dtype=None,
        optimization_level=0,
        trace_enabled=False,
        required_pcc=0.92,
    )


def test_falcon3_7b_tp_qb2(cli, request):
    from third_party.tt_forge_models.falcon.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(ModelLoader, ModelVariant.FALCON_7B, cli, request, optimization_level=2)


def test_falcon3_10b_tp_qb2(cli, request):
    from third_party.tt_forge_models.falcon.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader, ModelVariant.FALCON_10B, cli, request, optimization_level=2
    )


def test_llama_3_1_8b_instruct_tp_qb2(cli, request):
    from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.LLAMA_3_1_8B_INSTRUCT,
        cli,
        request,
        optimization_level=2,
    )


def test_ministral_8b_tp_qb2(cli, request):
    from third_party.tt_forge_models.mistral.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader, ModelVariant.MINISTRAL_8B, cli, request, optimization_level=2
    )


def test_mistral_nemo_instruct_2407_tp_qb2(cli, request):
    from third_party.tt_forge_models.mistral.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.MISTRAL_NEMO_INSTRUCT_2407,
        cli,
        request,
        optimization_level=2,
    )


def test_mistral_small_24b_instruct_2501_tp_qb2(cli, request):
    from third_party.tt_forge_models.mistral.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.MISTRAL_SMALL_24B_INSTRUCT_2501,
        cli,
        request,
        optimization_level=2,
    )


def test_qwen_2_5_14b_instruct_tp_qb2(cli, request):
    from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.QWEN_2_5_14B_INSTRUCT,
        cli,
        request,
        optimization_level=2,
    )


def test_qwen_2_5_coder_32b_instruct_tp_qb2(cli, request):
    from third_party.tt_forge_models.qwen_2_5_coder.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.QWEN_2_5_CODER_32B_INSTRUCT,
        cli,
        request,
        optimization_level=2,
    )


def test_qwen_3_8b_tp_qb2(cli, request):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(ModelLoader, ModelVariant.QWEN_3_8B, cli, request, optimization_level=2)


def test_qwen_3_14b_tp_qb2(cli, request):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader, ModelVariant.QWEN_3_14B, cli, request, optimization_level=2
    )


def test_qwen_3_32b_tp_qb2(cli, request):
    from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader, ModelVariant.QWEN_3_32B, cli, request, optimization_level=2
    )


def test_gpt_oss_20b_tp_qb2(cli, request):
    from tt_torch import TT_DENSE_EXPERTS_BACKEND_NAME

    from third_party.tt_forge_models.gpt_oss.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.GPT_OSS_20B,
        cli,
        request,
        mesh_config_fn=single_row_mesh,
        shard_spec_fn=gpt_oss_20b_shard_spec,
        optimization_level=2,
        experts_implementation=TT_DENSE_EXPERTS_BACKEND_NAME,
    )


# This test only runs 4 layers so we expect to see incoherent output
def test_deepseek_v3_1_tp_galaxy_4_layers(cli, request):
    from third_party.tt_forge_models.deepseek.deepseek_v3_1.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.DEEPSEEK_V3_1_MODIFIED,
        cli,
        request,
        default_num_layers=4,
        batch_size=64,  # Test hangs for a batch size of 128 - Issue: https://github.com/tenstorrent/tt-xla/issues/4565
        input_output_sharding_spec=("batch", None),
        use_mla_cache=True,
        optimization_level=0,
        trace_enabled=False,
        shard_spec_fn=deepseek_v3_1_shard_spec,
        required_pcc=0.96,
        experimental_kv_cache_dtype=None,
    )


# This test only runs 4 layers so we expect to see incoherent output
def test_glm_4_7_tp_galaxy_4_layers(cli, request):
    from third_party.tt_forge_models.glm.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _run_llm_tp(
        ModelLoader,
        ModelVariant.GLM_4_7,
        cli,
        request,
        default_num_layers=4,
        batch_size=64,  # Test hangs for a batch size of 128 - Issue: https://github.com/tenstorrent/tt-xla/issues/4565
        optimization_level=0,
        trace_enabled=False,
        shard_spec_fn=glm_4_7_shard_spec,
        input_output_sharding_spec=("batch", None),
        kv_cache_sharding_spec=("batch", "model", None, None),
        required_pcc=0.99,
    )
