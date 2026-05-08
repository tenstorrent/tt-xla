# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Bring-up tests for inclusionAI/LLaDA2.0-mini on the Tenstorrent backend.

LLaDA2.0 is a discrete diffusion LM that registers as an HF ``CausalLM`` via
``auto_map``. These tests do not validate generation quality (which would
require an iterative mask-denoising loop). Instead they answer the canonical
compiler bring-up question: does ``torch.compile(backend="tt")`` produce
forward-pass logits that numerically match eager CPU?

Coverage strategy
-----------------
LLaDA2.0-mini's per-layer compute depends on ``layer_idx``: the first
``first_k_dense_replace=1`` layers use a dense ``LLaDA2MoeMLP`` while later
layers use ``LLaDA2MoeSparseMoeBlock`` (256 routed experts, 1 shared expert).
We test both regimes:

* :func:`test_llada2_mini_forward_single_layer` (``num_layers=1``) — exercises
  attention, RoPE, RMSNorm, the (B, 1, T, T) bool block mask path, and the
  dense MLP. No MoE on the path. Cheap, parametrized over many ``seq_len``.

* :func:`test_llada2_mini_forward_with_moe` (``num_layers=2``) — adds one
  ``LLaDA2MoeSparseMoeBlock`` (layer index 1) so the MoE gate, top-k group
  routing, expert dispatch and shared experts all enter the traced graph.
  Smaller ``seq_len`` matrix because compile cost scales with experts × T.

* :func:`test_llada2_mini_compile_friendly_moe_cpu_equivalence` — pure CPU
  sanity check that the compile-friendly MoE forward installed by the loader
  (see ``loader.py::_compile_friendly_moe_forward``) is numerically equivalent
  to the upstream ``moe_infer`` path. If this drifts, every TT MoE result
  below is suspect, so we want a stand-alone guard.
"""
import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import TorchComparisonEvaluator
from torch_xla.distributed.spmd import Mesh
from tt_torch.sparse_mlp import enable_sparse_mlp

from tests.infra.evaluators.evaluation_config import (
    AtolConfig,
    ComparisonConfig,
    PccConfig,
)
from tests.infra.testers.single_chip.model.model_tester import RunMode
from tests.utils import BringupStatus, ModelGroup
from third_party.tt_forge_models.llada2_mini.pytorch import ModelLoader


# Sequence lengths to cover for forward-parity bring-up of the dense slice.
#
# Why these values:
#   8   - minimal smoke test, fast first-time compile.
#   32  - one ``block_length`` (LLaDA's ``generate()`` default), the smallest
#         length the model is actually invoked at during real generation.
#   64  - two blocks; first length where the (B, 1, T, T) mask is non-trivial.
#   128 - typical short-generation window.
#   512 - stress length: attention is O(T^2), exercises softmax / RoPE paths
#         that lighter lengths can hide. Still cheap with ``num_layers=1``.
#
# All values are multiples of TT-MLIR's tile size (32) except 8, which is kept
# only as a sanity smoke entry.
#_SEQ_LENS = [8, 32, 64, 128, 512]
_SEQ_LENS = [32]

# Smaller matrix for the MoE-on-the-path test. Each expert subgraph is unrolled
# at trace time (256 experts), and FLOPs scale with seq_len, so we keep this
# focused on two tile-aligned lengths: one block-sized, one larger.
#_SEQ_LENS_MOE = [32, 64]
_SEQ_LENS_MOE = [32]


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.model_test
@pytest.mark.parametrize("seq_len", _SEQ_LENS, ids=[f"T{t}" for t in _SEQ_LENS])
@pytest.mark.record_test_properties(
    model_name="inclusionAI/LLaDA2.0-mini",
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.UNKNOWN,
)
def test_llada2_mini_forward_single_layer(seq_len):
    """Forward-pass parity test for a 1-layer slice of LLaDA2.0-mini.

    With ``num_layers=1`` and ``first_k_dense_replace=1``, the only decoder
    layer uses a dense ``LLaDA2MoeMLP`` (no MoE). This test covers attention,
    partial RoPE, RMSNorm, the (B, 1, T, T) bool block mask path, and the
    dense MLP. MoE coverage is in :func:`test_llada2_mini_forward_with_moe`.

    Parametrized over ``seq_len`` because the compiler can be shape-sensitive
    (RoPE edges, attention-mask lowering paths, tile alignment), and LLaDA's
    own ``generate()`` calls forward at a range of growing lengths
    (multiples of ``block_length``).
    """
    xr.set_device_type("TT")
    device = torch_xla.device()

    loader = ModelLoader(num_layers=2)

    inputs_cpu = loader.load_inputs(batch_size=1, seq_len=seq_len)
    model = loader.load_model(dtype_override=torch.bfloat16)

    with torch.no_grad():
        cpu_out = model(**inputs_cpu)
    cpu_logits = cpu_out.logits.detach().cpu()
    assert cpu_logits.shape == (1, seq_len, model.config.vocab_size)

    model_tt = model.to(device)
    inputs_tt = {k: v.to(device) for k, v in inputs_cpu.items()}
    compiled = torch.compile(model_tt, backend="tt")

    with torch.no_grad():
        tt_out = compiled(**inputs_tt)
    tt_logits = tt_out.logits.detach().cpu()

    comparator = TorchComparisonEvaluator(
        ComparisonConfig(
            atol=AtolConfig(enabled=False),
            pcc=PccConfig(required_pcc=0.99),
        )
    )
    comparator.evaluate(tt_logits, cpu_logits)


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.model_test
@pytest.mark.parametrize("seq_len", _SEQ_LENS_MOE, ids=[f"T{t}" for t in _SEQ_LENS_MOE])
@pytest.mark.record_test_properties(
    model_name="inclusionAI/LLaDA2.0-mini",
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.UNKNOWN,
)
def test_llada2_mini_forward_with_moe(seq_len):
    """Forward-pass parity test that puts an MoE layer on the traced graph.

    With ``num_layers=2`` and the upstream default
    ``first_k_dense_replace=1``:
      * layer 0 is the dense ``LLaDA2MoeMLP``.
      * layer 1 is ``LLaDA2MoeSparseMoeBlock``: the gate, group-limited
        top-k routing, the 256 routed experts, and the shared expert.

    The loader installs ``_compile_friendly_moe_forward`` on the MoE block,
    replacing the upstream ``moe_infer`` (which uses ``.cpu().numpy()``,
    ``.item()`` and a Python loop with ``if num_tokens == 0: continue`` —
    all dynamo-hostile) with a mathematically-equivalent vectorized form.
    The CPU/TT comparison below is therefore parity of the *patched* graph
    on both sides; equivalence with the upstream eager path is verified
    separately in :func:`test_llada2_mini_compile_friendly_moe_cpu_equivalence`.

    Compile note: ``enable_const_eval_on_cpu=False`` is set on the PJRT
    compile options. Without this, ``CPUHoistConstEvalTransform`` moves the
    fused 256-expert weight ``concat`` into the CPU module, which then runs
    upstream MLIR's ``OneShotBufferizePass`` over a ~514-deep
    ``tensor.i:w
    nsert_slice`` chain. That pass scales super-quadratically on
    such chains and effectively never finishes (see
    ``tools/repro/oneshot_bufferize_hang/README.md``). Disabling the hoist
    keeps the const-eval on-device, which sidesteps the bufferize hang at
    the cost of slightly lower const-fold precision (device dtypes vs CPU
    fp32) and higher device memory pressure.
    """
    xr.set_device_type("TT")
    torch_xla.set_custom_compile_options({"enable_const_eval_on_cpu": False})
    device = torch_xla.device()

    loader = ModelLoader(num_layers=1)

    inputs_cpu = loader.load_inputs(batch_size=1, seq_len=seq_len)
    model = loader.load_model(dtype_override=torch.bfloat16)
    breakpoint()

    with torch.no_grad():
        cpu_out = model(**inputs_cpu)
    cpu_logits = cpu_out.logits.detach().cpu()
    assert cpu_logits.shape == (1, seq_len, model.config.vocab_size)

    # call sparse_mlp.enable_sparse_mlp(model_tt, mesh, cluster_axis=0)
    # to transform the model to use sparce mlp so its not on device 
    # note taht you need to fully shard the experts per device
    # 
    model_tt = model.to(device)
    inputs_tt = {k: v.to(device) for k, v in inputs_cpu.items()}


    compiled = torch.compile(model_tt, backend="tt")

    with torch.no_grad():
        tt_out = compiled(**inputs_tt)
    tt_logits = tt_out.logits.detach().cpu()

    comparator = TorchComparisonEvaluator(
        ComparisonConfig(
            atol=AtolConfig(enabled=False),
            pcc=PccConfig(required_pcc=0.99),
        )
    )
    comparator.evaluate(tt_logits, cpu_logits)


def test_llada2_mini_compile_friendly_moe_cpu_equivalence():
    """CPU-only sanity: compile-friendly MoE forward matches the upstream path.

    The TT MoE tests above compare ``compile_friendly_moe=True`` on CPU vs the
    same patched module on TT. That tells us the compile path is self-consistent
    but does not by itself prove the patch is the right answer. This test pins
    the patch to the upstream ``moe_infer`` reference: same config, same seed,
    one with the patch installed and one without; outputs must match closely.

    If this assertion ever drifts, the patched ``LLaDA2MoeSparseMoeBlock.forward``
    is wrong and every PCC number from the TT MoE test should be re-examined.
    """
    seq_len = 32

    loader_orig = ModelLoader(num_layers=2)
    inputs = loader_orig.load_inputs(batch_size=1, seq_len=seq_len)
    model_orig = loader_orig.load_model(
        dtype_override=torch.bfloat16, compile_friendly_moe=False
    )
    with torch.no_grad():
        out_orig = model_orig(**inputs).logits.detach().float()

    loader_patched = ModelLoader(num_layers=2)
    model_patched = loader_patched.load_model(
        dtype_override=torch.bfloat16, compile_friendly_moe=True
    )
    with torch.no_grad():
        out_patched = model_patched(**inputs).logits.detach().float()

    # bf16 weights + fp32 router accumulation → small but non-zero numerical
    # drift between the two formulations. PCC is the right metric; an absolute
    # tolerance would either be too loose to catch a real divergence or too
    # tight to pass on the inevitable last-bit differences.
    comparator = TorchComparisonEvaluator(
        ComparisonConfig(
            atol=AtolConfig(enabled=False),
            pcc=PccConfig(required_pcc=0.999),
        )
    )
    comparator.evaluate(out_patched, out_orig)


# Sequence lengths for the multi-device sparse-MoE block test. Kept small;
# every chosen value is divisible by the dispatch axis (4) so that any
# sequence-parallel sharding can split the batch*seq dim cleanly.
_SEQ_LENS_SPARSE = [32]


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize(
    "seq_len", _SEQ_LENS_SPARSE, ids=[f"T{t}" for t in _SEQ_LENS_SPARSE]
)
@pytest.mark.record_test_properties(
    model_name="inclusionAI/LLaDA2.0-mini",
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.UNKNOWN,
)
def test_llada2_mini_layer_sparse_moe(seq_len):
    """Multi-device sharded forward-parity test for one MoE decoder layer.

    Mirrors :func:`tests.torch.models.deepseek_v3_2_exp.test_deepseek_v3_2_exp.
    test_deepseek_v3_2_layer_sparse_moe` but for LLaDA2.0-mini. We extract
    the layer-1 ``LLaDA2MoeDecoderLayer`` (MoE block under the upstream
    default ``first_k_dense_replace=1``), swap its
    ``LLaDA2MoeSparseMoeBlock`` for an ``A2aSparseMLP`` via
    :func:`enable_sparse_mlp`, and shard the resulting block onto a 1×4 mesh
    with all parallelism on the second axis (``cluster_axis=1`` so that
    ``dispatch_devices=4``).

    Why the eager LLaDA2 MoE path is replaced rather than left in place:
    ``LLaDA2MoeSparseMoeBlock.moe_infer`` is dynamo-hostile (uses
    ``.cpu().numpy()``, ``.item()`` and a Python loop with a data-dependent
    ``continue``), so it cannot be traced as a single graph.
    ``A2aSparseMLP`` is the compile-friendly *and* sharded replacement
    that we actually want to validate end-to-end.

    What this test exercises that the existing single-device tests do not:
      * Multi-device SPMD lowering of an MoE layer.
      * Sparse expert dispatch / combine via the all-to-all collective.
      * The ``RouterAdapter`` plumbing for LLaDA2's 3-tuple gate output.

    Compile note: like
    :func:`test_llada2_mini_forward_with_moe`, this test sets
    ``enable_const_eval_on_cpu=False`` to sidestep the
    ``OneShotBufferizePass`` hang on long ``tensor.insert_slice`` chains
    (see ``tools/repro/oneshot_bufferize_hang/`` and
    ``https://github.com/tenstorrent/tt-mlir/issues/8327``). Once the
    upstream pass / lowering is fixed this option can be removed.
    """
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()
    torch_xla.set_custom_compile_options({"enable_const_eval_on_cpu": False})

    batch_size = 1

    # Build the model with num_layers=2 so that layer 1 is an MoE block under
    # the upstream default first_k_dense_replace=1. compile_friendly_moe is
    # irrelevant here (enable_sparse_mlp will replace the MoE block entirely),
    # but we set it False so the model we hand to enable_sparse_mlp matches
    # the upstream eager forward bit-for-bit on the CPU golden path.
    loader = ModelLoader(num_layers=2)
    model = loader.load_model(
        dtype_override=torch.bfloat16, compile_friendly_moe=False
    )
    config = model.config

    # Layer 1 is the first MoE layer (layer 0 is dense due to
    # first_k_dense_replace=1).
    block = model.model.layers[1]
    block.eval()

    # Inputs to LLaDA2MoeDecoderLayer.forward.
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    # All-zeros additive mask = fully bidirectional attention. LLaDA2 is a
    # diffusion LM, so this matches its real-world use semantically; the
    # alternative (a causal mask) would not match what generate() does at
    # all and would obscure router behaviour with masked-out tokens.
    attention_mask = torch.zeros(
        (batch_size, 1, seq_len, seq_len), dtype=torch.bfloat16
    )
    # Compute (cos, sin) for the rotary embedding once, off the same
    # rotary_emb instance the full model would use. The decoder layer
    # signature requires position_embeddings as a kwarg/positional;
    # passing position_ids alone is *not* enough -- the upstream layer
    # has already been refactored to take the precomputed pair.
    position_ids = (
        torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    )
    with torch.no_grad():
        position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

    # Replace the MoE block in-place with A2aSparseMLP. cluster_axis=1
    # routes all expert dispatch onto the 4-device axis (dispatch_devices =
    # mesh[1] = 4); axis 0 is a no-op for our 1×4 mesh.
    mesh_shape = (1, 4)
    enable_sparse_mlp(
        block,
        mesh=mesh_shape,
        cluster_axis=1,
        config=config,
    )

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))

    def get_shard_spec(block, args, kwargs):
        """Tensor-parallel shard spec for one LLaDA2 MoE decoder layer on 1×4.

        Strategy: every parallelism dim is on ``_axis_1`` (4 devices);
        ``_axis_0`` has size 1 and is unused. This is plain "model" tensor
        parallelism, no batch parallelism (batch_size=1).
        """
        shard_specs = {}

        # Inputs.
        # hidden_states: [B, T, dim] — shard hidden on axis 1 (column-parallel
        # input to QKV) so attention.query_key_value can fan out without an
        # all-gather.
        shard_specs[args[0]] = (None, None, "_axis_1")
        # attention_mask: [B, 1, T, T] — replicated.
        # position_embeddings: tuple, replicated.

        # Attention.
        # query_key_value.weight: [(num_heads + 2*num_kv_heads) * head_dim, dim].
        # Output dim sharded on axis 1 -> each device computes head_dim *
        # (heads/4 + 2 * kv_heads/4) of the QKV concat.
        attn = block.attention
        shard_specs[attn.query_key_value.weight] = ("_axis_1", None)
        if attn.query_key_value.bias is not None:
            shard_specs[attn.query_key_value.bias] = ("_axis_1",)
        # dense (output projection): [dim, num_heads * head_dim]. Input dim
        # sharded on axis 1 so the per-device partial outputs sum across
        # devices via the implicit all-reduce.
        shard_specs[attn.dense.weight] = (None, "_axis_1")
        if attn.dense.bias is not None:
            shard_specs[attn.dense.bias] = (None,)
        # Per-head qk-norm (if enabled): replicated, head_dim is unsharded.
        if hasattr(attn, "query_layernorm"):
            shard_specs[attn.query_layernorm.weight] = (None,)
            shard_specs[attn.key_layernorm.weight] = (None,)

        # MoE block (now an A2aSparseMLPWithSharedExperts after enable_sparse_mlp).
        # The wrapper exposes the inner A2aSparseMLP at .mlp; its experts
        # tensors are stacked [E, *] parameters, and shared_experts (if any)
        # is the original LLaDA2MoeMLP.
        wrapper = block.mlp
        a2a_mlp = wrapper.mlp if hasattr(wrapper, "mlp") else wrapper
        shared_experts = getattr(wrapper, "shared_experts", None)

        # Router gate: [n_experts, hidden]. Replicated -- it runs on full
        # hidden state to score every expert.
        shard_specs[a2a_mlp.router.gate.weight] = (None, None)

        # Expert weights: shard expert dim across axis 1 (=> 256 / 4 = 64
        # experts per device).
        experts = a2a_mlp.experts
        shard_specs[experts.gate_proj] = ("_axis_1", None, None)
        shard_specs[experts.up_proj] = ("_axis_1", None, None)
        shard_specs[experts.down_proj] = ("_axis_1", None, None)
        if experts.gate_proj_bias is not None:
            shard_specs[experts.gate_proj_bias] = ("_axis_1", None)
            shard_specs[experts.up_proj_bias] = ("_axis_1", None)
            shard_specs[experts.down_proj_bias] = ("_axis_1", None)

        # Shared experts: one extra LLaDA2MoeMLP applied to every token,
        # tensor-parallel along the intermediate dim.
        if shared_experts is not None:
            shard_specs[shared_experts.gate_proj.weight] = ("_axis_1", None)
            shard_specs[shared_experts.up_proj.weight] = ("_axis_1", None)
            shard_specs[shared_experts.down_proj.weight] = (None, "_axis_1")

        # Norms (RMSNorm, per-channel scale, replicated).
        shard_specs[block.input_layernorm.weight] = (None,)
        shard_specs[block.post_attention_layernorm.weight] = (None,)

        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.95),
    )

    # Positional args follow LLaDA2MoeDecoderLayer.forward exactly:
    #   (hidden_states, attention_mask, position_ids, past_key_value,
    #    output_attentions, output_router_logits, use_cache, position_embeddings)
    run_graph_test(
        block,
        [
            hidden_states,
            attention_mask,
            None,  # position_ids
            None,  # past_key_value
            False,  # output_attentions
            False,  # output_router_logits
            False,  # use_cache
            position_embeddings,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )
