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
import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra.evaluators import TorchComparisonEvaluator

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
_SEQ_LENS = [8, 32, 64, 128, 512]

# Smaller matrix for the MoE-on-the-path test. Each expert subgraph is unrolled
# at trace time (256 experts), and FLOPs scale with seq_len, so we keep this
# focused on two tile-aligned lengths: one block-sized, one larger.
_SEQ_LENS_MOE = [32, 64]


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

    loader = ModelLoader(num_layers=1)

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

    With ``num_layers=2``:
      * layer 0 is the dense ``LLaDA2MoeMLP`` (``layer_idx < first_k_dense_replace``).
      * layer 1 is ``LLaDA2MoeSparseMoeBlock``: the gate, group-limited top-k
        routing, the 256 routed experts, and the shared expert.

    The loader installs ``_compile_friendly_moe_forward`` on the MoE block,
    replacing the upstream ``moe_infer`` (which uses ``.cpu().numpy()``,
    ``.item()`` and a Python loop with ``if num_tokens == 0: continue`` —
    all dynamo-hostile) with a mathematically-equivalent vectorized form.
    The CPU/TT comparison below is therefore parity of the *patched* graph
    on both sides; equivalence with the upstream eager path is verified
    separately in :func:`test_llada2_mini_compile_friendly_moe_cpu_equivalence`.
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
