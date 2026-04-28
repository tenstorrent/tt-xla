# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Bring-up test for inclusionAI/LLaDA2.0-mini on the Tenstorrent backend.

LLaDA2.0 is a discrete diffusion LM that registers as an HF ``CausalLM`` via
``auto_map``. This test does not validate generation quality (which would
require an iterative mask-denoising loop). Instead it answers the canonical
compiler bring-up question: does ``torch.compile(backend="tt")`` produce
forward-pass logits that numerically match eager CPU?

Strategy mirrors ``tests/torch/models/kimi_k2/test_kimi_k2.py`` and
``tests/torch/models/llama3/test_llama_step_n300.py``:

  1. Use ``ModelLoader`` to instantiate a 1-layer LLaDA2.0-mini from config
     (random weights, no 32 GB checkpoint download).
  2. Build the (B, 1, T, T) bool block attention mask the model requires.
  3. Forward on CPU -> golden logits.
  4. Move module + inputs to the TT device, ``torch.compile(backend="tt")``,
     forward.
  5. ``TorchComparisonEvaluator`` with ``PccConfig(required_pcc=0.99)``.

The test is parametrized over ``seq_len`` because the compiler can be
shape-sensitive (RoPE edges, mask-lowering paths, tile alignment), and
LLaDA's own ``generate()`` invokes forward at a range of growing lengths
(multiples of ``block_length``).
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


# Sequence lengths to cover for forward-parity bring-up.
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

    A single layer keeps first-time TT compile tractable while still
    exercising the unusual bits of LLaDA's graph: the (B, 1, T, T) bool
    block mask, ``LLaDA2MoeRotaryEmbedding``, the MoE router + experts,
    and the bidirectional attention pattern.

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
