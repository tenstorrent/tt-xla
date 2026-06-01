# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Janus-Pro — ImageTokenStep decode decoder-layer ``run_op_test`` sanities.

Isolates ``language_model.model`` decoder execution (``LlamaDecoderLayer`` loop in
``janus_decoder_arc.txt``) using the same forge inputs as ``test_image_token_decode_*``
(``make_image_token_decode_inputs``: CPU prefill KV + single-step CFG embeds).

Prefill component tests pass; decode PCC drops (issue #4968). These op tests narrow
whether error accumulates across decoder layers vs ``gen_head``.

Pro-7B: ``@pytest.mark.p150`` only (DRAM OOM on n150).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import pytest
import torch
import torch_xla.runtime as xr
from infra.evaluators import ComparisonConfig, TorchComparisonEvaluator
from tests.runner.requirements import RequirementsManager
from tests.torch.models.janus_pro_pcc_drop.decoder_sanity import (
    JanusGenHeadDecode,
    llama_decode_hidden_states,
    load_image_token_decode_bundle,
)

import third_party.tt_forge_models.janus_pro.text_to_image.pytorch.loader as janus_loader
from third_party.tt_forge_models.janus_pro.text_to_image.pytorch import (
    ModelLoader,
    ModelVariant,
)

# Janus-Pro-1B / Pro-7B ``language_model.model`` depth (LlamaDecoderLayer count).
PRO_1B_NUM_HIDDEN_LAYERS = 24
PRO_7B_NUM_HIDDEN_LAYERS = 30


def _layer_checkpoints(num_hidden_layers: int) -> tuple[int, ...]:
    candidates = [1, 5, 10, 15, 20, 25, num_hidden_layers]
    seen: set[int] = set()
    out: list[int] = []
    for n in candidates:
        if 0 < n <= num_hidden_layers and n not in seen:
            seen.add(n)
            out.append(n)
    return tuple(out)


PARTIAL_LAYER_CHECKPOINTS_PRO_1B = _layer_checkpoints(PRO_1B_NUM_HIDDEN_LAYERS)
PARTIAL_LAYER_CHECKPOINTS_PRO_7B = _layer_checkpoints(PRO_7B_NUM_HIDDEN_LAYERS)


def _execute_decoder_op_test(
    decode_bundle: dict,
    *,
    use_explicit_layers_loop: bool,
    num_layers: int | None = None,
) -> None:
    from tests.torch.models.janus_pro_pcc_drop.decoder_layers_op_test import (
        run_full_decoder_forge_vs_cpu_isolated,
    )

    label = "decoder_layers_loop" if use_explicit_layers_loop else "decoder_model_forward"
    if num_layers is not None:
        label = f"{label}_n{num_layers}"
    run_full_decoder_forge_vs_cpu_isolated(
        decode_bundle,
        use_explicit_layers_loop=use_explicit_layers_loop,
        num_layers=num_layers,
        label=label,
        assert_on_failure=True,
    )


def _run_decoder_op_test(
    variant_name: str,
    *,
    use_explicit_layers_loop: bool,
    num_layers: int | None = None,
) -> None:
    loader_path = inspect.getsourcefile(janus_loader)
    with RequirementsManager.for_loader(loader_path, framework="torch"):
        from third_party.tt_forge_models.janus_pro.text_to_image.pytorch.src import (
            model_utils,
        )

        torch.manual_seed(42)
        model_utils._mmgpt_cache.clear()
        repo_id = ModelLoader(ModelVariant(variant_name))._repo_id()
        decode_bundle = load_image_token_decode_bundle(repo_id, dtype=torch.bfloat16)
        _execute_decoder_op_test(
            decode_bundle,
            use_explicit_layers_loop=use_explicit_layers_loop,
            num_layers=num_layers,
        )


def _run_decoder_partial_op_test(variant_name: str, num_layers: int) -> None:
    """One ``num_layers`` depth per test (fresh decode inputs, single compile)."""
    loader_path = inspect.getsourcefile(janus_loader)
    with RequirementsManager.for_loader(loader_path, framework="torch"):
        from third_party.tt_forge_models.janus_pro.text_to_image.pytorch.src import (
            model_utils,
        )
        from third_party.tt_forge_models.janus_pro.text_to_image.pytorch.src.model_utils import (
            load_mmgpt,
            make_image_token_decode_inputs,
        )

        torch.manual_seed(42)
        model_utils._mmgpt_cache.clear()

        repo_id = ModelLoader(ModelVariant(variant_name))._repo_id()
        dtype = torch.bfloat16
        decode = make_image_token_decode_inputs(repo_id, dtype)
        decode_bundle = {
            "llama_model": load_mmgpt(repo_id, dtype).language_model.model,
            "inputs_embeds": decode["inputs_embeds"],
            "past_key_values": decode["past_key_values"],
        }
        _execute_decoder_op_test(
            decode_bundle,
            use_explicit_layers_loop=True,
            num_layers=num_layers,
        )


@dataclass(frozen=True)
class _DecoderLayerProfileRow:
    pcc: float
    max_abs_diff: float
    mean_abs_diff: float
    rel_l2_diff: float


def _profile_row_for_hidden_pair(
    evaluator: TorchComparisonEvaluator,
    device_hidden: torch.Tensor,
    golden_hidden: torch.Tensor,
) -> _DecoderLayerProfileRow:
    tt = device_hidden.detach().cpu().to(torch.float64)
    ref = golden_hidden.detach().cpu().to(torch.float64)
    diff = tt - ref
    abs_diff = diff.abs()
    ref_norm = ref.norm().item()
    rel_l2 = float(diff.norm().item() / ref_norm) if ref_norm > 0 else float("inf")
    return _DecoderLayerProfileRow(
        pcc=_compare_pcc_for_hidden_pair(evaluator, tt, ref),
        max_abs_diff=float(abs_diff.max().item()),
        mean_abs_diff=float(abs_diff.mean().item()),
        rel_l2_diff=rel_l2,
    )


def _compare_pcc_for_hidden_pair(
    evaluator: TorchComparisonEvaluator,
    device_hidden: torch.Tensor,
    golden_hidden: torch.Tensor,
) -> float:
    return evaluator._compare_pcc(
        device_hidden,
        golden_hidden,
        evaluator._comparison_config.pcc,
        None,
    )


def _print_decoder_layer_profile_table(
    rows_after_layer: dict[int, _DecoderLayerProfileRow],
    *,
    row_after_norm: _DecoderLayerProfileRow | None,
    num_hidden_layers: int,
    variant_name: str,
) -> None:
    print(f"\n=== ImageTokenStep decode decoder profile ({variant_name}) ===")
    print(
        f"{'depth':>6}  {'after':<8}  {'pcc':>10}  {'max_abs':>12}  "
        f"{'mean_abs':>12}  {'rel_l2':>10}"
    )
    print("-" * 68)
    for depth in range(1, num_hidden_layers + 1):
        row = rows_after_layer[depth]
        print(
            f"{depth:>6}  {'layer':<8}  {row.pcc:>10.6f}  {row.max_abs_diff:>12.6e}  "
            f"{row.mean_abs_diff:>12.6e}  {row.rel_l2_diff:>10.6e}"
        )
    if row_after_norm is not None:
        row = row_after_norm
        print(
            f"{num_hidden_layers:>6}  {'norm':<8}  {row.pcc:>10.6f}  {row.max_abs_diff:>12.6e}  "
            f"{row.mean_abs_diff:>12.6e}  {row.rel_l2_diff:>10.6e}"
        )
    print("=" * 68 + "\n")


def _run_decoder_layer_pcc_profile(variant_name: str) -> dict[int, _DecoderLayerProfileRow]:
    """Isolated CPU vs Forge; cumulative PCC after each decoder layer depth."""
    from tests.torch.models.janus_pro_pcc_drop.decoder_layers_op_test import (
        load_decode_bundle_for_variant,
        run_decoder_cumulative_layer_profile_isolated,
    )

    decode_bundle = load_decode_bundle_for_variant(variant_name)
    isolated_rows = run_decoder_cumulative_layer_profile_isolated(
        decode_bundle,
        variant_name=variant_name,
    )
    return {
        depth: _DecoderLayerProfileRow(
            pcc=row.pcc,
            max_abs_diff=row.max_abs_diff,
            mean_abs_diff=row.mean_abs_diff,
            rel_l2_diff=row.rel_l2_diff,
        )
        for depth, row in isolated_rows.items()
    }


def _run_gen_head_op_test(variant_name: str) -> None:
    loader_path = inspect.getsourcefile(janus_loader)
    with RequirementsManager.for_loader(loader_path, framework="torch"):
        from third_party.tt_forge_models.janus_pro.text_to_image.pytorch.src import (
            model_utils,
        )

        torch.manual_seed(42)
        model_utils._mmgpt_cache.clear()

        repo_id = ModelLoader(ModelVariant(variant_name))._repo_id()
        bundle = load_image_token_decode_bundle(repo_id, dtype=torch.bfloat16)

        with torch.inference_mode():
            hidden_states = llama_decode_hidden_states(
                bundle["llama_model"],
                bundle["inputs_embeds"],
                bundle["past_key_values"],
            )

        xr.set_device_type("TT")

        from tests.torch.models.janus_pro_pcc_drop.decoder_op_test_utils import (
            run_forge_vs_cpu_op_test_isolated,
        )

        hidden_cpu = hidden_states.detach().cpu()

        def build() -> tuple:
            return JanusGenHeadDecode(bundle["gen_head"]), [hidden_cpu]

        run_forge_vs_cpu_op_test_isolated(build, label="gen_head_decode")


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_image_token_decoder_model_forward_decode_pro_1b():
    _run_decoder_op_test("Pro_1B", use_explicit_layers_loop=False)


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_image_token_decoder_layers_loop_full_decode_pro_1b():
    _run_decoder_op_test("Pro_1B", use_explicit_layers_loop=True, num_layers=None)


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
@pytest.mark.parametrize(
    "num_layers",
    PARTIAL_LAYER_CHECKPOINTS_PRO_1B,
    ids=[f"layers_{n}" for n in PARTIAL_LAYER_CHECKPOINTS_PRO_1B],
)
def test_image_token_decoder_layers_loop_partial_decode_pro_1b(num_layers: int):
    _run_decoder_partial_op_test("Pro_1B", num_layers)


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_image_token_gen_head_decode_pro_1b():
    _run_gen_head_op_test("Pro_1B")


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.n150
@pytest.mark.p150
def test_image_token_decoder_layer_pcc_profile_pro_1b():
    """Print PCC + abs/L2 diff per decoder layer (one compile); use ``pytest -s``."""
    _run_decoder_layer_pcc_profile("Pro_1B")


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.p150
def test_image_token_decoder_model_forward_decode_pro_7b():
    _run_decoder_op_test("Pro_7B", use_explicit_layers_loop=False)


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.p150
def test_image_token_decoder_layers_loop_full_decode_pro_7b():
    _run_decoder_op_test("Pro_7B", use_explicit_layers_loop=True, num_layers=None)


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.p150
@pytest.mark.parametrize(
    "num_layers",
    PARTIAL_LAYER_CHECKPOINTS_PRO_7B,
    ids=[f"layers_{n}" for n in PARTIAL_LAYER_CHECKPOINTS_PRO_7B],
)
def test_image_token_decoder_layers_loop_partial_decode_pro_7b(num_layers: int):
    _run_decoder_partial_op_test("Pro_7B", num_layers)


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.p150
def test_image_token_gen_head_decode_pro_7b():
    _run_gen_head_op_test("Pro_7B")


@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.p150
def test_image_token_decoder_layer_pcc_profile_pro_7b():
    """Print PCC + abs/L2 diff per decoder layer (one compile); use ``pytest -s``."""
    _run_decoder_layer_pcc_profile("Pro_7B")
