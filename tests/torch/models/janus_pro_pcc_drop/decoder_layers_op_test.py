# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Isolated Forge vs CPU op tests for full decoder and per-layer decode."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch_xla.runtime as xr
from infra.evaluators import ComparisonConfig, TorchComparisonEvaluator
from tests.torch.models.janus_pro_pcc_drop.decoder_op_test_utils import (
    TensorMatchMetrics,
    _tensor_match_metrics,
    run_forge_vs_cpu_op_test_isolated,
)
from tests.torch.models.janus_pro_pcc_drop.decoder_sanity import (
    DecoderLayerIsolatedContext,
    JanusLlamaDecoderDecodeLoop,
    JanusLlamaDecoderLayersLoop,
    JanusLlamaDecoderLayersPCCProfile,
    JanusLlamaDecoderSingleLayerIsolated,
    compute_decoder_layer_isolated_context,
    load_image_token_decode_bundle,
)
from tests.torch.models.janus_pro_pcc_drop.decoder_submodule_sanity import (
    clone_dynamic_cache,
)


def _decode_bundle_build_factory(
    decode_bundle: dict,
    *,
    use_explicit_layers_loop: bool,
    num_layers: int | None,
) -> Callable[[], tuple[nn.Module, list[torch.Tensor]]]:
    llama_model = decode_bundle["llama_model"]
    inputs_embeds = decode_bundle["inputs_embeds"]
    kv_template = decode_bundle["past_key_values"]

    def build() -> tuple[nn.Module, list[torch.Tensor]]:
        if use_explicit_layers_loop:
            wrapper: nn.Module = JanusLlamaDecoderLayersLoop(
                llama_model, num_layers=num_layers
            )
        else:
            wrapper = JanusLlamaDecoderDecodeLoop(llama_model)
        wrapper.past_key_values = clone_dynamic_cache(kv_template)
        return wrapper, [inputs_embeds]

    return build


def run_full_decoder_forge_vs_cpu_isolated(
    decode_bundle: dict,
    *,
    use_explicit_layers_loop: bool,
    num_layers: int | None = None,
    label: str,
    assert_on_failure: bool = True,
) -> None:
    xr.set_device_type("TT")
    run_forge_vs_cpu_op_test_isolated(
        _decode_bundle_build_factory(
            decode_bundle,
            use_explicit_layers_loop=use_explicit_layers_loop,
            num_layers=num_layers,
        ),
        label=label,
        assert_on_failure=assert_on_failure,
    )


@dataclass(frozen=True)
class DecoderLayerProfileRow:
    pcc: float
    max_abs_diff: float
    mean_abs_diff: float
    rel_l2_diff: float


def _metrics_for_pair(
    evaluator: TorchComparisonEvaluator,
    device_tensor: torch.Tensor,
    golden_tensor: torch.Tensor,
) -> DecoderLayerProfileRow:
    m = _tensor_match_metrics(evaluator, device_tensor, golden_tensor)
    return DecoderLayerProfileRow(
        pcc=m.pcc,
        max_abs_diff=m.max_abs_diff,
        mean_abs_diff=m.mean_abs_diff,
        rel_l2_diff=m.rel_l2_diff,
    )


def run_decoder_cumulative_layer_profile_isolated(
    decode_bundle: dict,
    *,
    variant_name: str,
) -> dict[int, DecoderLayerProfileRow]:
    """
    Cumulative decode: PCC after each layer depth (isolated CPU vs isolated Forge stacks).
    """
    llama_model = decode_bundle["llama_model"]
    num_hidden_layers = llama_model.config.num_hidden_layers
    evaluator = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))

    def build_stacked() -> tuple[nn.Module, list[torch.Tensor]]:
        wrapper = JanusLlamaDecoderLayersPCCProfile(llama_model)
        wrapper.past_key_values = clone_dynamic_cache(decode_bundle["past_key_values"])
        return wrapper, [decode_bundle["inputs_embeds"]]

    cpu_wrapper, cpu_inputs = build_stacked()
    from infra.utilities import Framework
    from infra.workloads.torch_workload import TorchWorkload
    from tests.infra.testers.single_chip.op.op_tester import OpTester

    tester = OpTester(
        comparison_config=ComparisonConfig(assert_on_failure=False),
        framework=Framework.TORCH,
    )
    xr.set_device_type("TT")
    cpu_stacked = tester._device_runner.run_on_cpu(
        TorchWorkload(model=cpu_wrapper, args=cpu_inputs)
    )
    forge_wrapper, forge_inputs = build_stacked()
    forge_wl = TorchWorkload(model=forge_wrapper, args=forge_inputs)
    tester._compile_for_tt_device(forge_wl)
    forge_stacked = tester._device_runner.run_on_tt_device(forge_wl)

    rows: dict[int, DecoderLayerProfileRow] = {}
    for depth in range(1, num_hidden_layers + 1):
        rows[depth] = _metrics_for_pair(
            evaluator,
            forge_stacked[depth - 1],
            cpu_stacked[depth - 1],
        )

    print(f"\n=== Cumulative decoder profile isolated ({variant_name}) ===")
    print(
        f"{'layer':>6}  {'pcc':>10}  {'max_abs':>12}  {'mean_abs':>12}  {'rel_l2':>10}"
    )
    print("-" * 60)
    for depth, row in rows.items():
        print(
            f"{depth:>6}  {row.pcc:10.6f}  {row.max_abs_diff:12.6e}  "
            f"{row.mean_abs_diff:12.6e}  {row.rel_l2_diff:10.6e}"
        )
    print("=" * 60 + "\n")
    return rows


def run_decoder_standalone_layer_profile_isolated(
    decode_bundle: dict,
    *,
    variant_name: str,
    layer_indices: tuple[int, ...] | None = None,
) -> dict[int, DecoderLayerProfileRow]:
    """
    For each layer ``i``: CPU reference inputs → run **only** layer ``i`` (Forge vs CPU).
    """
    llama_model = decode_bundle["llama_model"]
    num_layers = llama_model.config.num_hidden_layers
    indices = layer_indices if layer_indices is not None else tuple(range(num_layers))
    evaluator = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
    rows: dict[int, DecoderLayerProfileRow] = {}

    contexts: dict[int, DecoderLayerIsolatedContext] = {}
    for layer_idx in indices:
        contexts[layer_idx] = compute_decoder_layer_isolated_context(
            llama_model,
            decode_bundle["inputs_embeds"],
            decode_bundle["past_key_values"],
            layer_idx,
        )

    from infra.utilities import Framework
    from infra.workloads.torch_workload import TorchWorkload
    from tests.infra.testers.single_chip.op.op_tester import OpTester

    tester = OpTester(
        comparison_config=ComparisonConfig(assert_on_failure=False),
        framework=Framework.TORCH,
    )
    xr.set_device_type("TT")

    print(f"\n=== Standalone per-layer decoder isolated ({variant_name}) ===")
    print(
        f"{'layer':>6}  {'pcc':>10}  {'max_abs':>12}  {'mean_abs':>12}  {'rel_l2':>10}"
    )
    print("-" * 60)

    for layer_idx in indices:
        ctx = contexts[layer_idx]

        def build(ctx=ctx) -> tuple[nn.Module, list[torch.Tensor]]:
            layer = llama_model.layers[ctx.layer_idx]
            wrapper = JanusLlamaDecoderSingleLayerIsolated(layer, ctx)
            return wrapper, [ctx.hidden_in.clone()]

        cpu_wrapper, cpu_inputs = build()
        cpu_out = tester._device_runner.run_on_cpu(
            TorchWorkload(model=cpu_wrapper, args=cpu_inputs)
        )
        forge_wrapper, forge_inputs = build()
        forge_wl = TorchWorkload(model=forge_wrapper, args=forge_inputs)
        tester._compile_for_tt_device(forge_wl)
        forge_out = tester._device_runner.run_on_tt_device(forge_wl)
        row = _metrics_for_pair(evaluator, forge_out, cpu_out)
        rows[layer_idx] = row
        print(
            f"{layer_idx:>6}  {row.pcc:10.6f}  {row.max_abs_diff:12.6e}  "
            f"{row.mean_abs_diff:12.6e}  {row.rel_l2_diff:10.6e}"
        )
    print("=" * 60 + "\n")
    return rows


def load_decode_bundle_for_variant(variant_name: str) -> dict:
    import inspect

    import torch
    from tests.runner.requirements import RequirementsManager

    import third_party.tt_forge_models.janus_pro.text_to_image.pytorch.loader as janus_loader
    from third_party.tt_forge_models.janus_pro.text_to_image.pytorch import (
        ModelLoader,
        ModelVariant,
    )

    loader_path = inspect.getsourcefile(janus_loader)
    with RequirementsManager.for_loader(loader_path, framework="torch"):
        from third_party.tt_forge_models.janus_pro.text_to_image.pytorch.src import (
            model_utils,
        )

        torch.manual_seed(42)
        model_utils._mmgpt_cache.clear()
        repo_id = ModelLoader(ModelVariant(variant_name))._repo_id()
        return load_image_token_decode_bundle(repo_id, dtype=torch.bfloat16)
