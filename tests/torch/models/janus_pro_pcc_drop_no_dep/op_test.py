# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Fair Forge vs CPU op test for no-dep layer-0 (isolated module per side)."""

from __future__ import annotations

import torch.nn as nn

from tests.torch.models.janus_pro_pcc_drop.decoder_op_test_utils import (
    TensorMatchMetrics,
    run_decoder_stacked_stage_forge_vs_cpu_isolated,
)
from tests.torch.models.janus_pro_pcc_drop.decoder_submodule_sanity import (
    LAYER0_LN_ATTN_STAGE_NAMES_FROM_EMBEDS,
)
from tests.torch.models.janus_pro_pcc_drop_no_dep.arch_specs import Layer0ModuleSpec
from tests.torch.models.janus_pro_pcc_drop_no_dep.build_modules import (
    Layer0LnAttnNoDep,
    build_layer0_no_dep,
)


def _build_model_and_inputs(
    spec: Layer0ModuleSpec,
    *,
    use_saved_inputs: bool,
    load_hf_weights: bool,
) -> tuple[nn.Module, list]:
    bundle = build_layer0_no_dep(
        spec,
        use_saved_inputs=use_saved_inputs,
        load_hf_weights=load_hf_weights,
    )
    wrapper = Layer0LnAttnNoDep(bundle)
    return wrapper, [wrapper.inputs_embeds_decode]


def run_layer0_ln_attn_forge_vs_cpu_isolated(
    label: str,
    spec: Layer0ModuleSpec,
    *,
    use_saved_inputs: bool,
    load_hf_weights: bool,
    assert_on_failure: bool = False,
    return_metrics: bool = False,
) -> list[tuple[str, TensorMatchMetrics]] | None:
    """Fresh ``build_layer0_no_dep`` bundle for CPU and for Forge."""

    def build() -> tuple[nn.Module, list]:
        return _build_model_and_inputs(
            spec,
            use_saved_inputs=use_saved_inputs,
            load_hf_weights=load_hf_weights,
        )

    return run_decoder_stacked_stage_forge_vs_cpu_isolated(
        label,
        build,
        LAYER0_LN_ATTN_STAGE_NAMES_FROM_EMBEDS,
        assert_on_failure=assert_on_failure,
        return_metrics=return_metrics,
    )
