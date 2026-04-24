# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch

from tt_torch.backend.passes import run_selected_fusion_passes
from tt_torch.backend.quetzal_rewrite import (
    QUETZAL_REWRITE_PASSES_ENV,
    get_quetzal_rewrite_passes,
)


def test_get_quetzal_rewrite_passes_reads_env(monkeypatch):
    monkeypatch.setenv(QUETZAL_REWRITE_PASSES_ENV, "all")

    passes = get_quetzal_rewrite_passes(None)

    assert "fuse_gelu" in passes
    assert "reconstruct_sdpa" in passes


def test_run_selected_fusion_passes_fuses_tanh_gelu():
    def decomposed_gelu(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (
            1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x**3))
        )

    gm = torch.fx.symbolic_trace(decomposed_gelu)
    replacements = run_selected_fusion_passes(gm, ["fuse_gelu"])

    assert replacements == {"fuse_gelu": 1}

    targets = [node.target for node in gm.graph.nodes if node.op == "call_function"]
    assert torch.nn.functional.gelu in targets


def test_run_selected_fusion_passes_reconstructs_sdpa():
    def manual_sdpa(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        scale = 0.125
        return torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1) @ v

    gm = torch.fx.symbolic_trace(manual_sdpa)
    replacements = run_selected_fusion_passes(gm, ["reconstruct_sdpa"])

    assert replacements == {"reconstruct_sdpa": 1}

    targets = [node.target for node in gm.graph.nodes if node.op == "call_function"]
    assert torch.nn.functional.scaled_dot_product_attention in targets
