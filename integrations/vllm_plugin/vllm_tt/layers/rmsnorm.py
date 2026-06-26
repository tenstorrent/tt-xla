# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC

import os

import torch
import torch.nn as nn
from vllm.model_executor.layers.layernorm import RMSNorm

# [#5140 A/B toggle] When TTXLA_REVERT_RMSNORM_FUSION_5140 is set, revert the two
# behavioral changes #5140 made to this module (operand order of the weight
# multiply, and the .mean() signature) to their pre-#5140 form. Paired with the
# gate in tt_torch/fusion_providers.py, this lets a branch rebased on top of #5140
# be A/B-compared against the parent's behavior without checking out the parent
# commit (which would drop unreleased work). See ISSUE_*rmsnorm_fusion*.
REVERT_RMSNORM_FUSION_5140 = os.environ.get(
    "TTXLA_REVERT_RMSNORM_FUSION_5140", "0"
).lower() not in ("0", "", "false", "no")


class TTRMSNorm(nn.Module):
    """TT-compatible RMSNorm replacement for vLLM's RMSNorm.

    vLLM's RMSNorm.forward_native accesses `self.weight.data`, which causes an
    AssertionError during torch.compile/torch.export tracing with FakeTensors.
    Accessing `.data` on a FakeTensor lifts it out of the fake tensor context,
    resulting in: "cannot call `.data` on a Tensor, the Tensor is a FakeTensor".

    This class reimplements the RMSNorm forward pass using `self.weight` directly
    (without `.data`), making it compatible with TT tracing and compilation.
    """

    def __init__(self, layer: nn.Module):
        super().__init__()
        assert isinstance(layer, RMSNorm)
        self.hidden_size = layer.hidden_size
        self.variance_epsilon = layer.variance_epsilon
        self.variance_size_override = layer.variance_size_override
        self.has_weight = layer.has_weight
        self.weight = layer.weight

        if hasattr(layer, "rocm_norm_func") and hasattr(
            layer, "rocm_norm_func_with_add"
        ):
            self.rocm_norm_func = layer.rocm_norm_func
            self.rocm_norm_func_with_add = layer.rocm_norm_func_with_add

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        orig_dtype = x.dtype
        x = x.to(torch.float32)
        if residual is not None:
            # residual promoted f16->f32 automatically,
            # otherwise Inductor eliminates the casts to and from f16,
            # increasing memory usage (and complicating pattern matching)
            x = x + residual
            residual = x.to(orig_dtype)

        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden_size to be {self.hidden_size}, but found: {x.shape[-1]}"
            )

        if self.variance_size_override is None:
            x_var = x
        else:
            if self.hidden_size < self.variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{self.variance_size_override}, but found: {self.hidden_size}"
                )

            x_var = x[:, :, : self.variance_size_override]

        if REVERT_RMSNORM_FUSION_5140:
            variance = x_var.pow(2).mean(dim=-1, keepdim=True)
        else:
            variance = x_var.pow(2).mean(-1, keepdim=True)

        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = x.to(orig_dtype)
        if self.has_weight and self.weight is not None:
            x = x * self.weight if REVERT_RMSNORM_FUSION_5140 else self.weight * x
        if residual is None:
            return x
        return x, residual


def override_rmsnorm_module(layer: torch.nn.Module) -> torch.nn.Module:
    assert isinstance(layer, RMSNorm)
    return TTRMSNorm(layer)
