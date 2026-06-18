# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Minimal E2E repro of the DeepSeek-V3.2 vLLM MoE shared-experts crash.

Real failure (deepseek_v3_2_vllm_failing.log):

    RuntimeError: !at::functionalization::impl::isFunctionalTensor(t)
    INTERNAL ASSERT FAILED ... The composite op functionalization fallback
    expects its inputs all not to be functional tensors

vLLM registers ``vllm.moe_forward_shared`` as a custom op with
``mutates_args=["hidden_states"]``, so ``torch.compile`` wraps it in
``auto_functionalized_v2``. torch_xla's CPU-fallback collector
(``partition_fx_graph_for_cpu_fallback`` -> ``collector.run``) re-executes that
HOP eagerly under functionalization; inside, the DeepSeek shared-experts MLP
(our ``XlaMergedColumnParallelLinear``) runs ``F.linear(functional_input,
plain_weight)`` and the composite-op functionalization fallback asserts.

vLLM avoids this on TPU/CPU by selecting the *raw python* MoE function instead
of the custom op (DefaultMoERunner._select_forward). TT is PlatformEnum.OOT, so
it got the custom-op path -> the crash. The fix is to make TT use the raw
function like the other XLA backends.

This script reproduces the exact assert with a tiny model and shows the fix
(raw inline call) makes it pass. It runs on the torch_xla CPU backend because the
failing assert is in torch_xla's *generic* dynamo bridge, identical across PJRT
backends -- so it needs no TT hardware.

Run:
    XLA_REGISTER_INSTALLED_PLUGINS=0 PJRT_DEVICE=CPU python repro_moe_functionalize.py
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_xla

# Registers the tt custom ops and (for faithfulness to the real traceback) the
# global TorchFunctionMode in tt_torch.torch_overrides. Importing tt_torch sets
# PJRT_DEVICE=TT as a side effect, so re-assert CPU below before the runtime
# initializes: we drive this repro on the CPU PJRT backend (the failing assert is
# in torch_xla's *generic* dynamo bridge, identical across PJRT backends, so no TT
# hardware is needed). Run with XLA_REGISTER_INSTALLED_PLUGINS=0 so torch_xla does
# not auto-select the installed "tt" entry-point plugin.
import tt_torch  # noqa: F401

os.environ["PJRT_DEVICE"] = "CPU"

DEV = torch_xla.device()


class XlaMergedColumnParallelLinearLike(nn.Module):
    """Mirror of vllm_distributed_utils.XlaMergedColumnParallelLinear: the
    per-split weights live in a *plain Python list* (not registered as
    nn.Parameters), and forward does a per-split F.linear then concatenates."""

    def __init__(self, in_features, output_sizes):
        super().__init__()
        self.output_sizes = output_sizes
        self.weights = []  # plain list, exactly like the real class
        for o in output_sizes:
            self.weights.append(
                nn.Parameter(torch.randn(o, in_features, device=DEV), requires_grad=False)
            )

    def forward(self, x):
        projs = [F.linear(x, self.weights[i], None) for i in range(len(self.output_sizes))]
        return torch.cat(projs, dim=-1)


# Registry so the opaque custom op body can fetch its module by name, mirroring
# vLLM's forward_context.no_compile_layers lookup inside _moe_forward_shared.
_SHARED_EXPERTS: dict[str, nn.Module] = {}


def _moe_forward_shared_impl(
    hidden_states: torch.Tensor, shared_experts_input: torch.Tensor, layer_name: str
) -> torch.Tensor:
    """Plain python MoE-shared function -- the analogue of vLLM's
    ``_moe_forward_shared`` (the raw function the TPU/CPU branch uses inline)."""
    mlp = _SHARED_EXPERTS[layer_name]
    return mlp(shared_experts_input)


@torch.library.custom_op("ttrepro::moe_forward_shared", mutates_args=["hidden_states"])
def moe_forward_shared(
    hidden_states: torch.Tensor, shared_experts_input: torch.Tensor, layer_name: str
) -> torch.Tensor:
    """Analogue of ``torch.ops.vllm.moe_forward_shared``: a custom op declaring it
    mutates ``hidden_states`` (so it is auto-functionalized), whose opaque body
    runs the shared-experts MLP."""
    out = _moe_forward_shared_impl(hidden_states, shared_experts_input, layer_name)
    hidden_states.mul_(1.0)  # in-place write, justifying mutates_args
    return out


@moe_forward_shared.register_fake
def _(hidden_states, shared_experts_input, layer_name):
    mlp = _SHARED_EXPERTS[layer_name]
    out_dim = sum(mlp.output_sizes)
    return shared_experts_input.new_empty((*shared_experts_input.shape[:-1], out_dim))


class Model(nn.Module):
    def __init__(self, hidden, output_sizes, use_custom_op):
        super().__init__()
        self.mlp = XlaMergedColumnParallelLinearLike(hidden, output_sizes)
        _SHARED_EXPERTS["layer0"] = self.mlp
        self.use_custom_op = use_custom_op

    def forward(self, hidden_states, shared_input):
        if self.use_custom_op:
            # Buggy path: TT (OOT) selects the custom op -> auto_functionalized.
            shared_out = torch.ops.ttrepro.moe_forward_shared(
                hidden_states, shared_input, "layer0"
            )
        else:
            # Fixed path: raw python function inline, like vLLM's TPU/CPU branch.
            shared_out = _moe_forward_shared_impl(hidden_states, shared_input, "layer0")
        return shared_out + 1.0


HIDDEN = 64
OUTPUT_SIZES = [128, 128]  # gate, up -> gate_up output dim 256
SEQ = 16


def run(use_custom_op: bool) -> torch.Tensor:
    torch._dynamo.reset()
    model = Model(HIDDEN, OUTPUT_SIZES, use_custom_op)
    compiled = torch.compile(model, backend="openxla")
    hs = torch.randn(SEQ, HIDDEN, device=DEV)
    si = torch.randn(SEQ, HIDDEN, device=DEV)
    out = compiled(hs, si)
    return out.cpu()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "buggy"
    use_custom_op = mode != "fixed"
    label = "BUGGY (custom op / auto_functionalized)" if use_custom_op else "FIXED (raw inline fn)"
    print(f"=== Running {label} ===", flush=True)
    try:
        out = run(use_custom_op)
        print(f"SUCCESS: output shape {tuple(out.shape)}, sum {float(out.sum()):.4f}", flush=True)
        return 0
    except RuntimeError as e:
        msg = str(e).splitlines()[0]
        print(f"FAILED with RuntimeError: {msg}", flush=True)
        if "isFunctionalTensor" in str(e) or "functionalization fallback" in str(e):
            print(">>> Reproduced the DeepSeek MoE functionalization assert.", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
