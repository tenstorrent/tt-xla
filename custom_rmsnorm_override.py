# SPDX-License-Identifier: Apache-2.0
"""
Override RMSNorm's *backward* in a small model with a custom tt-lang kernel.

Forward stays native (regular torch ops -> normal StableHLO->TTIR->TTNN lowering).
Only the backward is replaced by the tt-lang `rmsnorm_bw_2pass` kernel from
tt-metal PR #44516:
    https://github.com/tenstorrent/tt-metal/pull/44516
    tt-train/sources/ttml/ttlang/ttl_rmsnorm_bw_2pass.py

Kernel operand contract (in order):
    rmsnorm_bw_2pass(input_t, gamma_t, rms_t, dL_dout_t,     # "in"
                     dL_dinput_out, dL_dgamma_comp_out)      # "out"
      input_t   [T, C]  x, tokens x hidden           (T,C multiples of 32)
      gamma_t   [1, C]  weight (row vector)
      rms_t     [T, 1]  rms = sqrt(mean(x^2, -1) + eps)   <-- saved by forward
      dL_dout_t [T, C]  grad wrt output
      dL_dinput_out       [T, C]  grad wrt x
      dL_dgamma_comp_out  [T, C]  per-token grad-weight COMPONENT
                                  (reduce over T afterwards -> grad_weight [C])

The kernel math (verified == analytic RMSNorm grad):
    r     = 1/rms
    scale = sum_c( x * (gamma*r) * dL_dout )                       # [T,1]
    dL_dinput      = (gamma*r)*dL_dout - scale * x * r^2 * (1/C)
    dL_dgamma_comp = x * r * dL_dout

Run
---
`python custom_rmsnorm_override.py` runs the model on the XLA device: the custom
backward dispatches the real tt-lang kernel (emits stablehlo.custom_call
@tt.tt_lang_op) and every gradient is PCC-compared against a CPU golden.
(On a non-XLA tensor the backward falls back to a torch reference of the same
2-pass math -- see `_rmsnorm_backward`.)

NOTE (backward-only + native forward): we use `torch.autograd.Function` because the
forward must stay transparent (so it lowers/fuses normally). tt-xla's other custom
ops use `torch.library.register_autograd`, but that is for ops that are themselves
opaque custom_calls; it is not the right tool when you want to keep a native forward.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch_xla

torch_xla.set_custom_compile_options({"export_path":"irs"})

import tt_torch  # noqa: F401

# `ttl_rmsnorm_bw_2pass.py` from tt-metal PR #44516 must be importable
# (add tt-train/sources/ttml/ttlang to PYTHONPATH).
from ttl_rmsnorm_bw_2pass import make_kernel

_bw_kernel = make_kernel()  # @ttl.operation(grid="auto", ...)-decorated

# Wrap the tt-lang kernel as a tt-xla custom operation.
# arg_roles must list all "in" operands before the "out" operands.
#
# We wrap `_bw_kernel` in a plain function with the explicit 6-arg
# signature (rather than decorating it directly) so that
# `tt_lang_operation`'s arity introspection (`_positional_arg_count`)
# reliably sees 6 positional args. `@ttl.operation` can hide the
# underlying signature, in which case direct decoration raises at
# registration time ("arg_roles has 6 entries but ... 0 positional
# tensors"). This mirrors examples/pytorch/tt_lang_eltwise_add.py.
@tt_torch.tt_lang_operation(
    operation_id="rmsnorm.bw.2pass.v1",
    arg_roles=("in", "in", "in", "in", "out", "out"),
    version_tag="pr44516-2pass-real",
)
def rmsnorm_bw_op(
    input_t, gamma_t, rms_t, dL_dout_t, dL_dinput_out, dL_dgamma_comp_out
):
    return _bw_kernel(
        input_t, gamma_t, rms_t, dL_dout_t, dL_dinput_out, dL_dgamma_comp_out
    )

def _rmsnorm_bw_reference(x, gamma_row, rms, dL_dout):
    """Torch reference of the exact kernel math. gamma_row: [1, C]; rms: [T, 1]."""
    C = x.shape[-1]
    r = 1.0 / rms
    scale = (x * (gamma_row * r) * dL_dout).sum(-1, keepdim=True)  # [T, 1]
    dL_dinput = (gamma_row * r) * dL_dout - scale * x * (r * r) * (1.0 / C)
    dL_dgamma_comp = x * r * dL_dout
    return dL_dinput, dL_dgamma_comp


def _rmsnorm_backward(x, gamma_row, rms, dL_dout):
    """Dispatch to the tt-lang kernel on XLA, else the torch reference.

    All tensors are 2D: x/dL_dout [T, C], gamma_row [1, C], rms [T, 1].
    Returns (dL_dinput [T, C], dL_dgamma_comp [T, C]).
    """
    if x.device.type == "xla":
        # The kernel is authored for bf16 TILE_LAYOUT operands (see PR #44516).
        # Feeding f32 trips a tt-metal unpack-mode constraint: an f32 CB that
        # feeds both an FPU matmul and an SFPU elementwise consumer needs two
        # mutually-exclusive unpack modes. bf16 avoids that; fp32_dest_acc_en
        # still accumulates the reduction in f32, so accuracy stays high.
        orig_dtype = x.dtype
        xb = x.to(torch.bfloat16)
        gb = gamma_row.to(torch.bfloat16)
        rb = rms.to(torch.bfloat16)
        db = dL_dout.to(torch.bfloat16)
        dL_dinput = torch.empty_like(xb)
        dL_dgamma_comp = torch.empty_like(xb)
        # Mutation-style: the decorated kernel writes into the "out" buffers.
        rmsnorm_bw_op(xb, gb, rb, db, dL_dinput, dL_dgamma_comp)
        return dL_dinput.to(orig_dtype), dL_dgamma_comp.to(orig_dtype)
    return _rmsnorm_bw_reference(x, gamma_row, rms, dL_dout)


# --------------------------------------------------------------------------
# autograd.Function: native forward, custom backward.
# --------------------------------------------------------------------------
class _CustomRMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])                       # [T, C]
        # rms EXACTLY as the kernel expects it (sqrt of mean-square + eps).
        rms = torch.sqrt(x2d.pow(2).mean(-1, keepdim=True) + eps)  # [T, 1]
        out = (x2d / rms) * weight.reshape(1, -1)
        ctx.save_for_backward(x2d, weight, rms)
        ctx.orig_shape = orig_shape
        return out.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_out):
        x2d, weight, rms = ctx.saved_tensors
        C = x2d.shape[-1]
        g2d = grad_out.reshape(-1, C)                              # [T, C]
        gamma_row = weight.reshape(1, C)                           # [1, C]
        dL_dinput, dL_dgamma_comp = _rmsnorm_backward(x2d, gamma_row, rms, g2d)
        grad_weight = dL_dgamma_comp.sum(0)                        # reduce T -> [C]
        grad_x = dL_dinput.reshape(ctx.orig_shape)
        return grad_x, grad_weight, None                           # None for eps


def custom_rms_norm(x, weight, eps=1e-6):
    return _CustomRMSNormFn.apply(x, weight, eps)


class CustomRMSNorm(nn.Module):
    """Drop-in RMSNorm whose backward runs the tt-lang kernel."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return custom_rms_norm(x, self.weight, self.eps)


# --------------------------------------------------------------------------
# A small example model that uses RMSNorm, and a swap helper.
# --------------------------------------------------------------------------
class RefRMSNorm(nn.Module):
    """Stand-in for a stock (e.g. HF LlamaRMSNorm) norm: native forward + autograd."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class TinyBlock(nn.Module):
    """Pre-norm MLP block: norm -> linear -> gelu -> linear -> residual."""

    def __init__(self, dim, hidden, norm_cls=RefRMSNorm):
        super().__init__()
        self.norm = norm_cls(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.norm(x))))


class TinyModel(nn.Module):
    """Stack of pre-norm blocks at *varying* widths.

    Each block runs its RMSNorm at a different hidden size, so the tt-lang
    backward kernel is resolved for multiple ``[T, C]`` operand shapes -- this is
    what exercises the per-shape resolve/compile cache. Blocks are bridged by a
    linear projection to change the residual-stream width between them. The
    ``head`` norm reuses the last block's width so at least one shape recurs
    (exercising a cache *hit* as well as misses).

    ``dims[i]`` is the width (RMSNorm ``C``) of block ``i``; input must be
    ``[..., dims[0]]``. All dims must be multiples of 32 (kernel tile constraint).
    """

    def __init__(self, dims=(128, 256, 64), hidden_mult=2, norm_cls=RefRMSNorm):
        super().__init__()
        self.dims = tuple(dims)
        self.projs = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i, d in enumerate(self.dims):
            # Identity into the first block (input already at dims[0]); a real
            # projection between successive blocks of differing width.
            self.projs.append(
                nn.Identity() if i == 0 else nn.Linear(self.dims[i - 1], d)
            )
            self.blocks.append(TinyBlock(d, d * hidden_mult, norm_cls))
        self.head = norm_cls(self.dims[-1])

    def forward(self, x):
        for proj, block in zip(self.projs, self.blocks):
            x = block(proj(x))
        return self.head(x)


def swap_rmsnorm(model: nn.Module, target_cls=RefRMSNorm) -> nn.Module:
    """Replace every `target_cls` instance with CustomRMSNorm, preserving weights.

    Mirrors the tt-xla named_modules/setattr pattern (see sparse_mlp.py). For a real
    HF model, set target_cls = transformers...LlamaRMSNorm and copy .variance_epsilon.
    """
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, target_cls):
                repl = CustomRMSNorm(child.weight.shape[0], child.eps)
                repl.weight = child.weight  # reuse the SAME Parameter (keeps optim/state_dict)
                setattr(parent, name, repl)
    return model


def _pcc(a, b):
    """Pearson correlation between two flattened tensors (tt-style accuracy gate)."""
    a = a.detach().reshape(-1).float()
    b = b.detach().reshape(-1).float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _build_pair(dims, hidden_mult=2):
    """A reference model (stock autograd) and a copy with RMSNorm backward overridden."""
    ref = TinyModel(dims, hidden_mult, norm_cls=RefRMSNorm)
    # RMSNorm weights default to all-ones. Combined with the `square().mean()`
    # loss on the final norm's output, that makes the top norm effectively
    # scale-invariant: RMSNorm's Jacobian nulls the radial component the loss
    # gradient points along, so the gradient to every upstream parameter
    # vanishes (~1e-9, i.e. numerical noise). PCC between two independent noise
    # vectors is meaningless, so the accuracy gate degenerates regardless of
    # kernel correctness (the fully-native reference fails it too). Give the
    # norm weights non-trivial values so every parameter carries a real,
    # comparable gradient and the gate actually tests the kernel.
    with torch.no_grad():
        for m in ref.modules():
            if isinstance(m, RefRMSNorm):
                m.weight.copy_(torch.randn_like(m.weight))
    cust = TinyModel(dims, hidden_mult, norm_cls=RefRMSNorm)
    cust.load_state_dict(ref.state_dict())
    swap_rmsnorm(cust)
    return ref, cust


# --------------------------------------------------------------------------
# Hardware path: run the override on the XLA device; the tt-lang kernel is
# dispatched inside backward. Grads are PCC-compared against the CPU golden.
# --------------------------------------------------------------------------
def _run_xla(pcc_gate=0.99):
    import torch_xla.core.xla_model as xm

    torch.manual_seed(0)
    # Blocks of differing width -> the backward kernel is resolved for several
    # [T, C] shapes (C in {128, 256, 64}); the last recurs via `head` to also
    # hit the cache. All dims and T = B*S must be multiples of 32.
    dims = (128, 256, 64)
    B, S = 4, 32
    dev = xm.xla_device()

    # CPU golden (stock autograd); cust gets the tt-lang backward override.
    ref, cust = _build_pair(dims)
    x_cpu = torch.randn(B, S, dims[0])
    xr = x_cpu.clone().requires_grad_(True)
    ref(xr).square().mean().backward()
    golden = {"input": xr.grad, **{n: p.grad for n, p in ref.named_parameters()}}

    # Move the overridden model to XLA and run the same step on silicon.
    cust = cust.to(dev)
    x = x_cpu.clone().to(dev).requires_grad_(True)
    loss = cust(x).square().mean()
    loss.backward()
    xm.mark_step()                            # trace + compile + execute on device

    got = {"input": x.grad.cpu(), **{n: p.grad.cpu() for n, p in cust.named_parameters()}}
    ok = True
    for name, ref_g in golden.items():
        pcc = _pcc(got[name], ref_g)
        ok &= pcc >= pcc_gate
        print(f"[xla] {'OK ' if pcc >= pcc_gate else '!! '}{name:<24} PCC : {pcc:.5f}")
    assert ok, f"one or more grads below PCC gate {pcc_gate}"
    print("\nHardware path validated: RMSNorm backward ran the tt-lang kernel on device.")


if __name__ == "__main__":
    _run_xla()
