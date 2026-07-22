# Handoff — custom RMSNorm backward via tt-lang on tt-xla

Self-contained context for running `custom_rmsnorm_override.py` on a Tenstorrent box
(target: `wh-lb-38`). No external doc needed.

## Goal
Take a **standard RMS norm** in a small model, keep its **forward native**, and override
**only the backward** with a custom tt-lang kernel — running on silicon, inside the XLA graph.
This proves the "customer-authored custom backward op" path end to end.

## The two moving pieces
1. **tt-xla forward mechanism** (merged PR tenstorrent/tt-xla#5539): `@tt_torch.tt_lang_operation`
   wraps a tt-lang `@ttl.operation` kernel, emits `stablehlo.custom_call @tt.tt_lang_op`, and a
   tt-mlir pass (`--ttnn-resolve-tt-lang-kernels`) JITs the kernel at the TTNN stage. To the
   compiler, **a backward kernel is just another forward kernel** keyed by `operation_id`.
2. **The backward kernel** (tt-metal PR tenstorrent/tt-metal#44516): `ttl_rmsnorm_bw_2pass.py`,
   a 2-pass tile-streaming RMSNorm backward (~1.5–2× faster than the ttml/metal version).

## Why `autograd.Function` (not `register_autograd`)
Forward must stay **transparent** so it lowers/fuses normally (native RMS norm → standard HLO /
`ttnn.rms_norm`). A `torch.library.custom_op` would make the forward **opaque** and need its own
tt-mlir lowering. So we use a `torch.autograd.Function`: native forward + custom backward. The
tt-lang op only runs in **backward** (under `no_grad`), so tt_lang_op's "autograd-raises" hook
is never on a live path — the one #5539 blocker doesn't apply here.

## Backward kernel operand contract (from PR #44516)
```
rmsnorm_bw_2pass(input_t, gamma_t, rms_t, dL_dout_t,     # "in"
                 dL_dinput_out, dL_dgamma_comp_out)      # "out"
  input_t   [T, C]  x, tokens x hidden      (T and C multiples of 32)
  gamma_t   [1, C]  weight (row vector)
  rms_t     [T, 1]  rms = sqrt(mean(x^2,-1) + eps)   <-- forward MUST save this
  dL_dout_t [T, C]  grad wrt output
  dL_dinput_out       [T, C]  grad wrt x
  dL_dgamma_comp_out  [T, C]  per-token grad-weight COMPONENT
                              -> sum over T afterwards to get grad_weight [C]
```
Math (verified == analytic RMSNorm gradient, no mean-subtraction):
```
r     = 1/rms
scale = sum_c( x * (gamma*r) * dL_dout )
dL_dinput      = (gamma*r)*dL_dout - scale * x * r^2 * (1/C)
dL_dgamma_comp = x * r * dL_dout
```
Mapped in `custom_rmsnorm_override.py`: forward saves `(x2d, weight, rms)`; backward dispatches
the kernel with those + `grad_out`, then `grad_weight = dL_dgamma_comp.sum(0)`.

## Files (copy both into one dir, add it to PYTHONPATH)
- `custom_rmsnorm_override.py` — override + `TinyModel` + `swap_rmsnorm()` + runners.
- `ttl_rmsnorm_bw_2pass.py` — the kernel (self-contained: imports only `ttl`, `ttnn`, `torch`).
- `HANDOFF.md` — this file.

## Prereqs on the box (verify first — de-risk)
- `python -c "import tt_torch, tt_torch.tt_lang"` works (tt-xla built with #5539).
- Its tt-mlir submodule defines `ttnn.tt_lang_op` + the resolve/lower passes.
- **`tt_torch.tt_lang.resolve_operation` is functional, not the `NotImplementedError` stub.**
  If it's a stub, even forward tt-lang ops can't run — hard blocker.
- `ttl` (tt-lang) + `ttnn` + `torch_xla` importable in the same venv.

## Run
```bash
# CPU wiring/grad-check (no hardware; validates the math + reshape plumbing):
python custom_rmsnorm_override.py

# On device (dispatches the tt-lang kernel in backward; PCC vs CPU golden):
cd <dir-with-both-files>
source <tt-xla-venv>/bin/activate
PYTHONPATH=$PWD TT_VISIBLE_DEVICES=0 python custom_rmsnorm_override.py --xla
```
Default shapes: `dim=128, hidden=256, depth=2, B=4, S=32` → tokens `T=128`, hidden `C=128`
(both multiples of 32, as the kernel requires).

## Expected result
`--xla`: every grad (input + each parameter) reports **PCC ≥ 0.99** vs the CPU golden, and it
prints "RMSNorm backward ran the tt-lang kernel on device."

## Known unknowns / likely failure points
- **Arity introspection**: `make_kernel()` returns an `@ttl.operation`-decorated object;
  `tt_lang_operation` calls `_positional_arg_count(fn)` + `_normalize_arg_roles`. If the ttl
  wrapper hides the 6-arg signature, this raises — may need a thin wrapper that re-exposes the
  positional signature before applying `@tt_torch.tt_lang_operation`.
- **Kernel assumptions**: `bc=2`, `get_block_size(wt, 2)`, `grid="auto"`, `fp32_dest_acc_en`,
  and the `TILE_WIDTH=32` divisibility. If the 128×128 shape misbehaves, try larger hidden
  (e.g. 2048/4096 as in the PR benchmark) — but keep it small enough to iterate.
- **Layout / dtype**: kernel authored for bf16 TILE_LAYOUT DRAM tensors; tt-xla feeds the
  resolved TTNN layout. If resolve/lower complains about operand layout, that's the T6 layout
  hand-off in #5539 — check the `kernel_artifact` JSON.
- **grad_weight reduction**: `dL_dgamma_comp.sum(0)` is a native op in our backward (not in the
  kernel). Confirm it lands in the same XLA program (it should).
- **mark_step**: `--xla` calls `xm.mark_step()` once after `backward()`; if grads look empty,
  the trace didn't flush — check torch_xla is actually driving the device.

## Provenance
- tt-xla mechanism: https://github.com/tenstorrent/tt-xla/pull/5539 (issue #4813)
- backward kernel: https://github.com/tenstorrent/tt-metal/pull/44516
  branch `lgalasTT/benchmark_ttlang_vs_ttml_rmsnorm_bw`
