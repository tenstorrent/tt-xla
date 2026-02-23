#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Standalone repro: FunctionalTensorMode inplace_view metadata fixup crashes on XLA.

Bug location: torch/_subclasses/functional_tensor.py, FunctionalTensorMode.__torch_dispatch__

After functionalization converts an inplace_view op (e.g. aten.as_strided_) to its
functional equivalent, lines 549-554 re-execute the original inplace op under
no_dispatch() to update the FunctionalTensor wrapper's shape/stride metadata:

    if torch.Tag.inplace_view in func.tags ...:
        with torch.utils._mode_utils.no_dispatch():
            func(*args, **kwargs)

The FunctionalTensor wrapper inherits device='xla:0' from its inner tensor (via
_make_wrapper_subclass at line 121-135). Under no_dispatch(), ALL Python dispatch is
disabled, so C++ dispatch sees device='xla:0' and routes to XLA's kernel, which expects
a native XLATensor but receives the wrapper subclass.

For CPU/CUDA this is fine — their as_strided_ kernels are pure metadata mutations on
TensorImpl. XLA (and likely other out-of-tree backends) register kernels that try to
extract their own tensor type, causing the crash.

Real-world trigger: nn.AdaptiveAvgPool1d/2d inside complex models compiled with
torch.compile using a custom aot_autograd backend. ATen's adaptive_avg_pool
implementation calls aten.as_strided_ internally. When this executes during
aot_autograd's run_functionalized_fw_and_collect_metadata, the bug is hit.

Requirements: torch, torch_xla (any XLA backend, not TT-specific)
"""

import traceback

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._subclasses.functional_tensor import FunctionalTensor, FunctionalTensorMode


def low_level_repro(device: str):
    """
    Directly trigger aten.as_strided_ inside FunctionalTensorMode on a FakeTensor.

    This is the minimal reproduction of the bug:
    real tensor -> FakeTensor -> FunctionalTensor -> as_strided_ -> crash (on XLA)
    """
    print(f"\n{'='*70}")
    print(f"LOW-LEVEL REPRO on device='{device}'")
    print(f"{'='*70}")

    # Step 1: Create a real tensor on the target device
    real = torch.randn(2, 3, 4, device=device)
    print(f"  Real tensor device: {real.device}")

    # Step 2: Create a FakeTensor from it (reports same device)
    fake_mode = FakeTensorMode()
    with fake_mode:
        fake = fake_mode.from_tensor(real)
    print(f"  FakeTensor device: {fake.device}")

    # Step 3: Enter FunctionalTensorMode and wrap as FunctionalTensor
    func_mode = FunctionalTensorMode()
    with func_mode:
        func_tensor = FunctionalTensor.to_functional(fake)
        print(f"  FunctionalTensor device: {func_tensor.device}")

        # Step 4: Call an inplace_view op — this triggers the bug on XLA
        # aten.as_strided_ is tagged with torch.Tag.inplace_view.
        # FunctionalTensorMode.__torch_dispatch__ will:
        #   a) functionalize it (line 511-515) — this works fine
        #   b) re-execute under no_dispatch() for metadata fixup (line 553-554) — CRASH on XLA
        try:
            torch.ops.aten.as_strided_(func_tensor, (2, 3, 4), (12, 4, 1), 0)
            print(f"  as_strided_ SUCCEEDED (device={device})")
        except Exception as e:
            print(f"  as_strided_ FAILED (device={device}): {e}")
            traceback.print_exc()


def high_level_repro():
    """
    Trigger the bug by executing a function containing as_strided_ inside
    FunctionalTensorMode + FakeTensorMode — the exact setup aot_autograd uses.

    This demonstrates that ANY function whose execution path includes an
    inplace_view op will crash during aot_autograd's metadata collection
    when the tensors report device='xla:0'.

    Real-world example: nn.AdaptiveAvgPool1d/2d in complex models (e.g. MaskFormer
    Swin) where ATen's C++ pooling implementation calls aten.as_strided_ internally.
    """
    print(f"\n{'='*70}")
    print("HIGH-LEVEL REPRO: function with as_strided_ in FunctionalTensorMode on XLA")
    print(f"{'='*70}")

    def my_forward(x):
        """A forward function that uses as_strided_ (inplace view)."""
        # This is what happens inside ATen's adaptive_avg_pool implementation.
        # as_strided_ is an inplace_view op — it modifies the tensor's view in-place.
        return torch.ops.aten.as_strided_(x, (2, 6), (6, 1), 0)

    # Create FakeTensor with device='xla:0' (same as aot_autograd does)
    real_input = torch.randn(2, 3, 4, device="xla")
    fake_mode = FakeTensorMode()
    with fake_mode:
        fake_input = fake_mode.from_tensor(real_input)

    print(f"  FakeTensor input: shape={fake_input.shape}, device={fake_input.device}")

    # Enter FakeTensorMode + FunctionalTensorMode — this is exactly what
    # aot_autograd's run_functionalized_fw_and_collect_metadata does
    func_mode = FunctionalTensorMode()
    try:
        with fake_mode, func_mode:
            func_input = FunctionalTensor.to_functional(fake_input)
            print(f"  FunctionalTensor input device: {func_input.device}")
            result = my_forward(func_input)
            print(f"  Forward SUCCEEDED, output shape: {result.shape}")
    except Exception as e:
        print(f"  Forward FAILED: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    print("Reproducing FunctionalTensorMode + XLA inplace_view bug")
    print(f"PyTorch version: {torch.__version__}")

    try:
        import torch_xla

        print(f"torch_xla version: {torch_xla.__version__}")
    except ImportError:
        print("torch_xla not installed — XLA repros will fail with device error")

    # CPU should work fine
    low_level_repro("cpu")

    # XLA should crash
    try:
        low_level_repro("xla")
    except Exception as e:
        print(f"  Unexpected top-level error: {e}")

    # High-level repro through metadata collection
    try:
        high_level_repro()
    except Exception as e:
        print(f"  Unexpected top-level error: {e}")

    print(f"\n{'='*70}")
    print("EXPECTED RESULTS:")
    print("  - CPU low-level repro: SUCCEEDS")
    print("  - XLA low-level repro: FAILS with 'Input tensor is not an XLA tensor'")
    print("  - XLA high-level repro: FAILS with same error via as_strided_ in forward")
    print(f"{'='*70}")
