# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Numerical-correctness helpers: PCC / relative-L2 math and the on-device
PCC assertion policy used by the vision / encoder / llm benchmarks.

Pure and device-free — operates on tensors only.
"""

import torch


def compute_pcc(golden_output: torch.Tensor, device_output: torch.Tensor) -> float:
    """Pearson correlation coefficient of two tensors, in [-1, 1].

    Returns 1.0 when the denominator is zero but the tensors are close.
    Raises ``ValueError`` if the denominator is zero and they are not.
    """
    golden_flat = golden_output.to(torch.float32).flatten()
    device_flat = device_output.to(torch.float32).flatten()

    golden_centered = golden_flat - golden_flat.mean()
    device_centered = device_flat - device_flat.mean()
    denom = golden_centered.norm() * device_centered.norm()

    if denom == 0:
        if torch.allclose(golden_flat, device_flat, rtol=1e-2, atol=1e-2):
            return 1.0
        raise ValueError(
            "PCC computation failed: denominator is zero but tensors are not close"
        )

    pcc = ((golden_centered @ device_centered) / denom).item()
    return max(-1.0, min(1.0, pcc))  # clamp away float rounding past +-1


def compute_rel_l2(golden_output: torch.Tensor, device_output: torch.Tensor) -> float:
    """Compute relative L2 error between two tensors.

    rel_l2 = ||device_output - golden_output||_2 / ||golden_output||_2

    Computed in float64 to avoid norm underflow in bf16/fp32.

    Complements PCC, which has blind spots:
    - PCC is scale-blind
    - max-atol is dominated by a single outlier
    - max-rtol blows up near zero
    rel_l2 is scale-sensitive, stable near zero globally (denominator is the
    golden norm, not per-element |y|), and captures distributed degradation
    rather than one bad element.

    Return value:
    - 0.0 if both tensors are exactly zero
    - ``inf`` if only the golden norm is zero
    - ``nan`` if either tensor contains a NaN
    - otherwise the non-negative relative error
    """
    golden_flat = golden_output.to(torch.float64).flatten()
    device_flat = device_output.to(torch.float64).flatten()

    diff_norm = torch.linalg.vector_norm(device_flat - golden_flat)
    golden_norm = torch.linalg.vector_norm(golden_flat)

    if torch.isnan(diff_norm) or torch.isnan(golden_norm):
        return float("nan")

    if golden_norm.item() == 0.0:
        return 0.0 if diff_norm.item() == 0.0 else float("inf")

    return (diff_norm / golden_norm).item()


def assert_pcc(device_output, golden_output, required_pcc: float) -> float:
    """Compute PCC against the CPU golden and assert it meets ``required_pcc``."""
    pcc_value = compute_pcc(device_output, golden_output)
    assert (
        pcc_value >= required_pcc
    ), f"PCC comparison failed. PCC={pcc_value:.6f}, Required={required_pcc}"
    print(f"PCC verification passed with PCC={pcc_value:.6f}")
    return pcc_value
