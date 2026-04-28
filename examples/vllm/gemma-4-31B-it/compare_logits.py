# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Compare per-step logit dumps from two vLLM runs.

Typical workflow
----------------
1. Run the server with optimization_level=0 (known-good) and dump logits::

       --additional-config '{"optimization_level": 0, "debug_dump_logits_dir": "/tmp/logits_opt0"}'

2. Run again with the failing optimization level::

       --additional-config '{"optimization_level": 1, "debug_dump_logits_dir": "/tmp/logits_opt1"}'

3. Compare::

       python examples/vllm/gemma-4-31B-it/compare_logits.py /tmp/logits_opt0 /tmp/logits_opt1

The script prints per-step statistics and highlights the first step where the
top-1 predicted token diverges.
"""

import argparse
import glob
import os
import sys

import torch


def pcc(ref: torch.Tensor, cmp: torch.Tensor) -> float:
    """Pearson Correlation Coefficient between two same-shape tensors."""
    x, y = ref.flatten().float(), cmp.flatten().float()
    vx, vy = x - x.mean(), y - y.mean()
    denom = vx.norm() * vy.norm()
    if denom == 0:
        return float("nan")
    return (vx @ vy / denom).item()


def load_steps(directory: str) -> dict[int, torch.Tensor]:
    files = sorted(glob.glob(os.path.join(directory, "step*_logits.pt")))
    if not files:
        sys.exit(f"No step*_logits.pt files found in {directory!r}")
    steps = {}
    for f in files:
        step = int(os.path.basename(f).split("_")[0].removeprefix("step"))
        steps[step] = torch.load(f, weights_only=True)
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare logit dumps from two runs")
    parser.add_argument("ref_dir", help="Reference run (e.g. opt_level=0)")
    parser.add_argument("cmp_dir", help="Run to compare against the reference")
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of top tokens to show when divergence is detected (default: 5)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1.0,
        help="Absolute logit difference threshold for reporting (default: 1.0)",
    )
    args = parser.parse_args()

    ref_steps = load_steps(args.ref_dir)
    cmp_steps = load_steps(args.cmp_dir)

    common = sorted(set(ref_steps) & set(cmp_steps))
    if not common:
        sys.exit("No steps in common between the two directories.")

    print(
        f"Comparing {len(common)} step(s)  |  ref={args.ref_dir}  cmp={args.cmp_dir}\n"
        f"{'step':>6}  {'max|Δ|':>10}  {'mean|Δ|':>10}  {'pcc':>8}  {'top1 match':>10}"
    )
    first_diverge = None
    for step in common:
        ref = ref_steps[step].float()
        cmp = cmp_steps[step].float()
        if ref.shape != cmp.shape:
            print(
                f"{step:6d}  shape mismatch ref={tuple(ref.shape)} cmp={tuple(cmp.shape)}"
            )
            continue
        diff = (ref - cmp).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        step_pcc = pcc(ref, cmp)
        ref_top1 = ref[0].argmax().item()
        cmp_top1 = cmp[0].argmax().item()
        match = ref_top1 == cmp_top1
        print(
            f"{step:6d}  {max_diff:10.4f}  {mean_diff:10.6f}  {step_pcc:8.6f}  {'yes' if match else 'NO':>10}"
        )
        if not match and first_diverge is None:
            first_diverge = step

    if first_diverge is not None:
        step = first_diverge
        ref = ref_steps[step].float()
        cmp = cmp_steps[step].float()
        k = args.atol and args.topk
        ref_vals, ref_idxs = ref[0].topk(args.topk)
        cmp_vals, cmp_idxs = cmp[0].topk(args.topk)
        print(f"\nFirst divergence at step {step}:")
        print(f"  ref top-{args.topk}: tokens={ref_idxs.tolist()}  logits={[f'{v:.3f}' for v in ref_vals.tolist()]}")
        print(f"  cmp top-{args.topk}: tokens={cmp_idxs.tolist()}  logits={[f'{v:.3f}' for v in cmp_vals.tolist()]}")
    else:
        print("\nAll steps: top-1 token agrees between the two runs.")


if __name__ == "__main__":
    main()
