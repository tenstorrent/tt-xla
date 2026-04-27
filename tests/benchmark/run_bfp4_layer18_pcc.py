# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Execute generated main.py and main_cpu_bypass.py and PCC-compare outputs.

Expects the directory produced by debug_bfp4_layer18.py, containing:
    main.py
    main_cpu_bypass.py          (from codegen_accuracy.run_alchemist_cpu_bypass)
    consteval.py                (optional — may be inlined into main.py)
    tensors/arg*.tensorbin      (serialized weights)
    utils.py                    (DeviceGetter etc.)

Each module is executed in its own subprocess (they share global state such as
_cached__main, and we do not want on-device residual state leaking between
runs). Each subprocess serializes its _main() outputs to outputs.pt, and we
load both and compute PCC.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import torch

# --------------------------------------------------------------------------
# PCC (Pearson Correlation Coefficient). Same definition used in tests/benchmark/utils.py.
# --------------------------------------------------------------------------


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float32).flatten()
    b = b.to(torch.float32).flatten()
    if a.numel() != b.numel():
        raise ValueError(f"shape mismatch: {a.numel()} vs {b.numel()}")
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom == 0.0:
        return float("nan")
    return ((a * b).sum() / denom).item()


# --------------------------------------------------------------------------
# Subprocess driver: runs a given module and writes list-of-torch-tensors to a file.
# --------------------------------------------------------------------------

_DRIVER_SNIPPET = '''
import os, sys, torch, ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath("{module_path}")))

import importlib.util
spec = importlib.util.spec_from_file_location("_target_module", "{module_path}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

inputs = mod.load_inputs_for__main()
result = mod._main(inputs)

# For multi-device sharded tensors, ttnn.to_torch needs a mesh_composer.
# We do not know the exact shard spec of each output, so instead pull every
# per-chip shard as a separate torch tensor via ttnn.get_device_tensors and
# stack them on a new leading axis. PCC between two runs that gather this
# way is invariant to the logical sharding: as long as both runs use the
# same gather, the element-wise correspondence is preserved.
def _gather_as_stacked_torch(t):
    host = ttnn.from_device(t)
    shards = ttnn.get_device_tensors(host)
    pieces = []
    for s in shards:
        th = ttnn.to_torch(s)
        pieces.append(th.to(torch.bfloat16).detach().contiguous().cpu())
    return torch.stack(pieces, dim=0)  # [num_devices, *shard_shape]

out = [_gather_as_stacked_torch(t) for t in result]
torch.save(out, "{output_path}")
print("WROTE", "{output_path}", flush=True)
'''


def _run_module(module_path: Path, output_path: Path) -> None:
    """Execute `module_path` (main.py or main_cpu_bypass.py) in a subprocess."""
    driver = _DRIVER_SNIPPET.format(module_path=str(module_path), output_path=str(output_path))
    work_dir = module_path.parent

    print(f"\n=== Running {module_path.name} ===")
    # cwd must be the export dir so that `./tensors/*.tensorbin` paths resolve.
    env = os.environ.copy()
    # Ensure the utils.py next to main.py is importable.
    env["PYTHONPATH"] = f"{work_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"

    result = subprocess.run(
        [sys.executable, "-c", driver],
        cwd=str(work_dir),
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Subprocess failed with code {result.returncode}")
    if not output_path.exists():
        raise RuntimeError(f"Output {output_path} was not produced")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <generated_dir>", file=sys.stderr)
        sys.exit(2)

    gen_dir = Path(sys.argv[1]).resolve()
    main_py = gen_dir / "main.py"
    bypass_py = gen_dir / "main_cpu_bypass.py"
    if not main_py.exists():
        sys.exit(f"Missing: {main_py}")
    if not bypass_py.exists():
        sys.exit(f"Missing: {bypass_py} (run debug_bfp4_layer18.py first)")

    device_out = gen_dir / "device_outputs.pt"
    bypass_out = gen_dir / "bypass_outputs.pt"

    _run_module(main_py, device_out)
    _run_module(bypass_py, bypass_out)

    device_tensors = torch.load(device_out)
    bypass_tensors = torch.load(bypass_out)

    if len(device_tensors) != len(bypass_tensors):
        sys.exit(
            f"Output count mismatch: device={len(device_tensors)}, "
            f"bypass={len(bypass_tensors)}"
        )

    print("\n=== Per-output PCC (device vs CPU bypass) ===")
    pccs = []
    for i, (d, b) in enumerate(zip(device_tensors, bypass_tensors)):
        if d.shape != b.shape:
            print(f"  [{i}] shape mismatch: device={tuple(d.shape)}, bypass={tuple(b.shape)} -- SKIPPING")
            continue
        pcc = compute_pcc(d, b)
        pccs.append(pcc)
        print(f"  [{i}] shape={tuple(d.shape)} PCC={pcc:.6f}")

    if pccs:
        overall = sum(pccs) / len(pccs)
        worst = min(pccs)
        print(f"\nOverall mean PCC: {overall:.6f}")
        print(f"Worst PCC:        {worst:.6f}")
        print(textwrap.dedent(f"""
            Interpretation (layer 18, end-to-end):
                PCC >= 0.995  kernel matches CPU;  quantization loss dominates.
                0.99-0.995    kernel contributes moderate error.
                PCC <  0.99   kernel is a significant contributor.
        """).strip())


if __name__ == "__main__":
    main()
