# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Verify a hand-edited codegen_py ``main.py`` against the CPU decode golden.

Companion driver for the ``--codegen-py-export-path`` pytest flag in
``tests/benchmark/test_llms.py``. The benchmark, when run with that flag and
``--decode-only``, produces:

    <codegen_dir>/
        main.py             # generated TTNN Python for the decode graph
        irs/                # vhlo / shlo / ttir / ttnn .mlir
        tensors/            # serialized inputs + parameters used by main.py
        run                 # shell script that executes main.py
        golden_logits.pt    # CPU decode logits (PCC reference)

This script executes the (possibly edited) ``main.py`` and compares its output
against ``golden_logits.pt`` using ``compute_pcc``. Typical iteration loop::

    # 1) one-shot: emit codegen + save golden
    pytest -svv tests/benchmark/test_llms.py::test_kimi_k2_tp_galaxy_2_layers \\
        --decode-only --max-output-tokens 1 \\
        --codegen-py-export-path kimi_codegen

    # 2) edit kimi_codegen/main.py
    # 3) python tests/benchmark/scripts/verify_emitpy.py kimi_codegen
    # PCC = 0.99987   max |Δ| = 1.2e-3

Notes on tt-alchemist ``main.py`` layout
----------------------------------------
The generated ``main.py`` is emitted by tt-alchemist; its exact entrypoint
signature and ``tensors/`` layout can evolve. The PJRT path that normally
executes it loads the module as ``main`` and calls ``main_for_test`` (see
``pjrt_implementation/src/api/so_loaded_executable_instance.cc``). The
``run`` script next to ``main.py`` is the source of truth for how to open the
device and feed inputs from ``tensors/``.

If the generated ``main.py`` already exposes top-level helpers like
``open_device()`` / ``load_inputs(tensors_dir)``, this driver will use them.
Otherwise it falls back to invoking ``main_for_test`` with the device opened
via ``ttnn.open_mesh_device`` (or ``ttnn.open_device`` for a single chip) and
asks the user to align ``load_inputs`` with the generated ``run`` script.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch


def _import_main_module(codegen_dir: Path):
    """Load ``<codegen_dir>/main.py`` as an importable module."""
    main_py = codegen_dir / "main.py"
    if not main_py.is_file():
        raise FileNotFoundError(f"No main.py found at {main_py}")
    sys.path.insert(0, str(codegen_dir.resolve()))
    spec = importlib.util.spec_from_file_location("emitpy_main", main_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _open_device(module, mesh_shape):
    """Open the device the generated code expects.

    Prefers a generated ``open_device()`` helper if one exists; otherwise
    opens a mesh device of the requested shape via ttnn.
    """
    if hasattr(module, "open_device"):
        return module.open_device()

    import ttnn

    if mesh_shape:
        rows, cols = mesh_shape
        return ttnn.open_mesh_device(ttnn.MeshShape(rows, cols))
    return ttnn.open_device(device_id=0)


def _close_device(module, device):
    if hasattr(module, "close_device"):
        module.close_device(device)
        return

    import ttnn

    if hasattr(device, "get_devices"):
        ttnn.close_mesh_device(device)
    else:
        ttnn.close_device(device)


def _load_inputs(module, tensors_dir: Path, device):
    """Materialize inputs for ``main_for_test`` from the tensors directory.

    Defers to a generated ``load_inputs(tensors_dir, device)`` helper if
    present. Otherwise raises with guidance to mirror the generated ``run``
    script's loading sequence.
    """
    if hasattr(module, "load_inputs"):
        return module.load_inputs(str(tensors_dir), device)

    raise RuntimeError(
        f"Generated main.py does not expose a load_inputs(tensors_dir, device) "
        f"helper. Inspect {tensors_dir.parent / 'run'} for the canonical "
        f"sequence (input file naming, ttnn.load_tensor calls, layout / dtype "
        f"conversions) and either: (a) add a load_inputs helper to main.py, "
        f"or (b) replace this function in verify_emitpy.py with the inlined "
        f"equivalent."
    )


def _to_torch(out) -> torch.Tensor:
    """Reduce ttnn output to a torch.Tensor, gathering shards if needed."""
    import ttnn

    if isinstance(out, (list, tuple)):
        out = out[0]
    if hasattr(out, "get_devices") or hasattr(out, "devices"):
        # Multichip: aggregate replicated / sharded mesh tensor to a single
        # torch tensor. Generated code may already all_gather lm_head outputs;
        # in that case aggregate_as_tensor returns the unsharded result.
        return ttnn.to_torch(out, mesh_composer=None)
    return ttnn.to_torch(out)


def _compute_pcc(golden: torch.Tensor, candidate: torch.Tensor) -> float:
    """Use the project's PCC implementation to keep numbers comparable."""
    repo_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo_root / "tests" / "benchmark"))
    from utils import compute_pcc  # noqa: E402

    return compute_pcc(golden, candidate)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run codegen_py main.py and PCC against the CPU golden."
    )
    parser.add_argument(
        "codegen_dir",
        type=Path,
        help="Directory passed to --codegen-py-export-path (contains main.py, "
        "tensors/, golden_logits.pt).",
    )
    parser.add_argument(
        "--mesh-shape",
        type=str,
        default=None,
        help="Mesh shape as ROWSxCOLS (e.g. 4x8 for galaxy). Defaults to no "
        "mesh / single device. Ignored if main.py provides open_device().",
    )
    parser.add_argument(
        "--save-output",
        type=Path,
        default=None,
        help="Optional path to torch.save the device output for later "
        "comparison or debugging.",
    )
    parser.add_argument(
        "--pcc-threshold",
        type=float,
        default=None,
        help="If set, exit non-zero when PCC < threshold. Use as a gate "
        "from an iteration loop (e.g. autoresearch). Without this flag the "
        "script always exits 0 — preserving the original standalone behavior.",
    )
    args = parser.parse_args()

    codegen_dir: Path = args.codegen_dir.resolve()
    golden_path = codegen_dir / "golden_logits.pt"
    tensors_dir = codegen_dir / "tensors"

    if not golden_path.is_file():
        print(
            f"error: {golden_path} not found. Re-run the benchmark with "
            f"--codegen-py-export-path {codegen_dir.name} --decode-only.",
            file=sys.stderr,
        )
        return 2
    if not tensors_dir.is_dir():
        print(f"error: {tensors_dir} not found.", file=sys.stderr)
        return 2

    mesh_shape = None
    if args.mesh_shape:
        try:
            rows, cols = (int(x) for x in args.mesh_shape.lower().split("x"))
            mesh_shape = (rows, cols)
        except ValueError:
            print(
                f"error: --mesh-shape must be ROWSxCOLS (e.g. 4x8); got "
                f"{args.mesh_shape!r}.",
                file=sys.stderr,
            )
            return 2

    module = _import_main_module(codegen_dir)

    device = _open_device(module, mesh_shape)
    try:
        inputs = _load_inputs(module, tensors_dir, device)

        if not hasattr(module, "main_for_test"):
            raise RuntimeError(
                f"{codegen_dir / 'main.py'} does not define main_for_test. "
                f"The PJRT path expects main_for_test as the entrypoint "
                f"(see so_loaded_executable_instance.cc)."
            )
        raw_out = module.main_for_test(inputs, device)
        candidate = _to_torch(raw_out)
    finally:
        _close_device(module, device)

    golden = torch.load(golden_path, map_location="cpu")
    candidate_cpu = candidate.detach().cpu()

    if golden.shape != candidate_cpu.shape:
        print(
            f"warning: shape mismatch — golden {tuple(golden.shape)} vs "
            f"candidate {tuple(candidate_cpu.shape)}; PCC will compare "
            f"flattened tensors.",
            file=sys.stderr,
        )

    pcc = _compute_pcc(golden, candidate_cpu)
    max_diff = (golden.float() - candidate_cpu.float()).abs().max().item()
    print(f"PCC      = {pcc:.6f}")
    print(f"max |Δ| = {max_diff:.3e}")

    if args.save_output is not None:
        os.makedirs(args.save_output.parent, exist_ok=True)
        torch.save(candidate_cpu, args.save_output)
        print(f"saved candidate output to {args.save_output}")

    if args.pcc_threshold is not None and pcc < args.pcc_threshold:
        print(
            f"FAIL: PCC {pcc:.6f} < threshold {args.pcc_threshold}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
