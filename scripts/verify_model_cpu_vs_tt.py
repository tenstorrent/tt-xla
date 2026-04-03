#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CPU vs TT-device output verification script for new model bringup.

Runs each model's load_model() + load_inputs() on both CPU and TT device,
then computes PCC (Pearson Correlation Coefficient) and max absolute error
to confirm numerical correctness.

Usage:
    # All new models
    python scripts/verify_model_cpu_vs_tt.py

    # Specific model(s)
    python scripts/verify_model_cpu_vs_tt.py --models isaac_sim nerf hivt

    # Verbose (print tensor stats per output)
    python scripts/verify_model_cpu_vs_tt.py --verbose

PCC >= 0.99  -> PASS
PCC < 0.99   -> FAIL
"""

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch_xla.core.xla_model as xm

# -- add tt_forge_models to path -----------------------------------------------
sys.path.insert(0, ".")
sys.path.insert(0, "third_party")


# -- PCC & comparison helpers --------------------------------------------------

def compute_pcc(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson Correlation Coefficient between two tensors (flattened)."""
    x = x.float().flatten()
    y = y.float().flatten()
    if torch.allclose(x, y, rtol=1e-3, atol=1e-3):
        return 1.0
    if x.numel() <= 1:
        return 0.0
    vx = x - x.mean()
    vy = y - y.mean()
    denom = vx.norm() * vy.norm()
    if denom == 0:
        return float("nan")
    return float((vx @ vy) / denom)


def _cast_to_bf16(inputs: Any) -> Any:
    """Cast all float tensors in a nested structure to bfloat16."""
    if isinstance(inputs, torch.Tensor):
        return inputs.to(torch.bfloat16) if inputs.is_floating_point() else inputs
    if isinstance(inputs, dict):
        return {k: _cast_to_bf16(v) for k, v in inputs.items()}
    if isinstance(inputs, (list, tuple)):
        return type(inputs)(_cast_to_bf16(v) for v in inputs)
    return inputs


def flatten_outputs(outputs: Any) -> List[torch.Tensor]:
    """Recursively extract all tensors from a nested dict/list/tuple/tensor."""
    if isinstance(outputs, torch.Tensor):
        return [outputs]
    if isinstance(outputs, dict):
        result = []
        for v in outputs.values():
            result.extend(flatten_outputs(v))
        return result
    if isinstance(outputs, (list, tuple)):
        result = []
        for v in outputs:
            result.extend(flatten_outputs(v))
        return result
    return []


def to_cpu(outputs: Any) -> Any:
    """Move all tensors in a nested structure to CPU."""
    if isinstance(outputs, torch.Tensor):
        return outputs.detach().cpu()
    if isinstance(outputs, dict):
        return {k: to_cpu(v) for k, v in outputs.items()}
    if isinstance(outputs, (list, tuple)):
        return type(outputs)(to_cpu(v) for v in outputs)
    return outputs


def compare_outputs(
    cpu_out: Any,
    tt_out: Any,
    pcc_threshold: float = 0.99,
    verbose: bool = False,
) -> Tuple[bool, float, float]:
    """
    Compare CPU and TT outputs tensor-by-tensor.

    Returns:
        (passed, min_pcc, max_abs_err)
    """
    cpu_tensors = flatten_outputs(cpu_out)
    tt_tensors  = flatten_outputs(tt_out)

    if len(cpu_tensors) != len(tt_tensors):
        print(f"  [!] Output count mismatch: cpu={len(cpu_tensors)} tt={len(tt_tensors)}")
        return False, 0.0, float("inf")

    all_pcc = []
    all_abs_err = []

    for i, (c, t) in enumerate(zip(cpu_tensors, tt_tensors)):
        c = c.float()
        t = t.float()

        if c.shape != t.shape:
            print(f"  [!] Shape mismatch at output[{i}]: cpu={c.shape} tt={t.shape}")
            return False, 0.0, float("inf")

        pcc = compute_pcc(c, t)
        abs_err = (c - t).abs().max().item()
        rel_err = abs_err / (c.abs().max().item() + 1e-8)

        all_pcc.append(pcc)
        all_abs_err.append(abs_err)

        if verbose:
            print(
                f"  output[{i}] shape={str(list(c.shape)):30s}  "
                f"PCC={pcc:.6f}  max_abs_err={abs_err:.4e}  max_rel_err={rel_err:.4e}"
            )

    min_pcc = min(all_pcc) if all_pcc else float("nan")
    max_abs  = max(all_abs_err) if all_abs_err else float("inf")
    passed   = min_pcc >= pcc_threshold

    return passed, min_pcc, max_abs


# -- model registry ------------------------------------------------------------

@dataclass
class ModelSpec:
    key: str
    module_path: str
    variant: Optional[str] = None   # None -> default variant


MODEL_REGISTRY: Dict[str, List[ModelSpec]] = {
    "yolov8": [
        ModelSpec("yolov8/YOLOv8s_640",
                  "tt_forge_models.yolov8.pytorch.loader",
                  "YOLOv8s_640"),
        ModelSpec("yolov8/YOLOv8l_1280",
                  "tt_forge_models.yolov8.pytorch.loader",
                  "YOLOv8l_1280"),
    ],
    "isaac_sim": [
        ModelSpec("isaac_sim/AnymalC_Flat",
                  "tt_forge_models.isaac_sim.pytorch.loader",
                  "AnymalC_Flat"),
        ModelSpec("isaac_sim/AnymalC_Rough",
                  "tt_forge_models.isaac_sim.pytorch.loader",
                  "AnymalC_Rough"),
        ModelSpec("isaac_sim/H1_Velocity",
                  "tt_forge_models.isaac_sim.pytorch.loader",
                  "H1_Velocity"),
    ],
    "nerf": [
        ModelSpec("nerf/NeRF_Vanilla",
                  "tt_forge_models.nerf.pytorch.loader",
                  "NeRF_Vanilla"),
        ModelSpec("nerf/NeRF_Coarse",
                  "tt_forge_models.nerf.pytorch.loader",
                  "NeRF_Coarse"),
    ],
    "hivt": [
        ModelSpec("hivt/HiVT_64",
                  "tt_forge_models.hivt.pytorch.loader",
                  "HiVT_64"),
        ModelSpec("hivt/HiVT_128",
                  "tt_forge_models.hivt.pytorch.loader",
                  "HiVT_128"),
    ],
    "unet_3d": [
        ModelSpec("unet_3d/UNet3D_Small",
                  "tt_forge_models.unet_3d.pytorch.loader",
                  "UNet3D_Small"),
        ModelSpec("unet_3d/UNet3D_Large",
                  "tt_forge_models.unet_3d.pytorch.loader",
                  "UNet3D_Large"),
    ],
    "bevfusion": [
        ModelSpec("bevfusion/BEVFusion_Camera_Lidar",
                  "tt_forge_models.bevfusion.pytorch.loader",
                  "BEVFusion_Camera_Lidar"),
        ModelSpec("bevfusion/BEVFusion_Camera_Only",
                  "tt_forge_models.bevfusion.pytorch.loader",
                  "BEVFusion_Camera_Only"),
    ],
    "centerpoint": [
        ModelSpec("centerpoint/CenterPoint_Pillar",
                  "tt_forge_models.centerpoint.pytorch.loader",
                  "CenterPoint_Pillar"),
    ],
    "centerpoint_original": [
        ModelSpec("centerpoint_original/CenterPoint_Original",
                  "tt_forge_models.centerpoint_original.pytorch.loader",
                  "CenterPoint_Original"),
    ],
    "llava_next": [
        ModelSpec("llava_next/LLaVA_NeXT_Vision_Encoder",
                  "tt_forge_models.llava_next.pytorch.loader",
                  "LLaVA_NeXT_Vision_Encoder"),
    ],
}


# -- inference helpers ---------------------------------------------------------

def load_loader(spec: ModelSpec):
    """Dynamically import and instantiate the ModelLoader for a spec."""
    import importlib
    mod = importlib.import_module(spec.module_path)
    ModelLoader = mod.ModelLoader
    ModelVariant = mod.ModelVariant

    if spec.variant is not None:
        # Find the variant enum member whose value matches the spec string
        variant = next(
            (v for v in ModelVariant if v.value == spec.variant), None
        )
        if variant is None:
            raise ValueError(
                f"Variant '{spec.variant}' not found in {ModelVariant}. "
                f"Available: {[v.value for v in ModelVariant]}"
            )
        return ModelLoader(variant)
    return ModelLoader()


def inputs_to_device(inputs: Any, device) -> Any:
    """Move all tensors in inputs to the given device."""
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {k: inputs_to_device(v, device) for k, v in inputs.items()}
    if isinstance(inputs, (list, tuple)):
        return type(inputs)(inputs_to_device(v, device) for v in inputs)
    return inputs


def run_inference(model: torch.nn.Module, inputs: Any) -> Any:
    """Run model forward pass. Handles tensor, tuple, list, and dict inputs."""
    with torch.no_grad():
        if isinstance(inputs, torch.Tensor):
            return model(inputs)
        if isinstance(inputs, dict):
            return model(**inputs)
        if isinstance(inputs, (list, tuple)):
            return model(*inputs)
        raise TypeError(f"Unsupported inputs type: {type(inputs)}")


def run_tt_inference(model: torch.nn.Module, inputs: Any, tt_device) -> Any:
    """Compile model with torch.compile('tt') and run on TT device."""
    # Move model to TT device
    model_tt = model.to(tt_device)

    # Move inputs to TT device (cast to bfloat16 to match model weights on TT)
    inputs_tt = _cast_to_bf16(inputs)
    inputs_tt = inputs_to_device(inputs_tt, tt_device)

    # Compile with TT backend
    compiled = torch.compile(model_tt, backend="tt", fullgraph=True)

    # Run
    with torch.no_grad():
        if isinstance(inputs_tt, torch.Tensor):
            output = compiled(inputs_tt)
        elif isinstance(inputs_tt, dict):
            output = compiled(**inputs_tt)
        elif isinstance(inputs_tt, (list, tuple)):
            output = compiled(*inputs_tt)
        else:
            raise TypeError(f"Unsupported inputs type: {type(inputs_tt)}")

    # Sync TT device and bring back to CPU
    xm.mark_step()
    return to_cpu(output)


# -- main verification logic ---------------------------------------------------

def verify_spec(
    spec: ModelSpec,
    tt_device,
    pcc_threshold: float = 0.99,
    verbose: bool = False,
) -> bool:
    """Load, run CPU + TT inference, compare, print results. Returns True if passed."""
    print(f"\n{'='*70}")
    print(f"  Model : {spec.key}")
    print(f"{'='*70}")

    # -- Load model & inputs ---------------------------------------------------
    try:
        loader = load_loader(spec)
        model = loader.load_model()
        inputs = loader.load_inputs()
        model.eval()
    except Exception as e:
        print(f"  [LOAD ERROR] {e}")
        return False

    # -- CPU (golden) inference ------------------------------------------------
    # Cast model+inputs to bfloat16 to match TT device dtype for a fair comparison.
    # TT device always runs in bfloat16; comparing float32 vs bfloat16 inflates errors.
    model_bf16 = model.to(torch.bfloat16)
    inputs_bf16 = inputs_to_device(inputs, "cpu")
    inputs_bf16 = _cast_to_bf16(inputs_bf16)
    print("  Running CPU (golden) inference ...", end=" ", flush=True)
    try:
        t0 = time.perf_counter()
        cpu_output = run_inference(model_bf16, inputs_bf16)
        cpu_output = to_cpu(cpu_output)
        cpu_time = time.perf_counter() - t0
        print(f"done ({cpu_time*1000:.1f} ms)")
    except Exception as e:
        print(f"\n  [CPU ERROR] {e}")
        return False

    # -- TT-device inference ---------------------------------------------------
    print("  Running TT-device inference ...", end=" ", flush=True)
    try:
        t0 = time.perf_counter()
        tt_output = run_tt_inference(model, inputs, tt_device)
        tt_time = time.perf_counter() - t0
        print(f"done ({tt_time*1000:.1f} ms)")
    except Exception as e:
        print(f"\n  [TT ERROR] {e}")
        return False

    # -- Output comparison -----------------------------------------------------
    print("  Comparing outputs:")
    passed, min_pcc, max_abs_err = compare_outputs(
        cpu_output, tt_output, pcc_threshold=pcc_threshold, verbose=verbose
    )

    status = "PASS ✓" if passed else "FAIL ✗"
    print(f"  Result : {status}")
    print(f"  PCC    : {min_pcc:.6f}  (threshold >= {pcc_threshold})")
    print(f"  MaxErr : {max_abs_err:.4e}")
    print(f"  CPU time: {cpu_time*1000:.1f} ms   TT time: {tt_time*1000:.1f} ms")

    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Verify TT-device model outputs against CPU reference"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        default=["all"],
        help="Which model groups to verify (default: all)",
    )
    parser.add_argument(
        "--pcc-threshold",
        type=float,
        default=0.99,
        help="Minimum PCC to consider output correct (default: 0.99)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-tensor PCC and error statistics",
    )
    args = parser.parse_args()

    # Get TT device
    print("Acquiring TT device...")
    tt_device = xm.xla_device()
    print(f"TT device: {tt_device}")

    # Gather specs to run
    if "all" in args.models:
        groups = list(MODEL_REGISTRY.keys())
    else:
        groups = args.models

    specs = []
    for g in groups:
        specs.extend(MODEL_REGISTRY[g])

    # Run verification
    results: Dict[str, bool] = {}
    for spec in specs:
        passed = verify_spec(
            spec,
            tt_device=tt_device,
            pcc_threshold=args.pcc_threshold,
            verbose=args.verbose,
        )
        results[spec.key] = passed

    # -- Summary ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    pass_count = sum(results.values())
    total = len(results)
    for key, ok in results.items():
        mark = "PASS ✓" if ok else "FAIL ✗"
        print(f"  {mark}  {key}")
    print(f"\n  {pass_count}/{total} models passed (PCC >= {args.pcc_threshold})")
    print(f"{'='*70}")

    sys.exit(0 if pass_count == total else 1)


if __name__ == "__main__":
    main()
