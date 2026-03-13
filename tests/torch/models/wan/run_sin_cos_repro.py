#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Minimal reproduction of SFPI compiler ICE triggered by sin/cos on Blackhole.

The Wan 2.1 T2V transformer fails because get_timestep_embedding() calls
torch.sin() and torch.cos(), which lower to the SFPU trigonometry kernel.
The SFPI compiler (GCC 15.1.0) hits an internal register allocation error:

    ckernel_sfpu_trigonometry.h:168:31: internal compiler error:
        in curr_insn_transform, at lra-constraints.cc:4355
    Failed to generate binaries for eltwise_sfpu

This script tests three progressively complex modules to isolate the trigger.

Usage:
    python tests/torch/models/wan/run_sin_cos_repro.py

With IR export:
    python tests/torch/models/wan/run_sin_cos_repro.py --export-ir

With codegen (emits TTNN Python code):
    python tests/torch/models/wan/run_sin_cos_repro.py --codegen
"""

import argparse
import math
import os
import sys
import time
import traceback

os.environ["XLA_HLO_DEBUG"] = "1"

import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

# ---------------------------------------------------------------------------
# Test modules — progressively more complex
# ---------------------------------------------------------------------------


class SinOnly(nn.Module):
    """Bare torch.sin — simplest possible trigger."""

    def forward(self, x):
        return torch.sin(x)


class CosOnly(nn.Module):
    """Bare torch.cos — simplest possible trigger."""

    def forward(self, x):
        return torch.cos(x)


class SinCosConcat(nn.Module):
    """sin + cos concatenated — matches get_timestep_embedding pattern."""

    def forward(self, x):
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class TimestepEmbedRepro(nn.Module):
    """Exact reproduction of diffusers get_timestep_embedding().

    Source: diffusers/models/embeddings.py:27-78
    """

    def __init__(self, embedding_dim=320):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        half_dim = self.embedding_dim // 2
        exponent = -math.log(10000) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / half_dim
        emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        # flip sin to cos (flip_sin_to_cos=True in Wan config)
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
        return emb


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

TESTS = [
    {
        "name": "sin_only_32x32",
        "module": SinOnly(),
        "input_fn": lambda dev: (torch.randn(32, 32, device=dev),),
        "desc": "torch.sin on (32, 32)",
    },
    {
        "name": "cos_only_32x32",
        "module": CosOnly(),
        "input_fn": lambda dev: (torch.randn(32, 32, device=dev),),
        "desc": "torch.cos on (32, 32)",
    },
    {
        "name": "sin_cos_concat_1x160",
        "module": SinCosConcat(),
        "input_fn": lambda dev: (torch.randn(1, 160, device=dev),),
        "desc": "cat([sin(x), cos(x)]) on (1, 160)",
    },
    {
        "name": "timestep_embed_repro",
        "module": TimestepEmbedRepro(embedding_dim=320),
        "input_fn": lambda dev: (torch.tensor([500], dtype=torch.long, device=dev),),
        "desc": "Full get_timestep_embedding (dim=320, timestep=500)",
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_test(name, module, inputs, device, export_path=None):
    """Compile and run a single module, returning (passed, output_or_error)."""
    if export_path:
        torch_xla.set_custom_compile_options(
            {
                "export_path": export_path,
                "export_model_name": name,
            }
        )

    module = module.to(device)
    compiled = torch.compile(module, backend="tt")

    with torch.no_grad():
        output = compiled(*inputs)
    torch_xla.sync(wait=True)

    result = output.cpu()
    return True, result


def main():
    parser = argparse.ArgumentParser(description="sin/cos SFPI repro")
    parser.add_argument("--export-ir", action="store_true", help="Export TTIR/TTNN IRs")
    parser.add_argument(
        "--codegen", action="store_true", help="Emit TTNN Python code via codegen_py"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("SIN/COS SFPI COMPILER BUG — Minimal Reproduction")
    print("=" * 70)

    # ---- Device setup ----
    xr.set_device_type("TT")
    device = torch_xla.device()
    print(f"[setup] TT device: {device}")
    print(f"[setup] XLA_HLO_DEBUG={os.environ.get('XLA_HLO_DEBUG', 'not set')}")

    export_base = "ir_export_sin_cos" if args.export_ir else None

    # ---- Run tests ----
    results = {}
    for test in TESTS:
        name = test["name"]
        print(f"\n{'─' * 70}")
        print(f"[test] {name}: {test['desc']}")
        print(f"{'─' * 70}")

        inputs = test["input_fn"](device)
        export_path = f"{export_base}/{name}" if export_base else None

        t0 = time.time()
        try:
            passed, result = run_test(name, test["module"], inputs, device, export_path)
            elapsed = time.time() - t0
            print(f"[result] PASS in {elapsed:.2f}s")
            print(f"[result] shape: {result.shape}, dtype: {result.dtype}")
            print(
                f"[result] mean={result.float().mean().item():.6f}, "
                f"std={result.float().std().item():.6f}"
            )
            print(
                f"[result] has NaN: {result.isnan().any().item()}, "
                f"has Inf: {result.isinf().any().item()}"
            )
            results[name] = "PASS"
        except Exception as e:
            elapsed = time.time() - t0
            err_msg = str(e)
            is_sfpi = (
                "eltwise_sfpu" in err_msg
                or "trigonometry" in err_msg
                or "SFPLOADI" in err_msg
            )
            print(f"[result] FAIL in {elapsed:.2f}s")
            if is_sfpi:
                print(f"[result] SFPI compiler ICE confirmed")
            print(f"[result] Error: {err_msg[:500]}")
            results[name] = (
                "FAIL (SFPI ICE)" if is_sfpi else f"FAIL ({type(e).__name__})"
            )

    # ---- Codegen attempt ----
    if args.codegen:
        print(f"\n{'─' * 70}")
        print("[codegen] Attempting TTNN Python code generation ...")
        print(f"{'─' * 70}")
        try:
            from tt_torch import codegen_py

            # Use the simplest failing case, or SinCosConcat as default
            model = SinCosConcat()
            x = torch.randn(1, 160)
            codegen_py(model, x, export_path="codegen_sin_cos", compiler_options={})
            print("[codegen] TTNN code written to codegen_sin_cos/")
            results["codegen"] = "PASS"
        except Exception as e:
            print(f"[codegen] Failed: {e}")
            results["codegen"] = f"FAIL ({type(e).__name__})"

    # ---- Summary ----
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for name, status in results.items():
        marker = "PASS" if "PASS" in status else "FAIL"
        print(f"  [{marker}] {name}: {status}")

    if export_base:
        print(f"\n  IR artifacts: {export_base}/")
    if args.codegen:
        print(f"  Codegen artifacts: codegen_sin_cos/")

    any_fail = any("FAIL" in s for s in results.values())
    print(f"\n{'=' * 70}")
    if any_fail:
        print("At least one test FAILED — SFPI bug reproduced")
    else:
        print("All tests PASSED — SFPI bug NOT reproduced with these modules")
    print(f"{'=' * 70}")

    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
