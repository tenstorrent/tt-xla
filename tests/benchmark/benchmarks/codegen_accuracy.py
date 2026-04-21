# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Accuracy testing using codegen_py generated Python modules.

Generates ttnn Python code from a compiled model, optionally applies CPU bypass
to bfp4 matmul operations, and compares PCC between device and CPU-bypassed runs
to isolate whether accuracy loss comes from quantization or the TT matmul kernel.
"""

import importlib
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Pattern detection for bfp4 matmul groups in generated code
# ---------------------------------------------------------------------------


@dataclass
class BFP4MatmulGroup:
    """Represents a typecast-to-bfp4 -> matmul -> typecast-back triplet."""

    # Line numbers (0-based) in the source
    typecast_to_bfp4_line: int
    matmul_line: int
    typecast_back_line: Optional[int]

    # Variable names
    typecast_to_bfp4_output: str  # e.g. "var_3"
    typecast_to_bfp4_input: str  # e.g. "var_2" (the weight before quantization)
    matmul_output: str  # e.g. "var_4"
    matmul_activation: str  # e.g. "var_1" (the activation input)
    typecast_back_output: Optional[str]  # e.g. "var_5"

    # Full source lines (for replacement)
    typecast_to_bfp4_src: str
    matmul_src: str
    typecast_back_src: Optional[str]


def find_bfp4_matmul_groups(source_code: str) -> list[BFP4MatmulGroup]:
    """Parse generated main.py to find bfp4 matmul groups.

    A bfp4 matmul group is:
    1. var_A = ttnn.typecast(input, ttnn.DataType.BFLOAT4_B, ...)
    2. var_B = ttnn.matmul(activation, var_A, ...)
    3. var_C = ttnn.typecast(var_B, ttnn.DataType.BFLOAT16, ...)  [optional]
    """
    lines = source_code.splitlines()

    # Pattern for typecast to BFLOAT4_B
    # Matches: var_X = ttnn.typecast(input_var, ttnn.DataType.BFLOAT4_B, ...)
    typecast_bfp4_pattern = re.compile(
        r"^\s*([\w]+)\s*=\s*ttnn\.typecast\(\s*([\w\[\]\.]+)\s*,"
        r"\s*ttnn\.DataType\.BFLOAT4_B"
    )

    # Pattern for matmul
    # Matches: var_X = ttnn.matmul(activation, weight, ...)
    matmul_pattern = re.compile(
        r"^\s*([\w]+)\s*=\s*ttnn\.matmul\(\s*([\w\[\]\.]+)\s*," r"\s*([\w\[\]\.]+)"
    )

    # Pattern for typecast to BFLOAT16
    # Matches: var_X = ttnn.typecast(input_var, ttnn.DataType.BFLOAT16, ...)
    typecast_bf16_pattern = re.compile(
        r"^\s*([\w]+)\s*=\s*ttnn\.typecast\(\s*([\w\[\]\.]+)\s*,"
        r"\s*ttnn\.DataType\.BFLOAT16"
    )

    # Step 1: Find all typecast-to-bfp4 operations
    bfp4_typecasts = {}  # output_var -> (line_idx, input_var, src_line)
    for i, line in enumerate(lines):
        m = typecast_bfp4_pattern.match(line)
        if m:
            out_var, in_var = m.group(1), m.group(2)
            bfp4_typecasts[out_var] = (i, in_var, line)

    # Step 2: Find matmuls that consume a bfp4-typecast output
    groups = []
    for i, line in enumerate(lines):
        m = matmul_pattern.match(line)
        if not m:
            continue

        matmul_out = m.group(1)
        matmul_arg1 = m.group(2)
        matmul_arg2 = m.group(3)

        # Check if either argument is a bfp4-typecast output
        bfp4_weight_var = None
        activation_var = None
        if matmul_arg2 in bfp4_typecasts:
            bfp4_weight_var = matmul_arg2
            activation_var = matmul_arg1
        elif matmul_arg1 in bfp4_typecasts:
            bfp4_weight_var = matmul_arg1
            activation_var = matmul_arg2

        if bfp4_weight_var is None:
            continue

        tc_line_idx, tc_input_var, tc_src = bfp4_typecasts[bfp4_weight_var]

        # Step 3: Find the typecast-back (bfloat16) of the matmul output
        typecast_back_line = None
        typecast_back_output = None
        typecast_back_src = None
        for j in range(i + 1, min(i + 10, len(lines))):
            mb = typecast_bf16_pattern.match(lines[j])
            if mb and mb.group(2) == matmul_out:
                typecast_back_line = j
                typecast_back_output = mb.group(1)
                typecast_back_src = lines[j]
                break

        groups.append(
            BFP4MatmulGroup(
                typecast_to_bfp4_line=tc_line_idx,
                matmul_line=i,
                typecast_back_line=typecast_back_line,
                typecast_to_bfp4_output=bfp4_weight_var,
                typecast_to_bfp4_input=tc_input_var,
                matmul_output=matmul_out,
                matmul_activation=activation_var,
                typecast_back_output=typecast_back_output,
                typecast_to_bfp4_src=tc_src,
                matmul_src=line,
                typecast_back_src=typecast_back_src,
            )
        )

    return groups


# ---------------------------------------------------------------------------
# CPU bypass code generation
# ---------------------------------------------------------------------------


def _make_bypass_code(group: BFP4MatmulGroup, bypass_idx: int) -> dict[int, str]:
    """Generate CPU bypass replacement lines for a single bfp4 matmul group.

    Returns a dict mapping line_number -> replacement_code.
    The typecast-to-bfp4 line is kept (preserves quantization loss).
    The matmul line is replaced with CPU bypass.
    The typecast-back line is replaced with a no-op assignment.
    """
    indent = "    "  # Generated code uses 4-space indent inside functions
    act = group.matmul_activation
    wt = group.typecast_to_bfp4_output
    out = group.matmul_output

    bypass = f"""{indent}# === CPU BYPASS #{bypass_idx} (was ttnn.matmul) ===
{indent}_host_act_{bypass_idx} = ttnn.from_device(ttnn.to_layout({act}, ttnn.ROW_MAJOR_LAYOUT))
{indent}_host_wt_{bypass_idx} = ttnn.from_device(ttnn.to_layout(ttnn.typecast({wt}, ttnn.DataType.BFLOAT16), ttnn.ROW_MAJOR_LAYOUT))
{indent}_torch_act_{bypass_idx} = ttnn.to_torch(_host_act_{bypass_idx}).to(torch.bfloat16)
{indent}_torch_wt_{bypass_idx} = ttnn.to_torch(_host_wt_{bypass_idx}).to(torch.bfloat16)
{indent}_cpu_result_{bypass_idx} = torch.matmul(_torch_act_{bypass_idx}, _torch_wt_{bypass_idx})
{indent}_device_{bypass_idx} = {act}.device()
{indent}{out} = ttnn.from_torch(_cpu_result_{bypass_idx}, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, device=_device_{bypass_idx}, memory_config=ttnn.DRAM_MEMORY_CONFIG)
{indent}# === END CPU BYPASS #{bypass_idx} ==="""

    replacements = {}
    # Keep the typecast-to-bfp4 line unchanged (preserves quantization)
    # Replace the matmul line with the CPU bypass
    replacements[group.matmul_line] = bypass
    # Replace the typecast-back with a no-op if it exists
    if group.typecast_back_line is not None:
        replacements[group.typecast_back_line] = (
            f"{indent}{group.typecast_back_output} = {out}"
            f"  # typecast-back skipped (already bfloat16 from CPU bypass)"
        )

    return replacements


def generate_cpu_bypass_code(source_code: str, groups: list[BFP4MatmulGroup]) -> str:
    """Generate a modified version of main.py with CPU-bypassed bfp4 matmuls.

    The bfp4 typecast (quantization) is preserved on device.
    Only the matmul itself is moved to CPU.
    """
    lines = source_code.splitlines()

    # Collect all line replacements
    all_replacements: dict[int, str] = {}
    for idx, group in enumerate(groups):
        replacements = _make_bypass_code(group, idx)
        all_replacements.update(replacements)

    # Apply replacements
    new_lines = []
    for i, line in enumerate(lines):
        if i in all_replacements:
            new_lines.append(all_replacements[i])
        else:
            new_lines.append(line)

    # Add torch import if not present
    result = "\n".join(new_lines)
    if "import torch" not in result:
        # Insert after the ttnn import
        result = result.replace("import ttnn", "import ttnn\nimport torch", 1)

    return result


# ---------------------------------------------------------------------------
# Module loading and execution
# ---------------------------------------------------------------------------


def _load_module(module_path: str, module_name: str):
    """Dynamically import a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    # Add the module's directory to sys.path so its imports (utils, etc.) resolve
    module_dir = os.path.dirname(module_path)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec.loader.exec_module(module)
    return module


def _compute_pcc(golden, test) -> float:
    """Compute Pearson Correlation Coefficient between two tensors."""
    import torch

    g = golden.to(torch.float32).flatten()
    t = test.to(torch.float32).flatten()
    g_centered = g - g.mean()
    t_centered = t - t.mean()
    denom = g_centered.norm() * t_centered.norm()
    if denom == 0:
        if torch.allclose(g, t, rtol=1e-2, atol=1e-2):
            return 1.0
        return 0.0
    pcc = ((g_centered @ t_centered) / denom).item()
    return max(-1.0, min(1.0, pcc))


def run_and_compare(export_path: str) -> dict:
    """Run original and CPU-bypassed generated code, compare outputs via PCC.

    Args:
        export_path: Directory containing main.py and main_cpu_bypass.py

    Returns:
        Dict with keys: pcc, max_abs_error, rmse, num_outputs
    """
    import ttnn

    original_path = os.path.join(export_path, "main.py")
    bypass_path = os.path.join(export_path, "main_cpu_bypass.py")

    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Original generated code not found: {original_path}")
    if not os.path.exists(bypass_path):
        raise FileNotFoundError(f"CPU bypass code not found: {bypass_path}")

    # Load modules
    original_module = _load_module(original_path, "main_original")
    bypass_module = _load_module(bypass_path, "main_cpu_bypass")

    # Find the forward function (could be named 'forward' or another program name)
    # Look for create_inputs_for_* functions to discover program names
    program_names = []
    for name in dir(original_module):
        if name.startswith("create_inputs_for_"):
            program_names.append(name.replace("create_inputs_for_", ""))

    if not program_names:
        raise RuntimeError(f"No create_inputs_for_* functions found in {original_path}")

    results = []
    for prog_name in program_names:
        create_fn_name = f"create_inputs_for_{prog_name}"
        create_fn_orig = getattr(original_module, create_fn_name)
        create_fn_bypass = getattr(bypass_module, create_fn_name)
        forward_fn_orig = getattr(original_module, prog_name)
        forward_fn_bypass = getattr(bypass_module, prog_name)

        # Create inputs
        inputs_orig = create_fn_orig()
        inputs_bypass = create_fn_bypass()

        # Run both
        print(f"Running original forward (program: {prog_name})...")
        outputs_orig = forward_fn_orig(inputs_orig)
        print(f"Running CPU-bypassed forward (program: {prog_name})...")
        outputs_bypass = forward_fn_bypass(inputs_bypass)

        if not isinstance(outputs_orig, (list, tuple)):
            outputs_orig = [outputs_orig]
        if not isinstance(outputs_bypass, (list, tuple)):
            outputs_bypass = [outputs_bypass]

        # Compare each output
        for i, (orig, bypass) in enumerate(zip(outputs_orig, outputs_bypass)):
            orig_torch = ttnn.to_torch(ttnn.from_device(orig))
            bypass_torch = ttnn.to_torch(ttnn.from_device(bypass))

            pcc = _compute_pcc(orig_torch, bypass_torch)
            diff = (orig_torch.float() - bypass_torch.float()).abs()
            max_abs = diff.max().item()
            rmse = diff.pow(2).mean().sqrt().item()

            results.append(
                {
                    "program": prog_name,
                    "output_index": i,
                    "pcc": pcc,
                    "max_abs_error": max_abs,
                    "rmse": rmse,
                }
            )
            print(
                f"  Output {i}: PCC={pcc:.6f}, "
                f"max_abs_error={max_abs:.6e}, RMSE={rmse:.6e}"
            )

    return {
        "per_output": results,
        "overall_pcc": (
            sum(r["pcc"] for r in results) / len(results) if results else 0.0
        ),
        "num_outputs": len(results),
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_codegen_cpu_bypass(
    model,
    sample_inputs: tuple,
    sample_kwargs: dict,
    export_path: str = "codegen_cpu_bypass_output",
    compiler_options: Optional[dict] = None,
) -> dict:
    """Run the full codegen -> parse -> bypass -> compare pipeline.

    Args:
        model: PyTorch model with weight_dtype_overrides already applied.
               Must be on CPU (not yet transferred to device).
        sample_inputs: Positional arguments for model forward.
        sample_kwargs: Keyword arguments for model forward.
        export_path: Directory to export generated code to.
        compiler_options: Compiler options dict for codegen_py.

    Returns:
        Dict with: groups_found, overall_pcc, per_output results, interpretation.
    """
    from tt_torch.codegen import codegen_py

    # Step 1: Generate device code with real weights
    print(f"Step 1: Generating device code via codegen_py -> {export_path}/")
    codegen_py(
        model,
        *sample_inputs,
        export_path=export_path,
        export_tensors=True,
        compiler_options=compiler_options or {},
        **sample_kwargs,
    )

    # Step 2: Parse generated code for bfp4 matmul patterns
    main_py_path = os.path.join(export_path, "main.py")
    if not os.path.exists(main_py_path):
        raise FileNotFoundError(
            f"codegen_py did not produce {main_py_path}. "
            f"Check that the model compiled successfully."
        )

    with open(main_py_path, "r") as f:
        source_code = f.read()

    print("Step 2: Parsing generated code for bfp4 matmul patterns...")
    groups = find_bfp4_matmul_groups(source_code)
    print(f"  Found {len(groups)} bfp4 matmul group(s)")

    for i, g in enumerate(groups):
        print(
            f"  Group {i}: typecast@line {g.typecast_to_bfp4_line}, "
            f"matmul@line {g.matmul_line}, "
            f"activation={g.matmul_activation}, weight={g.typecast_to_bfp4_output}"
        )

    if not groups:
        print("No bfp4 matmul groups found. Nothing to bypass.")
        return {
            "groups_found": 0,
            "overall_pcc": None,
            "interpretation": "No bfp4 matmuls found in generated code.",
        }

    # Step 3: Generate CPU-bypassed version
    print("Step 3: Generating CPU-bypassed code...")
    bypass_code = generate_cpu_bypass_code(source_code, groups)
    bypass_path = os.path.join(export_path, "main_cpu_bypass.py")
    with open(bypass_path, "w") as f:
        f.write(bypass_code)
    print(f"  Written to {bypass_path}")

    # Step 4: Run both and compare
    print("Step 4: Running both versions and comparing PCC...")
    comparison = run_and_compare(export_path)

    # Step 5: Interpret
    pcc = comparison["overall_pcc"]
    if pcc is not None and pcc < 0.99:
        interpretation = (
            f"PCC={pcc:.6f} < 0.99: The TT matmul kernel is introducing "
            f"significant error beyond quantization. "
            f"Investigate: fp32_dest_acc_en, math fidelity settings."
        )
    elif pcc is not None and pcc < 0.995:
        interpretation = (
            f"PCC={pcc:.6f}: Moderate difference between device and CPU matmul. "
            f"Both quantization and kernel contribute to accuracy loss."
        )
    else:
        interpretation = (
            f"PCC={pcc:.6f} >= 0.995: Device and CPU matmul produce very "
            f"similar results. Accuracy loss is dominated by bfp4 quantization "
            f"itself, not the kernel implementation."
        )

    comparison["groups_found"] = len(groups)
    comparison["interpretation"] = interpretation

    print(f"\n{'='*60}")
    print(f"BFP4 CPU Bypass Accuracy Report")
    print(f"{'='*60}")
    print(f"  BFP4 matmul groups found: {len(groups)}")
    print(f"  Overall PCC (device vs CPU-bypass): {pcc:.6f}")
    print(f"  Interpretation: {interpretation}")
    print(f"{'='*60}")

    return comparison
