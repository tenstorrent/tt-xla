# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Accuracy testing using tt-alchemist generated Python modules.

Parses generated ttnn Python code for matmuls that use bfp4 weights (via
const-eval), applies CPU bypass to those matmuls, and compares PCC between
device and CPU-bypassed runs to isolate quantization loss from TT kernel issues.

The generated code has two files:
- consteval.py: weight preprocessing (includes typecast to BFLOAT4_B)
- main.py: forward pass using cached const-eval results as matmul weights

Bfp4 matmuls are identified by cross-referencing:
1. consteval.py functions that produce BFLOAT4_B outputs
2. main.py matmul calls that consume those cached const-eval results
"""

import os
import re
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Pattern detection for bfp4 matmul groups across main.py + consteval.py
# ---------------------------------------------------------------------------


@dataclass
class BFP4MatmulGroup:
    """A matmul in main.py whose weight operand is a bfp4 const-eval result."""

    # In main.py
    matmul_line: int
    matmul_output: str  # e.g. "ttnn_matmul_3"
    matmul_activation: str  # e.g. "ttnn_repeat_1"
    matmul_weight_ref: str  # e.g. '_cached__main["main_const_eval_17"][0]'
    matmul_src: str  # full source line of the matmul call

    # In consteval.py
    consteval_fn_name: str  # e.g. "main_const_eval_17"


def find_bfp4_consteval_functions(consteval_code: str) -> set[str]:
    """Find const-eval function names that produce BFLOAT4_B outputs.

    Scans consteval.py for functions containing ttnn.typecast(..., BFLOAT4_B, ...)
    and returns their names.
    """
    bfp4_fns = set()

    current_fn = None
    for line in consteval_code.splitlines():
        fn_match = re.match(r"^def\s+(main_const_eval_\w+)\s*\(", line)
        if fn_match:
            current_fn = fn_match.group(1)
        if current_fn and "ttnn.DataType.BFLOAT4_B" in line:
            bfp4_fns.add(current_fn)

    return bfp4_fns


def find_bfp4_matmul_groups(
    main_code: str,
    consteval_code: str,
) -> list[BFP4MatmulGroup]:
    """Find matmuls in main.py that use bfp4 const-eval weights.

    Cross-references consteval.py (which functions produce BFLOAT4_B tensors)
    with main.py (which matmuls consume those cached results).
    """
    bfp4_fns = find_bfp4_consteval_functions(consteval_code)
    if not bfp4_fns:
        return []

    # Build set of cached reference strings that are bfp4
    # e.g. '_cached__main["main_const_eval_17"][0]'
    bfp4_refs = set()
    for fn_name in bfp4_fns:
        bfp4_refs.add(f'_cached__main["{fn_name}"][0]')

    # Multiline matmul pattern: capture the full call across lines
    # ttnn_matmul_X = ttnn.matmul(\n    activation,\n    weight_ref,\n    ...
    lines = main_code.splitlines()

    groups = []
    matmul_start_pattern = re.compile(r"^\s+([\w]+)\s*=\s*ttnn\.matmul\(\s*$")
    # Also handle single-line: ttnn_matmul_X = ttnn.matmul(act, weight, ...)
    matmul_single_pattern = re.compile(
        r'^\s+([\w]+)\s*=\s*ttnn\.matmul\(\s*([\w\[\]"_.]+)\s*,\s*([\w\[\]"_.]+)'
    )

    i = 0
    while i < len(lines):
        line = lines[i]

        # Try single-line match first
        m_single = matmul_single_pattern.match(line)
        if m_single:
            out_var = m_single.group(1)
            arg1 = m_single.group(2)
            arg2 = m_single.group(3)
            weight_ref = None
            act_var = None
            if arg2 in bfp4_refs:
                weight_ref = arg2
                act_var = arg1
            elif arg1 in bfp4_refs:
                weight_ref = arg1
                act_var = arg2
            if weight_ref:
                fn_name = re.search(r'"(main_const_eval_\w+)"', weight_ref).group(1)
                groups.append(
                    BFP4MatmulGroup(
                        matmul_line=i,
                        matmul_output=out_var,
                        matmul_activation=act_var,
                        matmul_weight_ref=weight_ref,
                        matmul_src=line,
                        consteval_fn_name=fn_name,
                    )
                )
            i += 1
            continue

        # Try multiline match: "var = ttnn.matmul(" on one line, args on next lines
        m_multi = matmul_start_pattern.match(line)
        if m_multi:
            out_var = m_multi.group(1)
            # Next two lines should have the two arguments
            if i + 2 < len(lines):
                arg1_line = lines[i + 1].strip().rstrip(",")
                arg2_line = lines[i + 2].strip().rstrip(",")
                weight_ref = None
                act_var = None
                if arg2_line in bfp4_refs:
                    weight_ref = arg2_line
                    act_var = arg1_line
                elif arg1_line in bfp4_refs:
                    weight_ref = arg1_line
                    act_var = arg2_line
                if weight_ref:
                    fn_name = re.search(r'"(main_const_eval_\w+)"', weight_ref).group(1)
                    groups.append(
                        BFP4MatmulGroup(
                            matmul_line=i,
                            matmul_output=out_var,
                            matmul_activation=act_var,
                            matmul_weight_ref=weight_ref,
                            matmul_src=line,
                            consteval_fn_name=fn_name,
                        )
                    )
        i += 1

    return groups


# ---------------------------------------------------------------------------
# CPU bypass code generation
# ---------------------------------------------------------------------------


def _find_matmul_end_line(lines: list[str], start_line: int) -> int:
    """Find the closing paren line of a multiline ttnn.matmul(...) call."""
    depth = 0
    for i in range(start_line, min(start_line + 30, len(lines))):
        depth += lines[i].count("(") - lines[i].count(")")
        if depth <= 0:
            return i
    return start_line


_BYPASS_HELPERS_MARKER = "# === CPU BYPASS HELPERS ==="

_BYPASS_HELPERS = f'''
{_BYPASS_HELPERS_MARKER}
# Mesh shape used by the generated main.py. Kept here so bypass helpers can
# reshape the per-device shard list into a (rows, cols) grid for concat.
_BYPASS_MESH_SHAPE = (4, 8)


def _find_diff_dim(a, b):
    """Return the first dim where a and b differ in shape; None if identical."""
    if list(a.shape) == list(b.shape):
        return None if bool((a == b).all()) else -1   # same shape, diff values => impossible for shards
    for d in range(max(a.ndim, b.ndim)):
        if a.shape[d] != b.shape[d]:
            return d
    return None


def _bypass_to_host_bf16(t):
    """Dequant (if bfp4) -> per-chip shards to host -> reconstruct full logical tensor on CPU.

    Avoids in-graph all_gather (which OOMs on large sharded weights when
    gathered along the wrong dim). Instead pulls each shard to host, detects
    the shard dim per mesh axis by shape-diffing neighbor shards, and
    reconstructs with torch.cat. Works for any shard spec on the (4, 8)
    mesh including (None, None) replication and single-axis sharding.
    """
    import ttnn, torch
    if t.dtype == ttnn.DataType.BFLOAT4_B:
        t = ttnn.typecast(
            t,
            ttnn.DataType.BFLOAT16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    host = ttnn.from_device(t)
    shards_flat = ttnn.get_device_tensors(host)
    rows, cols = _BYPASS_MESH_SHAPE
    # Convert each shard to torch.bfloat16 on CPU.
    grid = [[None] * cols for _ in range(rows)]
    for i, s in enumerate(shards_flat):
        grid[i // cols][i % cols] = ttnn.to_torch(s).to(torch.bfloat16).contiguous()
    # Determine per-axis shard dims by shape-diffing neighbors on the grid.
    col_diff_dim = _find_diff_dim(grid[0][0], grid[0][1]) if cols > 1 else None
    row_diff_dim = _find_diff_dim(grid[0][0], grid[1][0]) if rows > 1 else None
    # First concat along columns (mesh axis 1) within each row.
    row_tensors = []
    for r in range(rows):
        if col_diff_dim is not None and col_diff_dim >= 0:
            row_tensors.append(torch.cat([grid[r][c] for c in range(cols)], dim=col_diff_dim))
        else:
            row_tensors.append(grid[r][0])    # replicated along cols
    # Then concat along rows (mesh axis 0).
    if row_diff_dim is not None and row_diff_dim >= 0:
        return torch.cat(row_tensors, dim=row_diff_dim)
    return row_tensors[0]                      # replicated along rows


def _bypass_from_host_bf16(cpu_tensor, reference_device_tensor):
    """Push a CPU torch tensor back to device with the same mesh as the reference.

    The result is replicated across every chip in the mesh. Downstream ops
    that expected a specific shard spec may see a different layout than the
    device run — but this is consistent between device and bypass runs for
    the purposes of PCC on OUTPUTS (we gather both via the same per-shard
    path on output).
    """
    import ttnn
    return ttnn.from_torch(
        cpu_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=reference_device_tensor.device(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
# === END CPU BYPASS HELPERS ===
'''


def generate_cpu_bypass_code(
    main_code: str,
    groups: list[BFP4MatmulGroup],
) -> str:
    """Generate a modified main.py with CPU-bypassed bfp4 matmuls.

    Emitted per-matmul pattern (matches the user's requested flow):
        W (bfp4 on device)
          -> ttnn.typecast(..., BFLOAT16)   # on-device dequant
          -> ttnn.all_gather(axis=0), all_gather(axis=1)   # gather shards
          -> ttnn.from_device -> ttnn.to_torch -> torch.bfloat16
          -> torch.matmul(act_host, wt_host)
          -> ttnn.from_torch back to device
    The bfp4 weight quantization in consteval.py is preserved (we only
    substitute the matmul computation).
    """
    lines = main_code.splitlines()

    replacements: dict[int, tuple[int, str]] = {}  # start_line -> (end_line, new_code)

    for idx, group in enumerate(groups):
        start = group.matmul_line
        end = _find_matmul_end_line(lines, start)
        indent = "    "
        act = group.matmul_activation
        wt = group.matmul_weight_ref
        out = group.matmul_output

        bypass = f"""{indent}# === CPU BYPASS #{idx} (was ttnn.matmul) ===
{indent}_torch_act_{idx} = _bypass_to_host_bf16({act})
{indent}_torch_wt_{idx} = _bypass_to_host_bf16({wt})
{indent}_cpu_result_{idx} = torch.matmul(_torch_act_{idx}, _torch_wt_{idx})
{indent}{out} = _bypass_from_host_bf16(_cpu_result_{idx}, {act})
{indent}# === END CPU BYPASS #{idx} ==="""

        replacements[start] = (end, bypass)

    # Apply replacements (from bottom to top to preserve line numbers)
    new_lines = []
    skip_until = -1
    for i, line in enumerate(lines):
        if i <= skip_until:
            continue
        if i in replacements:
            end, bypass_code = replacements[i]
            new_lines.append(bypass_code)
            skip_until = end
        else:
            new_lines.append(line)

    result = "\n".join(new_lines)
    if "import torch" not in result:
        result = result.replace("import ttnn", "import ttnn\nimport torch", 1)
    if _BYPASS_HELPERS_MARKER not in result:
        # Inject helpers right after the imports.
        lines_out = result.splitlines()
        insert_at = 0
        for i, line in enumerate(lines_out):
            if line.startswith("import ") or line.startswith("from "):
                insert_at = i + 1
        lines_out.insert(insert_at, _BYPASS_HELPERS)
        result = "\n".join(lines_out)

    return result


# ---------------------------------------------------------------------------
# Orchestrator for tt-alchemist based flow
# ---------------------------------------------------------------------------


def run_alchemist_cpu_bypass(
    generated_dir: str,
) -> dict:
    """Parse generated code, apply CPU bypass, report bfp4 matmul groups.

    Args:
        generated_dir: Directory containing main.py and consteval.py
                       produced by tt-alchemist generate_python().

    Returns:
        Dict with groups_found, bypass code path, group details.
    """
    main_path = os.path.join(generated_dir, "main.py")
    consteval_path = os.path.join(generated_dir, "consteval.py")

    if not os.path.exists(main_path):
        raise FileNotFoundError(f"main.py not found in {generated_dir}")

    with open(main_path, "r") as f:
        main_code = f.read()

    consteval_code = ""
    if os.path.exists(consteval_path):
        with open(consteval_path, "r") as f:
            consteval_code = f.read()
    else:
        # Newer tt-alchemist inlines the main_const_eval_N functions into main.py
        # rather than splitting consteval.py. Scan main.py itself in that case.
        consteval_code = main_code

    # Find bfp4 matmul groups
    groups = find_bfp4_matmul_groups(main_code, consteval_code)
    print(f"Found {len(groups)} bfp4 matmul group(s)")
    for i, g in enumerate(groups):
        print(
            f"  Group {i}: matmul@line {g.matmul_line}, "
            f"output={g.matmul_output}, activation={g.matmul_activation}, "
            f"weight={g.consteval_fn_name}"
        )

    if not groups:
        return {
            "groups_found": 0,
            "bypass_path": None,
            "interpretation": "No bfp4 matmuls found.",
        }

    # Generate CPU-bypassed version
    bypass_code = generate_cpu_bypass_code(main_code, groups)
    bypass_path = os.path.join(generated_dir, "main_cpu_bypass.py")
    with open(bypass_path, "w") as f:
        f.write(bypass_code)
    print(f"CPU-bypassed code written to {bypass_path}")

    return {
        "groups_found": len(groups),
        "bypass_path": bypass_path,
        "groups": [
            {
                "matmul_output": g.matmul_output,
                "matmul_line": g.matmul_line,
                "activation": g.matmul_activation,
                "consteval_fn": g.consteval_fn_name,
            }
            for g in groups
        ],
    }
