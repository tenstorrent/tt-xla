#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Convert TTNN MLIR graphs to model_tracer V2-inspired JSON.

Usage:
    python mlir_to_tracer.py test_*.mlir -o traces/
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

# --- Constants ---

EXCLUDED_OPS = frozenset(
    {
        "deallocate",
        "get_device",
        "to_device",
        "from_device",
        "to_layout",
        "to_memory_config",
        "assign",
        "full",
        "constant",
        "arange",
    }
)

DTYPE_MAP = {
    "bf16": "DataType.BFLOAT16",
    "f32": "DataType.FLOAT32",
    "f16": "DataType.FLOAT16",
    "si32": "DataType.INT32",
    "ui32": "DataType.UINT32",
    "ui16": "DataType.UINT16",
    "ui8": "DataType.UINT8",
    "i1": "DataType.BOOLEAN",
    "bfp_f8": "DataType.BFLOAT8_B",
    "bfp_bf8": "DataType.BFLOAT8_B",
}

BUFFER_TYPE_MAP = {
    "dram": "BufferType.DRAM",
    "l1": "BufferType.L1",
    "l1_small": "BufferType.L1_SMALL",
    "system_memory": "BufferType.SYSTEM_MEMORY",
}

MEMORY_LAYOUT_MAP = {
    "interleaved": "TensorMemoryLayout.INTERLEAVED",
    "block_sharded": "TensorMemoryLayout.BLOCK_SHARDED",
    "height_sharded": "TensorMemoryLayout.HEIGHT_SHARDED",
    "width_sharded": "TensorMemoryLayout.WIDTH_SHARDED",
}


@dataclass
class LayoutInfo:
    layout: str  # "Layout.TILE" or "Layout.ROW_MAJOR"
    buffer_type: str  # "BufferType.DRAM" etc.
    memory_layout: str | None  # "TensorMemoryLayout.INTERLEAVED" etc.


# --- Utility ---


def split_at_top_level(text, delimiter):
    """Split text at delimiter, respecting nested <>, (), [] brackets."""
    result = []
    depth = 0
    current = []
    for char in text:
        if char in "<([":
            depth += 1
        elif char in ">)]":
            depth -= 1
        elif char == delimiter and depth == 0:
            result.append("".join(current))
            current = []
            continue
        current.append(char)
    if current:
        result.append("".join(current))
    return result


def extract_balanced(text, pos):
    """Return content between balanced <> starting at text[pos] which must be '<'.

    Skips '>' when preceded by '-' (MLIR arrow '->').
    """
    assert text[pos] == "<"
    depth = 0
    i = pos
    while i < len(text):
        if text[i] == "<":
            depth += 1
        elif text[i] == ">" and (i == 0 or text[i - 1] != "-"):
            depth -= 1
            if depth == 0:
                return text[pos + 1 : i]
        i += 1
    return None


# --- Preamble parsing ---


def parse_layout_body(body, buffer_types):
    """Parse the body of #ttnn.ttnn_layout<BODY> into a LayoutInfo."""
    # Find buffer type by checking for known aliases
    buffer_type_str = None
    for ref, bt in buffer_types.items():
        if ref in body:
            buffer_type_str = BUFFER_TYPE_MAP.get(bt, bt)
            break

    # Memory layout tag: <interleaved>, <block_sharded>, etc.
    mem_match = re.search(
        r"<(interleaved|block_sharded|height_sharded|width_sharded)>", body
    )
    memory_layout = MEMORY_LAYOUT_MAP.get(mem_match.group(1)) if mem_match else None

    # Tile layout: !ttcore.tile<32x32, bf16>
    tile_match = re.search(r"!ttcore\.tile<(\d+)x(\d+),\s*(\w+)>", body)
    if tile_match:
        return LayoutInfo(
            layout="Layout.TILE",
            buffer_type=buffer_type_str or "BufferType.DRAM",
            memory_layout=memory_layout,
        )

    # Row-major
    return LayoutInfo(
        layout="Layout.ROW_MAJOR",
        buffer_type=buffer_type_str or "BufferType.SYSTEM_MEMORY",
        memory_layout=memory_layout,
    )


def parse_preamble(text):
    """Parse buffer type aliases, layout aliases, and mesh shape from MLIR text."""
    buffer_types = {}
    layouts = {}
    mesh_shape = None

    for line in text.split("\n"):
        # Buffer type alias: #dram = #ttnn.buffer_type<dram>
        buf_match = re.match(r"(#\w+)\s*=\s*#ttnn\.buffer_type<(\w+)>", line)
        if buf_match:
            buffer_types[buf_match.group(1)] = buf_match.group(2)
            continue

        # Layout alias: #ttnn_layout1 = #ttnn.ttnn_layout<...>
        layout_match = re.match(r"(#ttnn_layout\w*)\s*=\s*#ttnn\.ttnn_layout<", line)
        if layout_match:
            alias = layout_match.group(1)
            start = line.index("#ttnn.ttnn_layout<") + len("#ttnn.ttnn_layout")
            body = extract_balanced(line, start)
            if body:
                layouts[alias] = parse_layout_body(body, buffer_types)
            continue

        # Mesh shape: ttcore.meshes = #ttcore.meshes<[<"mesh" = 4x8>]>
        mesh_match = re.search(
            r'ttcore\.meshes\s*=\s*#ttcore\.meshes<\[<"mesh"\s*=\s*(\d+)x(\d+)>\]>',
            line,
        )
        if mesh_match:
            mesh_shape = [int(mesh_match.group(1)), int(mesh_match.group(2))]

    return buffer_types, layouts, mesh_shape


# --- Tensor type parsing ---


def parse_tensor_type(type_str, layouts):
    """Parse 'tensor<32x720xbf16, #ttnn_layout14>' into a dict."""
    type_str = type_str.strip()
    if not type_str.startswith("tensor<"):
        return None  # Skip !ttnn.device, etc.

    inner = type_str[len("tensor<") : -1]

    # Split into shape+dtype and optional layout ref
    parts = inner.rsplit(", #", 1)
    shape_dtype_str = parts[0]
    layout_ref = "#" + parts[1] if len(parts) > 1 else None

    # Parse shape and dtype: "32x720xbf16" or "f32" or "8x8x1x64xbf16"
    segments = shape_dtype_str.split("x")
    shape = []
    dtype_str = None
    for i, seg in enumerate(segments):
        if seg.isdigit():
            shape.append(int(seg))
        else:
            dtype_str = "x".join(segments[i:])
            break

    if dtype_str is None:
        dtype_str = "unknown"

    result = {
        "type": "ttnn.Tensor",
        "original_shape": shape,
        "original_dtype": DTYPE_MAP.get(dtype_str, dtype_str),
    }

    if layout_ref and layout_ref in layouts:
        li = layouts[layout_ref]
        result["layout"] = li.layout
        mem_cfg = {}
        if li.memory_layout:
            mem_cfg["memory_layout"] = li.memory_layout
        if li.buffer_type:
            mem_cfg["buffer_type"] = li.buffer_type
        if mem_cfg:
            result["memory_config"] = mem_cfg

    return result


# --- Attribute parsing ---


def parse_attr_value(value):
    """Convert a single MLIR attribute value to a Python type."""
    if value == "true":
        return True
    if value == "false":
        return False

    # Typed integer: "1 : si32", "-1 : i64"
    m = re.match(r"^(-?\d+)\s*:\s*\w+$", value)
    if m:
        return int(m.group(1))

    # Plain integer
    if re.match(r"^-?\d+$", value):
        return int(value)

    # Typed float: "2.0e+00 : f32"
    m = re.match(r"^(-?[\d.]+(?:[eE][+-]?\d+)?)\s*:\s*\w+$", value)
    if m:
        return float(m.group(1))

    # Plain float
    if re.match(r"^-?[\d.]+(?:[eE][+-]?\d+)?$", value):
        return float(value)

    # array<i64: 0, -1>
    m = re.match(r"array<\w+:\s*(.+)>$", value)
    if m:
        return [int(x.strip()) for x in m.group(1).split(",")]

    # [1 : i32, 1 : i32]
    m = re.match(r"^\[(.+)\]$", value)
    if m:
        items = []
        for item in m.group(1).split(","):
            item = item.strip()
            tm = re.match(r"(-?\d+)\s*:\s*\w+", item)
            if tm:
                items.append(int(tm.group(1)))
            elif re.match(r"^-?\d+$", item):
                items.append(int(item))
            else:
                items.append(item)
        return items

    # #ttcore.supportedDataTypes<f32> -> "DataType.FLOAT32"
    m = re.match(r"#ttcore\.supportedDataTypes<(\w+)>", value)
    if m:
        return DTYPE_MAP.get(m.group(1), m.group(1))

    # Everything else: keep as string
    return value


def parse_attributes(attr_str):
    """Parse the content of <{...}> into a dict."""
    attrs = {}
    pairs = split_at_top_level(attr_str, ",")
    for pair in pairs:
        pair = pair.strip()
        eq_idx = pair.find(" = ")
        if eq_idx == -1:
            continue
        key = pair[:eq_idx].strip()
        value = pair[eq_idx + 3 :].strip()
        attrs[key] = parse_attr_value(value)
    return attrs


# --- Op line parsing ---


def parse_op_line(line, layouts):
    """Parse a single TTNN op line. Returns a dict or None if excluded/unparseable."""
    # Find op name
    op_match = re.search(r'"ttnn\.(\w+)"', line)
    if not op_match:
        return None

    op_name = op_match.group(1)
    if op_name in EXCLUDED_OPS:
        return None

    # Parse attributes from <{...}>
    attrs = {}
    attr_start = line.find("<{", op_match.end())
    if attr_start != -1:
        attr_end = line.find("}>", attr_start)
        if attr_end != -1:
            attrs = parse_attributes(line[attr_start + 2 : attr_end])

    # Parse type signature: between "}> : " or ") : " and " loc("
    sig_start = line.find("}> : ")
    if sig_start != -1:
        sig_start += 5  # skip "}> : "
    else:
        # No attribute block — look for ") : (" after the operand list
        sig_start = line.find(") : ", op_match.end())
        if sig_start == -1:
            return None
        sig_start += 4  # skip ") : "

    loc_idx = line.rfind(" loc(")
    if loc_idx == -1:
        return None

    type_sig = line[sig_start:loc_idx].strip()

    # Split at " -> "
    arrow_idx = type_sig.find(" -> ")
    if arrow_idx == -1:
        return None

    input_str = type_sig[:arrow_idx].strip()
    output_str = type_sig[arrow_idx + 4 :].strip()

    # Parse input types
    input_tensors = []
    if input_str.startswith("(") and input_str.endswith(")"):
        inner = input_str[1:-1].strip()
        if inner:
            for t in split_at_top_level(inner, ","):
                tensor = parse_tensor_type(t.strip(), layouts)
                if tensor:
                    input_tensors.append(tensor)

    # Parse output types
    output_tensors = []
    if output_str == "()":
        pass
    elif output_str.startswith("(") and output_str.endswith(")"):
        for t in split_at_top_level(output_str[1:-1], ","):
            tensor = parse_tensor_type(t.strip(), layouts)
            if tensor:
                output_tensors.append(tensor)
    else:
        tensor = parse_tensor_type(output_str, layouts)
        if tensor:
            output_tensors.append(tensor)

    return {
        "op_name": f"ttnn::{op_name}",
        "inputs": input_tensors,
        "outputs": output_tensors,
        "attributes": attrs,
    }


# --- Main parsing ---


def parse_mlir(filepath):
    """Parse a TTNN MLIR file and return (ops_list, mesh_shape, model_name)."""
    text = Path(filepath).read_text()
    _buffer_types, layouts, mesh_shape = parse_preamble(text)

    # Find @main function body and extract ops
    lines = text.split("\n")
    in_main = False
    brace_depth = 0
    ops = []

    for line in lines:
        if "func.func @main(" in line:
            in_main = True
            brace_depth = line.count("{") - line.count("}")
            continue

        if in_main:
            brace_depth += line.count("{") - line.count("}")
            if brace_depth <= 0:
                break
            if '"ttnn.' in line:
                op = parse_op_line(line, layouts)
                if op:
                    ops.append(op)

    return ops, mesh_shape, Path(filepath).stem


# --- JSON assembly ---


def config_key(op):
    """Create a string key for deduplicating op configurations."""
    key_data = {
        "op": op["op_name"],
        "inputs": op["inputs"],
        "attrs": {k: str(v) for k, v in sorted(op["attributes"].items())},
    }
    return json.dumps(key_data, sort_keys=True)


def build_json(ops, mesh_shape, model_name):
    """Build the tracer JSON from a list of parsed ops."""
    # Group by op name, then dedup within each group
    by_op = {}
    for op in ops:
        by_op.setdefault(op["op_name"], []).append(op)

    operations = {}
    for op_name in sorted(by_op):
        seen = {}
        for op in by_op[op_name]:
            key = config_key(op)
            if key in seen:
                seen[key]["count"] += 1
            else:
                seen[key] = {**op, "count": 1}

        configs = []
        for i, op in enumerate(seen.values(), 1):
            config = {"config_id": i}

            # Arguments (inputs)
            arguments = {}
            for j, tensor in enumerate(op["inputs"]):
                arguments[f"arg{j}"] = tensor
            config["arguments"] = arguments

            # Results (outputs)
            if op["outputs"]:
                results = {}
                for j, tensor in enumerate(op["outputs"]):
                    results[f"result{j}"] = tensor
                config["results"] = results

            if op["attributes"]:
                config["attributes"] = op["attributes"]
            config["count"] = op["count"]
            configs.append(config)

        operations[op_name] = {"configurations": configs}

    return {
        "metadata": {
            "source": "tt-xla",
            "model": model_name,
            "mesh_shape": mesh_shape,
            "format_note": "V2-inspired from TTNN MLIR. No runtime execution counts or HF model IDs.",
        },
        "operations": operations,
    }


def build_combined_json(all_results):
    """Merge per-model results into a single combined JSON."""
    model_names = []
    mesh_shapes = {}
    # op_name -> key -> {op, count, models}
    combined = {}

    for ops, mesh_shape, model_name in all_results:
        model_names.append(model_name)
        mesh_shapes[model_name] = mesh_shape

        for op in ops:
            op_name = op["op_name"]
            key = config_key(op)

            if op_name not in combined:
                combined[op_name] = {}

            if key in combined[op_name]:
                combined[op_name][key]["count"] += 1
                if model_name not in combined[op_name][key]["models"]:
                    combined[op_name][key]["models"].append(model_name)
            else:
                combined[op_name][key] = {
                    **op,
                    "count": 1,
                    "models": [model_name],
                }

    operations = {}
    for op_name in sorted(combined):
        configs = []
        for i, entry in enumerate(combined[op_name].values(), 1):
            config = {"config_id": i}
            arguments = {}
            for j, tensor in enumerate(entry["inputs"]):
                arguments[f"arg{j}"] = tensor
            config["arguments"] = arguments

            if entry["outputs"]:
                results = {}
                for j, tensor in enumerate(entry["outputs"]):
                    results[f"result{j}"] = tensor
                config["results"] = results

            if entry["attributes"]:
                config["attributes"] = entry["attributes"]
            config["count"] = entry["count"]
            config["models"] = entry["models"]
            configs.append(config)

        operations[op_name] = {"configurations": configs}

    return {
        "metadata": {
            "source": "tt-xla",
            "models": model_names,
            "mesh_shapes": mesh_shapes,
            "format_note": "V2-inspired from TTNN MLIR. Combined trace across models.",
        },
        "operations": operations,
    }


# --- CLI ---


def main():
    parser = argparse.ArgumentParser(
        description="Convert TTNN MLIR graphs to model_tracer V2-inspired JSON."
    )
    parser.add_argument("mlir_files", nargs="+", help="TTNN MLIR files to convert")
    parser.add_argument(
        "-o",
        "--output-dir",
        default="traces",
        help="Output directory (default: traces)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for filepath in args.mlir_files:
        ops, mesh_shape, model_name = parse_mlir(filepath)
        all_results.append((ops, mesh_shape, model_name))

        # Per-model JSON
        result = build_json(ops, mesh_shape, model_name)
        out_path = out_dir / f"{model_name}_trace.json"
        out_path.write_text(json.dumps(result, indent=2) + "\n")

        n_ops = sum(len(v["configurations"]) for v in result["operations"].values())
        total = sum(
            c["count"]
            for v in result["operations"].values()
            for c in v["configurations"]
        )
        print(
            f"{model_name}: {len(result['operations'])} op types, {n_ops} unique configs, {total} total ops → {out_path}"
        )

    # Combined JSON
    if len(all_results) > 1:
        combined = build_combined_json(all_results)
        combined_path = out_dir / "combined_trace.json"
        combined_path.write_text(json.dumps(combined, indent=2) + "\n")
        n_ops = sum(len(v["configurations"]) for v in combined["operations"].values())
        print(
            f"Combined: {len(combined['operations'])} op types, {n_ops} unique configs → {combined_path}"
        )


if __name__ == "__main__":
    main()
