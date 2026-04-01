#!/usr/bin/env python3
"""Parse ttnn.mlir to extract arg index -> weight name mapping.

Reads the func.func @main declaration and extracts each %argN's ttir.name
attribute to produce a JSON mapping file: {arg_index: weight_name}.

Also extracts tensor shape and dtype for each arg.

Usage:
    python parse_mlir_args.py [--mlir ttnn.mlir] [--output arg_mapping.json]
"""

import argparse
import json
import re
from pathlib import Path


def parse_mlir_args(mlir_path: str) -> dict:
    """Parse MLIR file and extract arg index -> {name, shape, dtype} mapping."""
    text = Path(mlir_path).read_text()

    # Find the func.func @main declaration (very long, contains all %argN)
    main_match = re.search(r"func\.func @main\((.+?)\)\s*->", text, re.DOTALL)
    if not main_match:
        raise ValueError("Could not find func.func @main in MLIR file")

    func_args = main_match.group(1)

    # Split on %arg boundaries to get one chunk per argument
    arg_chunks = re.split(r"(?=%arg\d+:)", func_args)

    arg_re = re.compile(r"%arg(\d+):\s*tensor<([^,>]+)")
    name_re = re.compile(r'ttir\.name\s*=\s*"([^"]*)"')

    mapping = {}
    for chunk in arg_chunks:
        arg_match = arg_re.search(chunk)
        name_match = name_re.search(chunk)
        if not arg_match or not name_match:
            continue

        arg_idx = int(arg_match.group(1))
        tensor_desc = arg_match.group(2).strip()
        weight_name = name_match.group(1)

        # Parse shape and dtype from "64x3840xbf16"
        parts = tensor_desc.split("x")
        dtype = parts[-1]
        shape = [int(p) for p in parts[:-1]] if len(parts) > 1 else []

        mapping[arg_idx] = {
            "name": weight_name,
            "shape": shape,
            "dtype": dtype,
        }

    return mapping


def analyze_mapping(mapping: dict) -> None:
    """Print summary analysis of the mapping."""
    print(f"Total args: {len(mapping)}")

    # Group by module prefix
    modules = {}
    for idx in sorted(mapping.keys()):
        name = mapping[idx]["name"]
        prefix = name.split(".")[0]
        modules.setdefault(prefix, []).append((idx, name))

    print(f"\nTop-level modules ({len(modules)}):")
    for prefix, entries in sorted(modules.items()):
        print(f"  {prefix}: {len(entries)} weights")

    # Show full structure
    print(f"\nComplete arg -> name mapping:")
    for idx in sorted(mapping.keys()):
        info = mapping[idx]
        shape_str = "x".join(str(s) for s in info["shape"])
        print(f"  arg{idx}: {info['name']} [{shape_str} {info['dtype']}]")


def main():
    parser = argparse.ArgumentParser(description="Parse MLIR arg-to-weight mapping")
    parser.add_argument("--mlir", default="ttnn.mlir", help="Path to MLIR file")
    parser.add_argument("--output", default="arg_mapping.json", help="Output JSON path")
    parser.add_argument("--verbose", action="store_true", help="Print detailed analysis")
    args = parser.parse_args()

    mapping = parse_mlir_args(args.mlir)

    # Save as JSON (with string keys for JSON compat, sorted by index)
    output = {str(k): v for k, v in sorted(mapping.items())}
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"Wrote {len(mapping)} arg mappings to {args.output}")

    if args.verbose:
        analyze_mapping(mapping)


if __name__ == "__main__":
    main()
