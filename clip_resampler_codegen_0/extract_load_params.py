# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Extract tensor loading parameters from the generated main.py.
Creates a complete mapping of arg -> {weight_name, layout, on_device}.
"""

import json
import re

MAIN_PY = "main.py"
WEIGHT_MAP = "weight_map.json"
OUTPUT_FILE = "tensor_load_config.json"


def parse_load_tensor_calls():
    """Parse load_tensor calls from main.py to extract parameters."""
    with open(MAIN_PY, "r") as f:
        content = f.read()

    # Find the load_inputs_for__main function
    match = re.search(
        r"def load_inputs_for__main\(\):(.*?)(?=\ndef |\Z)",
        content,
        re.DOTALL,
    )
    if not match:
        raise RuntimeError("Could not find load_inputs_for__main function")

    func_content = match.group(1)

    # Pattern to match load_tensor calls
    # utils_load_tensor_N = utils.load_tensor(
    #     "./tensors/argM.tensorbin",
    #     ttnn.Layout.TILE or ttnn.Layout.ROW_MAJOR,
    #     ttnn.DataType.BFLOAT16,
    #     utils_DeviceGetter_get_device_172 or None,
    #     ttnn.MemoryConfig(...) or None,
    # )
    pattern = re.compile(
        r"utils_load_tensor_(\d+) = utils\.load_tensor\(\s*"
        r'"./tensors/(arg\d+)\.tensorbin",\s*'
        r"ttnn\.Layout\.(\w+),\s*"
        r"ttnn\.DataType\.(\w+),\s*"
        r"(utils_DeviceGetter_get_device_\d+|None),",
        re.MULTILINE,
    )

    load_configs = {}
    for m in pattern.finditer(func_content):
        tensor_idx = int(m.group(1))
        arg_name = m.group(2)
        layout = m.group(3)
        dtype = m.group(4)
        device_ref = m.group(5)
        on_device = device_ref != "None"

        load_configs[arg_name] = {
            "tensor_idx": tensor_idx,
            "layout": layout,
            "dtype": dtype,
            "on_device": on_device,
        }

    return load_configs


def main():
    # Load weight map
    with open(WEIGHT_MAP, "r") as f:
        weight_map = json.load(f)

    # Parse load parameters
    load_configs = parse_load_tensor_calls()

    # Merge
    result = {}
    for arg_name, config in load_configs.items():
        weight_name = weight_map.get(arg_name)
        result[arg_name] = {
            **config,
            "weight_name": weight_name,
        }

    # Sort by tensor_idx
    result = dict(sorted(result.items(), key=lambda x: x[1]["tensor_idx"]))

    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Extracted {len(result)} tensor configurations to {OUTPUT_FILE}")

    # Stats
    on_device = sum(1 for v in result.values() if v["on_device"])
    tile_layout = sum(1 for v in result.values() if v["layout"] == "TILE")
    print(f"  On device: {on_device}")
    print(f"  TILE layout: {tile_layout}")
    print(f"  ROW_MAJOR layout: {len(result) - tile_layout}")


if __name__ == "__main__":
    main()
