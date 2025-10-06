#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path


def parse_logs():
    """Parse decomposition logs and extract all decomposition data"""
    log_file = Path("n150_decomposition_logs/combined_decomposition_logs.txt")
    decomp_stats = {
        "decomposition_core_aten": {},
        "decomposition_default": {},
        "decomposition_custom": {},
    }
    model_stats = {}

    current_model = None

    with open(log_file) as f:
        for line in f:
            line = line.strip()

            if line.startswith("=== MODEL:"):
                current_model = line.split(": ", 1)[1].split(" ===")[0]
                model_stats[current_model] = {
                    "decomposition_core_aten": [],
                    "decomposition_default": [],
                    "decomposition_custom": [],
                }

            elif line.startswith(
                (
                    "decomposition_core_aten:",
                    "decomposition_default:",
                    "decomposition_custom:",
                )
            ):
                parts = line.split(": ", 1)
                decomp_type = parts[0]
                decomp_name = parts[1]

                if decomp_name not in decomp_stats[decomp_type]:
                    decomp_stats[decomp_type][decomp_name] = {"count": 0, "models": []}

                decomp_stats[decomp_type][decomp_name]["count"] += 1
                if (
                    current_model
                    not in decomp_stats[decomp_type][decomp_name]["models"]
                ):
                    decomp_stats[decomp_type][decomp_name]["models"].append(
                        current_model
                    )

                if decomp_name not in model_stats[current_model][decomp_type]:
                    model_stats[current_model][decomp_type].append(decomp_name)

    return decomp_stats, model_stats


def load_decomp_list():
    """Load decomposition list from decomps.txt"""
    with open("decomps.txt") as f:
        return [line.strip() for line in f if line.strip()]


def analyze_usage(decomp_stats, decomp_list):
    """Analyze usage of decompositions from decomps.txt against logs"""
    used_decomps = {}
    unused_decomps = []

    for decomp in decomp_list:
        found = False

        for decomp_type in ["decomposition_default", "decomposition_custom"]:
            if decomp in decomp_stats[decomp_type]:
                used_decomps[decomp] = {
                    "count": decomp_stats[decomp_type][decomp]["count"],
                    "models": decomp_stats[decomp_type][decomp]["models"],
                    "type": decomp_type.replace("decomposition_", ""),
                }
                found = True
                break

        if not found:
            unused_decomps.append(decomp)

    return {
        "used_decompositions": used_decomps,
        "unused_decompositions": unused_decomps,
        "summary": {
            "total_in_decomps_txt": len(decomp_list),
            "used": len(used_decomps),
            "unused": len(unused_decomps),
        },
    }


def main():
    """Main function to generate all statistics"""
    output_dir = Path("n150_decomposition_logs")

    decomp_stats, model_stats = parse_logs()
    decomp_list = load_decomp_list()
    usage_analysis = analyze_usage(decomp_stats, decomp_list)

    with open(output_dir / "decomposition_statistics.json", "w") as f:
        json.dump(decomp_stats, f, indent=2)

    with open(output_dir / "model_statistics.json", "w") as f:
        json.dump(model_stats, f, indent=2)

    with open(output_dir / "decomposition_usage_analysis.json", "w") as f:
        json.dump(usage_analysis, f, indent=2)


if __name__ == "__main__":
    main()
