#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Generate XML or CSV from model_configs.yaml with model, status, and reason
for each config that is not commented out. Commented-out entries are omitted
because the YAML parser does not include them.
CSV format imports cleanly into Google Sheets (File > Import > Upload).
"""

import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create XML or CSV from model_configs.yaml (model, status, reason per config)."
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=Path(__file__).resolve().parent / "model_configs.yaml",
        help="Path to model_configs.yaml",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output path (default: model_configs.xml or model_configs.csv based on --format)",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=("xml", "csv"),
        default="xml",
        help="Output format: xml or csv (csv imports into Google Sheets via File > Import > Upload)",
    )
    args = parser.parse_args()

    out = args.output or Path(f"model_configs.{args.format}")

    try:
        import yaml
    except ImportError:
        raise SystemExit(
            "This script requires PyYAML. Install with: pip install pyyaml"
        ) from None

    with open(args.yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    model_configs = data.get("model_configs") or {}
    rows = []
    for config_name, config in sorted(model_configs.items()):
        if not isinstance(config, dict):
            continue
        model = config.get("model", "") or ""
        status = config.get("status", "unspecified")
        reason = config.get("reason", "") or ""
        rows.append((str(model), str(status), str(reason)))

    if args.format == "csv":
        with open(out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(("model_name", "status", "reason"))
            w.writerows(rows)
    else:
        root = ET.Element("model_configs")
        for model, status, reason in rows:
            entry = ET.SubElement(root, "config")
            ET.SubElement(entry, "model_name").text = model
            ET.SubElement(entry, "status").text = status
            ET.SubElement(entry, "reason").text = reason
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        with open(out, "wb") as f:
            tree.write(f, encoding="utf-8", default_namespace="", xml_declaration=True)

    print(f"Wrote {len(rows)} configs to {out}")


if __name__ == "__main__":
    main()
