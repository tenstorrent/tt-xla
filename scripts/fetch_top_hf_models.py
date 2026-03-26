#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Fetch the top 10,000 models from Hugging Face filtered by PyTorch and JAX frameworks."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import requests

HF_API_URL = "https://huggingface.co/api/models"
PAGE_SIZE = 1000  # Max allowed by the API
DEFAULT_TOTAL = 100000
FRAMEWORKS = ["pytorch", "jax"]


def fetch_models(library: str, total: int, sort: str = "downloads"):
    """Fetch models for a given library, paginating via Link headers until we reach `total`."""
    models = []
    params = {
        "library": library,
        "sort": sort,
        "direction": "-1",
        "limit": PAGE_SIZE,
    }
    next_url = HF_API_URL

    while next_url and len(models) < total:
        resp = requests.get(next_url, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json()

        if not batch:
            break

        models.extend(batch)
        print(f"  [{library}] Fetched {len(models)} models so far...", file=sys.stderr)

        # Follow the "next" Link header for pagination
        next_url = resp.links.get("next", {}).get("url")
        # After the first request, params are encoded in the next_url from the Link header
        params = None

        # Be polite to the API
        time.sleep(0.5)

    return models[:total]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-n",
        "--total",
        type=int,
        default=DEFAULT_TOTAL,
        help=f"Number of models to fetch per framework (default: {DEFAULT_TOTAL})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path (default: stdout). Use .json or .csv extension.",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="downloads",
        choices=["downloads", "likes", "lastModified", "trending", "createdAt"],
        help="Sort criterion (default: downloads)",
    )
    parser.add_argument(
        "--frameworks",
        nargs="+",
        default=FRAMEWORKS,
        help=f"Frameworks to filter on (default: {FRAMEWORKS})",
    )
    args = parser.parse_args()

    all_models = {}  # keyed by model id to deduplicate across frameworks

    for framework in args.frameworks:
        print(f"Fetching top {args.total} {framework} models...", file=sys.stderr)
        models = fetch_models(framework, args.total, sort=args.sort)
        for m in models:
            model_id = m["id"]
            if model_id not in all_models:
                all_models[model_id] = {
                    "id": model_id,
                    "downloads": m.get("downloads", 0),
                    "likes": m.get("likes", 0),
                    "pipeline_tag": m.get("pipeline_tag", ""),
                    "library_name": m.get("library_name", ""),
                    "tags": m.get("tags", []),
                    "frameworks": [],
                    "created_at": m.get("createdAt", ""),
                }
            # Track which frameworks this model supports
            if framework not in all_models[model_id]["frameworks"]:
                all_models[model_id]["frameworks"].append(framework)

    # Sort combined results by downloads descending
    results = sorted(all_models.values(), key=lambda m: m["downloads"], reverse=True)

    print(
        f"\nTotal unique models: {len(results)} "
        f"(from {sum(len(v['frameworks']) for v in results)} framework-model pairs)",
        file=sys.stderr,
    )

    # Output
    output_path = Path(args.output) if args.output else None

    if output_path and output_path.suffix == ".csv":

        def write_csv(f):
            writer = csv.writer(f)
            writer.writerow(
                [
                    "id",
                    "downloads",
                    "likes",
                    "pipeline_tag",
                    "library_name",
                    "frameworks",
                    "created_at",
                ]
            )
            for m in results:
                writer.writerow(
                    [
                        m["id"],
                        m["downloads"],
                        m["likes"],
                        m["pipeline_tag"],
                        m["library_name"],
                        ";".join(m["frameworks"]),
                        m["created_at"],
                    ]
                )

        if output_path:
            with open(output_path, "w", newline="") as f:
                write_csv(f)
        else:
            write_csv(sys.stdout)
    else:
        # JSON output (default)
        output = json.dumps(results, indent=2)
        if output_path:
            output_path.write_text(output)
        else:
            print(output)

    if output_path:
        print(f"Results written to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
