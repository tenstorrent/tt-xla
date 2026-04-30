#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Parse two nightly analysis markdown files (produced by analyze-nightly with the
"save" argument) and classify every test failure as New, Persisting, or Fixed.

Usage:
    python3 analyse_failures.py <baseline.md> <comparison.md>

Output:
    Prints three JSON arrays to stdout:
        { "new": [...], "persisting": [...], "fixed": [...] }

Each item in the arrays is a dict with keys:
    test_id, arch, root_cause, ownership_area,
    baseline_url (persisting/fixed only), comparison_url (new/persisting only)
"""

import json
import re
import sys


def parse_analysis_file(path: str) -> list[dict]:
    """Return a flat list of failure records from an analyze-nightly markdown file."""
    records = []
    current_area = ""
    current_cause = ""

    with open(path) as f:
        for line in f:
            line = line.rstrip()

            # # Ownership area heading
            if line.startswith("# ") and not line.startswith("## "):
                current_area = line[2:].strip()
                current_cause = ""
                continue

            # ## Root-cause heading
            if line.startswith("## "):
                current_cause = line[3:].strip()
                continue

            # - {test-or-step-name} ({arch-list}) -> [job-link]({url})
            if not line.startswith("- "):
                continue

            content = line[2:].strip()

            # Extract job URL from [job-link]({url})
            url_match = re.search(r"\[job-link\]\(([^)]+)\)", content)
            job_url = url_match.group(1) if url_match else ""

            # Strip the " -> [job-link](...)" suffix
            arrow_idx = content.find(" -> ")
            if arrow_idx != -1:
                content = content[:arrow_idx].strip()

            # Split "test_id (arch)" — last parenthesised group is the arch
            paren_match = re.search(r"^(.*?)\s+\(([^)]+)\)\s*$", content)
            if paren_match:
                test_id = paren_match.group(1).strip()
                arch = paren_match.group(2).strip()
            else:
                test_id = content
                arch = ""

            records.append(
                {
                    "test_id": test_id,
                    "arch": arch,
                    "root_cause": current_cause,
                    "ownership_area": current_area,
                    "job_url": job_url,
                }
            )

    return records


def normalise_test_id(test_id: str) -> str:
    """
    Return a canonical key for matching failures across runs.

    Two test_ids are considered the same logical test when they differ only in
    hardware arch suffix. Strip trailing arch tokens of the form:
        -n150, -p150, -n300-llmbox, -galaxy-wh-6u, -n150-perf, etc.
    """
    return re.sub(
        r"[-_](n150|p150|n300|n300-llmbox|galaxy[-\w]*|n150-perf|p150-perf)$",
        "",
        test_id,
        flags=re.IGNORECASE,
    )


def compare(baseline: list[dict], comparison: list[dict]) -> dict:
    """Return {"new": [...], "persisting": [...], "fixed": [...]}."""

    def key(r: dict) -> str:
        return normalise_test_id(r["test_id"])

    baseline_by_key: dict[str, list[dict]] = {}
    for r in baseline:
        baseline_by_key.setdefault(key(r), []).append(r)

    comparison_by_key: dict[str, list[dict]] = {}
    for r in comparison:
        comparison_by_key.setdefault(key(r), []).append(r)

    baseline_keys = set(baseline_by_key)
    comparison_keys = set(comparison_by_key)

    new_items = []
    for k in comparison_keys - baseline_keys:
        for r in comparison_by_key[k]:
            new_items.append({**r, "comparison_url": r["job_url"]})

    fixed_items = []
    for k in baseline_keys - comparison_keys:
        for r in baseline_by_key[k]:
            fixed_items.append({**r, "baseline_url": r["job_url"]})

    persisting_items = []
    for k in baseline_keys & comparison_keys:
        for comp_r in comparison_by_key[k]:
            base_r = baseline_by_key[k][0]
            persisting_items.append(
                {
                    **comp_r,
                    "comparison_url": comp_r["job_url"],
                    "baseline_url": base_r["job_url"],
                }
            )

    return {"new": new_items, "persisting": persisting_items, "fixed": fixed_items}


def render_markdown(result: dict, baseline_run: str, comparison_run: str) -> str:
    """Render the comparison result as a Markdown report (Steps 4-5 of the skill)."""
    lines = []

    def section(title: str, items: list[dict], url_key: str, extra_key: str = ""):
        if not items:
            return
        lines.append(f"\n# {title}\n")
        by_cause: dict[str, list[dict]] = {}
        for item in items:
            by_cause.setdefault(item["root_cause"] or "Unknown", []).append(item)
        for cause, group in by_cause.items():
            lines.append(f"## {cause}\n")
            # Merge arches for the same test_id
            by_test: dict[str, list[dict]] = {}
            for item in group:
                by_test.setdefault(item["test_id"], []).append(item)
            for test_id, variants in by_test.items():
                arches = ", ".join(sorted({v["arch"] for v in variants if v["arch"]}))
                arch_str = f" ({arches})" if arches else ""
                primary_url = variants[0].get(url_key, "")
                link = f"[job-link]({primary_url})" if primary_url else ""
                if extra_key and variants[0].get(extra_key):
                    extra_link = f" [also in baseline]({variants[0][extra_key]})"
                else:
                    extra_link = ""
                lines.append(f"- {test_id}{arch_str} -> {link}{extra_link}")
            lines.append("")

    section(
        f"New Failures (in {comparison_run}, not in {baseline_run})",
        result["new"],
        url_key="comparison_url",
    )
    section(
        f"Persisting Failures (in both {baseline_run} and {comparison_run})",
        result["persisting"],
        url_key="comparison_url",
        extra_key="baseline_url",
    )
    section(
        f"Fixed Since Baseline (in {baseline_run}, not in {comparison_run})",
        result["fixed"],
        url_key="baseline_url",
    )

    return "\n".join(lines)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    baseline_path, comparison_path = sys.argv[1], sys.argv[2]

    # Optional: extract run IDs from filenames like nightly-analysis-12345678.md
    def run_id_from_path(p: str) -> str:
        m = re.search(r"(\d{8,})", p)
        return m.group(1) if m else p

    baseline_run = run_id_from_path(baseline_path)
    comparison_run = run_id_from_path(comparison_path)

    baseline = parse_analysis_file(baseline_path)
    comparison = parse_analysis_file(comparison_path)

    result = compare(baseline, comparison)

    if "--json" in sys.argv:
        print(json.dumps(result, indent=2))
    else:
        print(render_markdown(result, baseline_run, comparison_run))

    # Print a terse summary to stderr so it's visible without polluting stdout
    print(
        f"\nSummary: {len(result['new'])} new, "
        f"{len(result['persisting'])} persisting, "
        f"{len(result['fixed'])} fixed",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
