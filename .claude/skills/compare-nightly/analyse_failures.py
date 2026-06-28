#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Parse two nightly analysis markdown files (produced by analyze-nightly with the
"save" argument) and classify every test failure as New, Persisting, or Fixed.

Usage:
    python3 analyse_failures.py <baseline.md> <comparison.md>
    python3 analyse_failures.py <baseline.md> <comparison.md> --json

Output:
    Markdown report (default) or, with --json, a structured object:
        { "new": [...], "persisting": [...], "fixed": [...],
          "reconciled": [...] }

Each failure record is a dict with keys:
    test_id, arch, root_cause, ownership_area,
    baseline_url (persisting/fixed only), comparison_url (new/persisting only)

Matching across runs is done in two passes:
  1. Exact match on the arch-normalised test_id.
  2. Fuzzy reconciliation of the leftovers, to absorb the case where the two
     analyze-nightly passes labelled the SAME underlying failure with different
     naming conventions (e.g. a "perf vllm_bge_m3" label vs a
     "test_vllm_benchmarks.py::...[bge_m3]" pytest nodeid). Without this, such a
     failure is double-counted as both New and Fixed. Reconciled pairs are
     reclassified as Persisting and reported, with a warning, so the drift is
     visible rather than silently inflating the counts.
"""

import json
import re
import sys

# ---------------------------------------------------------------------------
# Arch handling
# ---------------------------------------------------------------------------

# Known hardware-arch tokens that may be appended to a test id. Longest first so
# the alternation prefers the most specific match (e.g. "n300-llmbox" before
# "n300", "n150-perf" before "n150").
ARCH_TOKENS = sorted(
    [
        "n150-perf",
        "p150-perf",
        "n300-llmbox",
        "galaxy-wh-6u",
        "qb2-blackhole",
        "n300-perf",
        "llmbox",
        "blackhole",
        "galaxy",
        "n150",
        "p150",
        "n300",
        "qb2",
    ],
    key=len,
    reverse=True,
)

_ARCH_ALT = "|".join(re.escape(t) for t in ARCH_TOKENS)
# A trailing arch token preceded by a separator, OR any "galaxy..." variant.
_ARCH_SUFFIX_RE = re.compile(
    rf"[-_ ]({_ARCH_ALT}|galaxy[-\w]*)$",
    flags=re.IGNORECASE,
)


def normalise_test_id(test_id: str) -> str:
    """
    Return a canonical key for matching failures across runs.

    Two test_ids are considered the same logical test when they differ only in
    hardware arch suffix(es). Strips *all* trailing arch tokens (looping, since a
    test id can carry more than one), e.g. "...-n150", "...-p150-perf".
    """
    prev = None
    out = test_id.strip()
    while out != prev:
        prev = out
        out = _ARCH_SUFFIX_RE.sub("", out).strip()
    return out


# ---------------------------------------------------------------------------
# Token signatures (for fuzzy cross-naming reconciliation)
# ---------------------------------------------------------------------------

# Words that carry no discriminating signal — boilerplate from pytest nodeids,
# file paths, perf labels, and run-mode tags. Dropping these (consistently on
# both sides) lets a perf label and a pytest nodeid for the same test share a
# token signature.
_BOILERPLATE = {
    "test",
    "tests",
    "perf",
    "py",
    "pytest",
    "torch",
    "pytorch",
    "jax",
    "models",
    "model",
    "graphs",
    "single",
    "device",
    "causal",
    "lm",
    "tt",
    "xla",
    "inference",
    "training",
    "default",
    "instruct",
    "it",
    "base",
}


def signature(test_id: str) -> frozenset:
    """
    Return the set of discriminating tokens for a test id.

    Lowercased, split on any non-alphanumeric run, with boilerplate, arch
    tokens, and single-digit noise removed. Used only to reconcile leftovers
    that did not match exactly; it deliberately ignores naming convention,
    file path, and ordering.
    """
    raw = normalise_test_id(test_id).lower()
    tokens = re.split(r"[^a-z0-9]+", raw)
    arch_set = {t.lower() for t in ARCH_TOKENS}
    sig = set()
    for tok in tokens:
        if not tok:
            continue
        if tok in _BOILERPLATE:
            continue
        if tok in arch_set:
            continue
        if tok.isdigit() and len(tok) == 1:
            continue  # lone digit from e.g. "qwen2.5" -> drop noisy "2","5"
        sig.add(tok)
    return frozenset(sig)


# Reconciliation thresholds. A leftover-new and a leftover-fixed are treated as
# the same logical test when they share at least MIN_SHARED discriminating
# tokens AND the smaller signature is at least MIN_CONTAINMENT contained in the
# larger. 0.8 (not 0.75) so that e.g. {vllm,bge,m3,batch1} vs
# {vllm,bge,m3,batch32} (containment 0.75) is NOT merged, while a label fully
# contained in a nodeid (containment ~1.0) is.
MIN_SHARED = 2
MIN_CONTAINMENT = 0.8


def _match_score(a: frozenset, b: frozenset):
    """Return (shared_count, containment) or None if below thresholds."""
    if not a or not b:
        return None
    shared = len(a & b)
    if shared < MIN_SHARED:
        return None
    containment = shared / min(len(a), len(b))
    if containment < MIN_CONTAINMENT:
        return None
    return (shared, containment)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_analysis_file(path: str) -> list:
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

            # Extract job URL from [job-link]({url}) (accept any link label)
            url_match = re.search(r"\[[^\]]*\]\((https?://[^)]+)\)", content)
            job_url = url_match.group(1) if url_match else ""

            # Strip the " -> ..." suffix
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


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _baseline_url_for(comp_record: dict, base_records: list) -> str:
    """Pick the baseline job URL whose arch matches the comparison variant."""
    for b in base_records:
        if b["arch"] and b["arch"] == comp_record["arch"]:
            return b["job_url"]
    return base_records[0]["job_url"] if base_records else ""


def _make_persisting(comp_records: list, base_records: list, reconciled: bool):
    """Build persisting items from grouped comparison + baseline records."""
    base_cause = base_records[0]["root_cause"] if base_records else ""
    items = []
    for comp_r in comp_records:
        item = {
            **comp_r,
            "comparison_url": comp_r["job_url"],
            "baseline_url": _baseline_url_for(comp_r, base_records),
            "root_cause_baseline": base_cause,
            "root_cause_changed": bool(
                base_cause and base_cause != comp_r["root_cause"]
            ),
            "reconciled": reconciled,
        }
        items.append(item)
    return items


def compare(baseline: list, comparison: list) -> dict:
    """Return {"new", "persisting", "fixed", "reconciled", "warnings"}."""

    def key(r: dict) -> str:
        return normalise_test_id(r["test_id"])

    baseline_by_key = {}
    for r in baseline:
        baseline_by_key.setdefault(key(r), []).append(r)

    comparison_by_key = {}
    for r in comparison:
        comparison_by_key.setdefault(key(r), []).append(r)

    baseline_keys = set(baseline_by_key)
    comparison_keys = set(comparison_by_key)

    persisting_items = []
    # Pass 1: exact arch-normalised match.
    for k in baseline_keys & comparison_keys:
        persisting_items.extend(
            _make_persisting(comparison_by_key[k], baseline_by_key[k], reconciled=False)
        )

    # Leftovers after exact matching.
    new_keys = comparison_keys - baseline_keys
    fixed_keys = baseline_keys - comparison_keys

    # Pass 2: fuzzy reconciliation of leftovers across naming conventions.
    # Build candidate pairs, then greedily assign best-first so that e.g.
    # batch1<->batch1 wins over batch1<->batch32.
    new_sigs = {k: signature(k) for k in new_keys}
    fixed_sigs = {k: signature(k) for k in fixed_keys}

    candidates = []
    for nk in new_keys:
        for fk in fixed_keys:
            score = _match_score(new_sigs[nk], fixed_sigs[fk])
            if score is not None:
                candidates.append((score, nk, fk))
    # Highest (shared, containment) first.
    candidates.sort(key=lambda c: c[0], reverse=True)

    matched_new = set()
    matched_fixed = set()
    reconciled_pairs = []  # (new_key, fixed_key)
    for _score, nk, fk in candidates:
        if nk in matched_new or fk in matched_fixed:
            continue
        matched_new.add(nk)
        matched_fixed.add(fk)
        reconciled_pairs.append((nk, fk))

    reconciled_items = []
    warnings = []
    for nk, fk in reconciled_pairs:
        reconciled_items.extend(
            _make_persisting(
                comparison_by_key[nk], baseline_by_key[fk], reconciled=True
            )
        )
        warnings.append(
            f'reconciled (naming drift): comparison "{nk}" <-> baseline "{fk}"'
        )
    persisting_items.extend(reconciled_items)

    new_items = []
    for k in new_keys - matched_new:
        for r in comparison_by_key[k]:
            new_items.append({**r, "comparison_url": r["job_url"]})

    fixed_items = []
    for k in fixed_keys - matched_fixed:
        for r in baseline_by_key[k]:
            fixed_items.append({**r, "baseline_url": r["job_url"]})

    return {
        "new": new_items,
        "persisting": persisting_items,
        "fixed": fixed_items,
        "reconciled": reconciled_items,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_markdown(result: dict, baseline_run: str, comparison_run: str) -> str:
    lines = []

    if result.get("warnings"):
        lines.append(
            "> **Note:** the two analysis files used inconsistent test-ID naming "
            "for some failures; the following were matched across naming "
            "conventions and counted as *persisting* (not new+fixed):"
        )
        for w in result["warnings"]:
            lines.append(f"> - {w}")
        lines.append("")

    def section(title, items, url_key, link_label, extra_key="", extra_label=""):
        if not items:
            return
        lines.append(f"\n# {title}\n")
        by_cause = {}
        for item in items:
            by_cause.setdefault(item["root_cause"] or "Unknown", []).append(item)
        for cause, group in by_cause.items():
            lines.append(f"## {cause}\n")
            by_test = {}
            for item in group:
                by_test.setdefault(item["test_id"], []).append(item)
            for test_id, variants in by_test.items():
                arches = ", ".join(sorted({v["arch"] for v in variants if v["arch"]}))
                arch_str = f" ({arches})" if arches else ""
                primary = variants[0]
                url = primary.get(url_key, "")
                link = f"[{link_label}]({url})" if url else ""
                extra = ""
                if extra_key and primary.get(extra_key):
                    extra = f" [{extra_label}]({primary[extra_key]})"
                note = ""
                if primary.get("root_cause_changed"):
                    note = (
                        f' _(root cause changed: "{primary.get("root_cause_baseline")}"'
                        f' -> "{primary["root_cause"]}")_'
                    )
                if primary.get("reconciled"):
                    note += " _(matched across naming drift)_"
                lines.append(f"- {test_id}{arch_str} -> {link}{extra}{note}")
            lines.append("")

    section(
        f"New Failures (in {comparison_run}, not in {baseline_run})",
        result["new"],
        url_key="comparison_url",
        link_label="job-link",
    )
    section(
        f"Persisting Failures (in both {baseline_run} and {comparison_run})",
        result["persisting"],
        url_key="comparison_url",
        link_label="job-link",
        extra_key="baseline_url",
        extra_label="also in baseline",
    )
    section(
        f"Fixed Since Baseline (in {baseline_run}, not in {comparison_run})",
        result["fixed"],
        url_key="baseline_url",
        link_label="baseline job",
    )

    return "\n".join(lines)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    baseline_path, comparison_path = sys.argv[1], sys.argv[2]

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

    for w in result["warnings"]:
        print(f"WARNING: {w}", file=sys.stderr)
    print(
        f"\nSummary: {len(result['new'])} new, "
        f"{len(result['persisting'])} persisting "
        f"({len(result['reconciled'])} reconciled across naming drift), "
        f"{len(result['fixed'])} fixed",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
