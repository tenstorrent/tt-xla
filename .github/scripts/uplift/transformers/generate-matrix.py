#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Generate a `test_suite_custom` JSON matrix for call-test.yml from
the current list of failing test IDs (one per line in --input).

Each matrix entry targets one (device pool, framework, run-mode) combo
the regular nightly model tests cover. The same `-k` filter (joined
from the failing param-ids) is applied to every entry; pytest's marker
filter (`test-mark`) drops tests that don't fit the entry's combo, so
broadcasting the same `-k` across all combos is safe and removes the
need to know each test's target device up front.

If the input file is missing or empty, prints `[]` — the caller's
`filter_empty` gate is what actually skips the base-coverage job.

Usage:
  generate-matrix.py <last_failed.txt>
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# (runs-on, framework, test-mark, parallel-groups, forked)
# Mirrors the device/framework/run-mode coverage of
# .github/workflows/test-matrix-presets/model-test-passing.json so the
# uplift exercises the same matrix as nightly, just narrowed.
TEMPLATES = [
    {
        "runs-on": "n150",
        "framework": "torch",
        "test-mark": "n150 and nightly and inference and expected_passing",
        "parallel-groups": 3,
        "forked": True,
    },
    {
        "runs-on": "p150",
        "framework": "torch",
        "test-mark": "p150 and nightly and inference and expected_passing",
        "parallel-groups": 3,
        "forked": True,
    },
    {
        "runs-on": "n150",
        "framework": "jax",
        "test-mark": "n150 and nightly and expected_passing and not large",
        "parallel-groups": 1,
        "forked": True,
    },
    {
        "runs-on": "p150",
        "framework": "jax",
        "test-mark": "p150 and nightly and expected_passing and not large",
        "parallel-groups": 1,
        "forked": True,
    },
    {
        "runs-on": "n300-llmbox",
        "framework": "torch",
        "test-mark": "n300-llmbox and nightly and tensor_parallel and expected_passing",
        "parallel-groups": 4,
        "forked": False,
    },
    {
        "runs-on": "n300",
        "framework": "torch",
        "test-mark": "n300 and nightly and data_parallel and expected_passing",
        "parallel-groups": 1,
        "forked": False,
    },
]

DIR_BY_FRAMEWORK = {
    "torch": "./tests/runner/test_models.py::test_all_models_torch",
    "jax": "./tests/runner/test_models.py::test_all_models_jax",
}


def extract_param_id(test_id: str) -> str:
    """Pull the `[...]` parametrize content from a pytest test id.

    `tests/runner/test_models.py::test_all_models_torch[albert/masked_lm-Base-...]`
      -> `albert/masked_lm-Base-...`
    """
    m = re.search(r"\[([^\]]+)\]\s*$", test_id.strip())
    return m.group(1) if m else test_id.strip()


def build_matrix(failed: list[str]) -> list[dict]:
    if not failed:
        return []
    param_ids = sorted({extract_param_id(t) for t in failed})
    # pytest's -k accepts `<id> or <id> or ...` — boolean substring match
    # against test names.
    contains = " or ".join(param_ids)
    matrix = []
    for tmpl in TEMPLATES:
        matrix.append(
            {
                "runs-on": tmpl["runs-on"],
                "name": f"uplift-base-coverage-{tmpl['framework']}",
                "dir": DIR_BY_FRAMEWORK[tmpl["framework"]],
                "test-mark": tmpl["test-mark"],
                "contains": contains,
                "parallel-groups": tmpl["parallel-groups"],
                "forge-models": True,
                "forked": tmpl["forked"],
            }
        )
    return matrix


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: generate-matrix.py <last_failed.txt>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(json.dumps([]))
        return 0
    failed = [
        ln.strip()
        for ln in path.read_text().splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    print(json.dumps(build_matrix(failed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
