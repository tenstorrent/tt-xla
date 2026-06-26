# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only guard that keeps the CI perf-benchmark matrix in sync with the code.

The nightly perf pipeline runs whatever ``pytest`` node ids are listed in
``.github/workflows/perf-bench-matrix.json``. When a benchmark test is renamed
or removed, the matrix silently keeps pointing at the old id and the job fails
(or worse, quietly stops running) deep in CI. This test parses the matrix and
asserts every referenced node id still resolves to a function defined in the
named file - caught locally and at PR time instead.

It uses ``ast`` (not ``import``) so it stays free of torch / jax / vllm and runs
anywhere.
"""

import ast
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = REPO_ROOT / ".github" / "workflows" / "perf-bench-matrix.json"


def _matrix_node_ids():
    matrix = json.loads(MATRIX_PATH.read_text())
    node_ids = []
    for block in matrix:
        for test in block.get("tests", []):
            node_ids.append(test["pytest"])
    return node_ids


def _functions_defined_in(path: Path) -> set:
    tree = ast.parse(path.read_text(), filename=str(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_matrix_file_exists():
    assert MATRIX_PATH.is_file(), f"Missing CI matrix at {MATRIX_PATH}"


def test_matrix_is_non_empty():
    assert _matrix_node_ids(), "CI perf-benchmark matrix lists no tests"


@pytest.mark.parametrize("node_id", _matrix_node_ids())
def test_matrix_entry_resolves(node_id):
    rel_path, _, func = node_id.partition("::")
    assert func, f"Matrix entry '{node_id}' is missing a '::test_name'"

    # Drop pytest parametrization id, e.g. test_foo[bar] -> test_foo.
    func_name = func.split("[", 1)[0]

    test_path = REPO_ROOT / rel_path
    assert (
        test_path.is_file()
    ), f"Matrix entry '{node_id}' points at missing file {rel_path}"

    defined = _functions_defined_in(test_path)
    assert func_name in defined, (
        f"Matrix entry '{node_id}' references '{func_name}', which is not defined "
        f"in {rel_path}. Did a benchmark test get renamed or removed?"
    )
