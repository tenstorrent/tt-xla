#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Run test_matmul_mp.py once without --forked (relying on the conftest
# cache-clear fixture) and once with --forked, then diff the per-test outcome
# sequences. Used to verify that clear_torchxla_computation_cache provides
# the same isolation as fork()ing for this suite.
#
# Requires: PYTHONPATH already set (or run from the repo root with an active venv).

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

TEST=tests/torch/single_chip/sweeps_derived/test_matmul_mp.py
ART="$(dirname "$0")"

# Common pytest args. -q --no-header keeps output compact and parseable.
ARGS=(--no-header --tb=no -q "$TEST")

echo "[1/2] running WITHOUT --forked"
pytest "${ARGS[@]}" > "$ART/noforked.log" 2>&1 || true

echo "[2/2] running WITH --forked"
pytest --forked "${ARGS[@]}" > "$ART/forked.log" 2>&1 || true

# Pytest's -q output prints a per-test character ('.', 'x', 'F', 'X', 's', 'E')
# in collection order. Extract that strip from both logs and diff. They should
# match byte-for-byte if the cache-clear fixture is equivalent to forking.
extract_dots() {
    grep -E '^(\.|x|F|X|s|E)+' "$1" \
        | tr -d '\n[%]' \
        | tr -d ' 0-9'
}

extract_dots "$ART/noforked.log" > "$ART/.dots_noforked"
extract_dots "$ART/forked.log"   > "$ART/.dots_forked"

echo
echo "noforked:"; grep -E '[0-9]+ (passed|failed|xfailed|xpassed)' "$ART/noforked.log" | tail -1
echo "forked:  "; grep -E '[0-9]+ (passed|failed|xfailed|xpassed)' "$ART/forked.log"   | tail -1
echo

if diff -q "$ART/.dots_noforked" "$ART/.dots_forked" > /dev/null; then
    echo "outcome sequences IDENTICAL"
    rm "$ART/.dots_noforked" "$ART/.dots_forked"
    exit 0
else
    echo "outcome sequences DIFFER:"
    diff "$ART/.dots_noforked" "$ART/.dots_forked"
    exit 1
fi
