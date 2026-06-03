#!/usr/bin/env bash
#
# Rewrites the `transformers==X.Y.Z` pin in venv/requirements-dev.txt to the
# given target version. Does not install anything — the caller is responsible
# for reinstalling the environment after this script runs.
#
# Usage:
#   bump-transformers.sh <new_version>

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <new_version>" >&2
  exit 2
fi

NEW_VERSION="$1"
REPO_ROOT="$(git rev-parse --show-toplevel)"
REQ_FILE="${REPO_ROOT}/venv/requirements-dev.txt"

if [[ ! -f "$REQ_FILE" ]]; then
  echo "::error::requirements file not found: $REQ_FILE" >&2
  exit 1
fi

if ! grep -qE '^transformers==[0-9]+\.[0-9]+\.[0-9]+' "$REQ_FILE"; then
  echo "::error::no 'transformers==X.Y.Z' pin found in $REQ_FILE" >&2
  exit 1
fi

OLD_LINE=$(grep -E '^transformers==' "$REQ_FILE")

TMP=$(mktemp)
sed -E "s/^transformers==[0-9]+\.[0-9]+\.[0-9]+([a-zA-Z0-9.+-]*)?$/transformers==${NEW_VERSION}/" \
  "$REQ_FILE" > "$TMP"
mv "$TMP" "$REQ_FILE"

NEW_LINE=$(grep -E '^transformers==' "$REQ_FILE")

echo "Before: ${OLD_LINE}"
echo "After:  ${NEW_LINE}"

if [[ "$OLD_LINE" == "$NEW_LINE" ]]; then
  echo "::warning::pin was already at ${NEW_VERSION}; no change"
fi
