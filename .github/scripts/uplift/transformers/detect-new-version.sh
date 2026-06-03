#!/usr/bin/env bash
#
# Detects whether a newer stable release of `transformers` is available on
# PyPI compared to the pin in venv/requirements-dev.txt.
#
# Emits (to $GITHUB_OUTPUT when set, otherwise stdout only):
#   current_version=<X.Y.Z>
#   new_version=<X.Y.Z>   (empty when no update)
#   has_update=<true|false>
#
# Exit code is always 0 unless something goes wrong with the lookup itself
# ("no new version" is not an error).

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
REQ_FILE="${REPO_ROOT}/venv/requirements-dev.txt"

if [[ ! -f "$REQ_FILE" ]]; then
  echo "::error::requirements file not found: $REQ_FILE" >&2
  exit 1
fi

CURRENT=$(grep -oP '^transformers==\K[0-9]+\.[0-9]+\.[0-9]+([a-zA-Z0-9.+-]*)?' "$REQ_FILE" || true)
if [[ -z "$CURRENT" ]]; then
  echo "::error::could not parse 'transformers==X.Y.Z' pin from $REQ_FILE" >&2
  exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "::error::gh CLI not found; this script requires GitHub CLI" >&2
  exit 1
fi

LATEST_TAG=$(gh release view --repo huggingface/transformers --json tagName -q .tagName)
LATEST="${LATEST_TAG#v}"

if [[ -z "$LATEST" ]]; then
  echo "::error::failed to resolve latest transformers release tag" >&2
  exit 1
fi

HAS_UPDATE=$(python3 - "$CURRENT" "$LATEST" <<'PY'
import sys
from packaging.version import Version
cur, new = Version(sys.argv[1]), Version(sys.argv[2])
print("true" if new > cur else "false")
PY
)

NEW_VERSION=""
if [[ "$HAS_UPDATE" == "true" ]]; then
  NEW_VERSION="$LATEST"
fi

echo "Current pin:           $CURRENT"
echo "Latest stable on PyPI: $LATEST"
echo "Update available:      $HAS_UPDATE"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "current_version=$CURRENT"
    echo "new_version=$NEW_VERSION"
    echo "has_update=$HAS_UPDATE"
  } >> "$GITHUB_OUTPUT"
fi
