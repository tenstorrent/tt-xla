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

# Take the NEXT stable release after the current pin (not the latest),
# so the uplift advances one release at a time.
RELEASE_TAGS=$(gh release list --repo huggingface/transformers \
  --limit 100 --json tagName,isPrerelease \
  -q '.[] | select(.isPrerelease == false) | .tagName')

LATEST=$(python3 - "$CURRENT" <<PY
import sys
from packaging.version import Version, InvalidVersion
cur = Version(sys.argv[1])
nexts = []
for t in """$RELEASE_TAGS""".split():
    try:
        v = Version(t.lstrip("v"))
    except InvalidVersion:
        continue
    if v > cur:
        nexts.append(v)
print(min(nexts) if nexts else "")
PY
)

HAS_UPDATE="false"
NEW_VERSION=""
if [[ -n "$LATEST" ]]; then
  HAS_UPDATE="true"
  NEW_VERSION="$LATEST"
fi

echo "Current pin:            $CURRENT"
echo "Next release after pin: $LATEST"
echo "Update available:       $HAS_UPDATE"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "current_version=$CURRENT"
    echo "new_version=$NEW_VERSION"
    echo "has_update=$HAS_UPDATE"
  } >> "$GITHUB_OUTPUT"
fi
