#!/bin/bash
# Install docs requirements and build the Sphinx documentation.
# This is the entry point used by CI and recommended for first-time
# local builds. Repeat local builds can skip this script and run
# `make -C docs html` directly after the one-time install.

set -eo pipefail

DOCS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing docs requirements..."
pip install -r "${DOCS_DIR}/requirements-docs.txt"

echo "Building docs..."
cd "${DOCS_DIR}"
make clean
make html
