#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Install docs requirements and build the Sphinx documentation.
# This is the entry point used by CI and recommended for first-time
# local builds. Repeat local builds can skip this script and run
# `make -C docs html` directly after the one-time install.
#
# If a venv is already active when this script runs, deps are installed
# into it. Otherwise an isolated venv is created at /tmp/tt-xla-docs-venv.

set -eo pipefail

DOCS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCS_REQ_FILE="${DOCS_DIR}/requirements-docs.txt"
DEFAULT_VENV=/tmp/tt-xla-docs-venv

if [ -n "${VIRTUAL_ENV}" ]; then
  echo "Active venv detected (${VIRTUAL_ENV}); installing docs requirements here."
else
  echo "No active venv: using isolated venv at ${DEFAULT_VENV}."
  if [ ! -x "${DEFAULT_VENV}/bin/python" ]; then
    python3 -m venv "${DEFAULT_VENV}"
  fi
  # shellcheck disable=SC1091
  source "${DEFAULT_VENV}/bin/activate"
fi

echo "Installing docs requirements from ${DOCS_REQ_FILE}..."
pip install --upgrade pip
pip install -r "${DOCS_REQ_FILE}"

echo "Building docs..."
cd "${DOCS_DIR}"
make clean
make html
