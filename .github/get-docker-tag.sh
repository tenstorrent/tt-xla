#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Calculate hash for docker image tag.
# The hash is based on the MLIR docker tag  and the hash of the Dockerfile(s).

# Exit immediately if a command exits with a non-zero status
set -e

# Check if parameter $1 is provided
if [ -z "$1" ]; then
	echo "Error: MLIR_DOCKER_TAG parameter is required" >&2
	exit 1
fi

MLIR_DOCKER_TAG=$1
DOCKERFILE_HASH=$( (cat .github/Dockerfile .github/docker_install.sh venv/requirements-dev.txt python_package/requirements.txt | sha256sum) | cut -d ' ' -f 1)
COMBINED_HASH=$( (echo $DOCKERFILE_HASH $MLIR_DOCKER_TAG | sha256sum) | cut -d ' ' -f 1)
echo dt-$COMBINED_HASH
