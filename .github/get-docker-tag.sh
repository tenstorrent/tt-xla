#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Calculate hash for docker image tag.
# The hash is based on the MLIR docker tag  and the hash of the Dockerfile(s).

# Exit immediately if a command exits with a non-zero status
set -e

# Execute this in a separate bash process
(
    # Read tt-mlir version from third_party/CMakeLists.txt
    # clone tt-mlir version to tmp/third_party/tt-mlir
    # Get the MLIR docker tag
    TT_MLIR_PATH=third_party/tt-mlir/src/tt-mlir
    TT_MLIR_VERSION=$(grep -oP 'set\(TT_MLIR_VERSION "\K[^"]+' third_party/CMakeLists.txt)
    if [ ! -d $TT_MLIR_PATH ]; then
        git clone https://github.com/tenstorrent/tt-mlir.git $TT_MLIR_PATH --quiet
    fi
    cd $TT_MLIR_PATH
    git fetch --quiet
    git checkout $TT_MLIR_VERSION --quiet
    if [ -f ".github/get-docker-tag.sh" ]; then
        MLIR_DOCKER_TAG=$(.github/get-docker-tag.sh)
    else
        MLIR_DOCKER_TAG="default-tag"
    fi
)

DOCKERFILE_HASH_FILES=".github/Dockerfile.base .github/Dockerfile.ci"
DOCKERFILE_HASH=$( (echo $MLIR_DOCKER_TAG; sha256sum $DOCKERFILE_HASH_FILES) | sha256sum | cut -d ' ' -f 1)
echo dt-$DOCKERFILE_HASH
