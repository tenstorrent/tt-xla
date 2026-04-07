#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

if [ $# -ne 3 ]; then
    echo "Error: Exactly 3 arguments are required."
    echo "Usage: $0 <project> <docker-commit-tag> <disto>"
    exit 1
fi

PROJECT=$1
COMMIT_TAG=$2
DISTRO=$3

if [[ "$DISTRO" == "ubuntu" ]]; then
  IMAGE_NAME=ghcr.io/tenstorrent/$PROJECT-slim
else
  IMAGE_NAME=ghcr.io/tenstorrent/$PROJECT-slim-$DISTRO
fi

build_and_push() {
  local image_name=$1
  local dockerfile=$2

  echo "Building image $image_name:$COMMIT_TAG"
  docker build \
    --progress=plain \
    -t $image_name:$COMMIT_TAG \
    -f $dockerfile .

  echo "Pushing image $image_name:$COMMIT_TAG"
  docker push $image_name:$COMMIT_TAG
}

build_and_push $IMAGE_NAME .github/Dockerfile.release-$DISTRO $ON_MAIN

echo "Image built and pushed successfully"
echo $IMAGE_NAME
echo $IMAGE_NAME:$COMMIT_TAG
