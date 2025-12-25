#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

if [ $# -ne 2 ]; then
    echo "Error: Exactly 2 arguments are required."
    echo "Usage: $0 <docker-version-tag> <docker-hash-tag>"
    exit 1
fi

COMMIT_TAG=$1
IS_NIGHTLY_RELEASE=$2

REPO=tenstorrent/tt-xla
IMAGE_NAME=ghcr.io/$REPO/tt-xla-slim
ON_MAIN=$(git branch --show-current | grep -q main && echo "true" || echo "false")
LATEST_TAG="latest"

build_and_push() {
  local image_name=$1
  local dockerfile=$2
  local on_main=$3

  echo "Building image $image_name:$COMMIT_TAG"
  docker build \
    --progress=plain \
    -t $image_name:$COMMIT_TAG \
    -f $dockerfile .

  echo "Pushing image $image_name:$COMMIT_TAG"
  docker push $image_name:$COMMIT_TAG
}

build_and_push $IMAGE_NAME .github/Dockerfile.release $ON_MAIN

echo "Image built and pushed successfully"
echo "IMAGE_NAME:"
echo $IMAGE_NAME:$COMMIT_TAG
