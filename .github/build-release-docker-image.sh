#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

if [ $# -ne 1 ]; then
    echo "Error: Exactly 1 argument is required."
    echo "Usage: $0 <docker-commit-tag>"
    exit 1
fi

COMMIT_TAG=$1

# todo(vvukoman): change REPO to "tenstorrent" when done to avoid end-user changes
REPO=tenstorrent/tt-xla
IMAGE_NAME=ghcr.io/$REPO/tt-xla-slim

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

build_and_push $IMAGE_NAME .github/Dockerfile.release $ON_MAIN

echo "Image built and pushed successfully"
echo $IMAGE_NAME
echo $IMAGE_NAME:$COMMIT_TAG
