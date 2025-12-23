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

VERSION_TAG=$1
IS_NIGHTLY_RELEASE=$2

REPO=tenstorrent/tt-xla
IMAGE_NAME=ghcr.io/$REPO/tt-xla-slim
ON_MAIN=$(git branch --show-current | grep -q main && echo "true" || echo "false")
LATEST_TAG="latest"

if [[ "$IS_NIGHTLY_RELEASE" == "true" ]]; then
  LATEST_TAG="nightly-latest"
fi

build_and_push() {
  local image_name=$1
  local dockerfile=$2
  local on_main=$3

  echo "Building image $image_name:$VERSION_TAG"
  docker build \
    --progress=plain \
    -t $image_name:$VERSION_TAG \
    -f $dockerfile .

  echo "Pushing image $image_name:$VERSION_TAG"
  docker push $image_name:$VERSION_TAG

  if [[ "$on_main" = "true" ]]; then
    printf "\nPushing latest tag for $image_name"
    docker buildx imagetools create $image_name:$DOCKER_TAG --tag $image_name:$LATEST_TAG --tag $image_name:$VERSION_TAG
  fi
}

build_and_push $IMAGE_NAME .github/Dockerfile.release $ON_MAIN

echo "Image built and pushed successfully"
echo "IMAGE_NAME:"
echo $IMAGE_NAME:$VERSION_TAG
