#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

REPO=tenstorrent/tt-xla
BASE_IMAGE_NAME=ghcr.io/$REPO/tt-xla-base-ubuntu-22-04
CI_IMAGE_NAME=ghcr.io/$REPO/tt-xla-ci-ubuntu-22-04

# Compute the hash of the Dockerfile
DOCKER_TAG=$(./.github/get-docker-tag.sh)
echo "Docker tag: $DOCKER_TAG"

# Are we on main branch
ON_MAIN=$(git branch --show-current | grep -q main && echo "true" || echo "false")

build_and_push() {
    local image_name=$1
    local dockerfile=$2
    local on_main=$3


    if docker manifest inspect $image_name:$DOCKER_TAG > /dev/null; then
        echo "Image $image_name:$DOCKER_TAG already exists"
    else
        echo "Building image $image_name:$DOCKER_TAG"
        docker build \
            --progress=plain \
            --build-arg FROM_TAG=$DOCKER_TAG \
            -t $image_name:$DOCKER_TAG \
            -t $image_name:latest \
            -f $dockerfile .

        echo "Pushing image $image_name:$DOCKER_TAG"
        docker push $image_name:$DOCKER_TAG
    fi

    if [ "$on_main" = "true"]; then
        echo "Pushing latest tag for $image_name"
        docker buildx imagetools create $image_name:$DOCKER_TAG --tag $image_name:latest --tag $image_name:$DOCKER_TAG
    fi
}

build_and_push $BASE_IMAGE_NAME .github/Dockerfile.base $ON_MAIN
build_and_push $CI_IMAGE_NAME .github/Dockerfile.ci $ON_MAIN

echo "All images built and pushed successfully"
echo "CI_IMAGE_NAME:"
echo $CI_IMAGE_NAME:$DOCKER_TAG
