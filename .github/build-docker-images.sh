#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

# Parse command line arguments
CHECK_ONLY=false
tt_mlir_sha=$1
if [[ "$2" == "--check-only" ]]; then
    CHECK_ONLY=true
fi

# Get tt-mlir and ensure the required docker image exists
TT_MLIR_PATH=.tmp/tt-mlir
cwd=$(pwd)
if [ ! -d $TT_MLIR_PATH ]; then
    git clone https://github.com/tenstorrent/tt-mlir.git $TT_MLIR_PATH --quiet
fi
cd $TT_MLIR_PATH
git fetch --quiet
git checkout $tt_mlir_sha --quiet


if [ -f ".github/get-docker-tag.sh" ]; then
    MLIR_DOCKER_TAG=$(.github/get-docker-tag.sh)
else
    echo "ERROR: No get-docker-tag.sh found in tt-mlir"
    exit 1
fi

if [ "$CHECK_ONLY" = false ]; then
    echo "Ensure tt-mlir docker images with tag: $MLIR_DOCKER_TAG exist"
    if ! ./.github/build-docker-images.sh ci --check-only; then
        echo -e "\033[31mDocker image does not exist.\033[0m"
        echo -e "\033[31mYou should build tt-mlir docker image for sha $tt_mlir_sha first, and then rerun the tt-xla workflow.\033[0m"
        exit 9
    fi
fi

cd $cwd

# Compute the hash of the Dockerfile
DOCKER_TAG=$(./.github/get-docker-tag.sh "$MLIR_DOCKER_TAG")
echo "Docker tag: $DOCKER_TAG"

REPO=tenstorrent/tt-xla
BASE_IMAGE_NAME=ghcr.io/$REPO/tt-xla-base-ubuntu-22-04
CI_IMAGE_NAME=ghcr.io/$REPO/tt-xla-ci-ubuntu-22-04
IRD_IMAGE_NAME=ghcr.io/$REPO/tt-xla-ird-ubuntu-22-04

build_and_push() {
    local image_name=$1
    local dockerfile=$2
    local target_image=$3

    IMAGE_EXISTS=false
    if docker manifest inspect $image_name:$DOCKER_TAG > /dev/null; then
        IMAGE_EXISTS=true
    fi

    if [ "$CHECK_ONLY" = true ] && [ "$IMAGE_EXISTS" = true ]; then
        echo "Image $image_name:$DOCKER_TAG already exists"
        return 0
    elif [ "$CHECK_ONLY" = true ] && [ "$IMAGE_EXISTS" = false ]; then
        echo "Image $image_name:$DOCKER_TAG does not exist (check-only mode)"
        return 2
    elif [ "$CHECK_ONLY" = false ] && [ "$IMAGE_EXISTS" = true ]; then
        echo "Image $image_name:$DOCKER_TAG already exists"
    elif [ "$CHECK_ONLY" = false ] && [ "$IMAGE_EXISTS" = false ]; then
        echo "Docker build neccessary, ensure dependencies for toolchain build..."
        sudo apt-get update && sudo apt-get install -y cmake build-essential

        echo "Building image $image_name:$DOCKER_TAG"
        docker build \
            --progress=plain \
            --build-arg FROM_TAG=$DOCKER_TAG \
            --build-arg MLIR_TAG=$MLIR_DOCKER_TAG \
            ${target_image:+--target $target_image} \
            -t $image_name:$DOCKER_TAG \
            -t $image_name:latest \
            -f $dockerfile .

        echo "Pushing image $image_name:$DOCKER_TAG"
        docker push $image_name:$DOCKER_TAG
    fi

}

build_and_push $BASE_IMAGE_NAME .github/Dockerfile.base
build_and_push $CI_IMAGE_NAME .github/Dockerfile.ci ci
build_and_push $IRD_IMAGE_NAME .github/Dockerfile.ci ird

echo "All images built and pushed successfully"
echo "CI_IMAGE_NAME:"
echo $CI_IMAGE_NAME:$DOCKER_TAG
