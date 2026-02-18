#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

# Install XLA specific dependencies
# Model dependencies:
# - Whisper and Wav2Vec2 : ffmpeg
pkg_install \
    git-lfs \
    patchelf \
    protobuf-compiler

if [ "$1" == "dnf" ]; then
    # For Manylinux based images
    pkg_install \
        protobuf-devel \
        ffmpeg-free
#        gcc-toolset-12 \
else
    # For Ubuntu based images
    pkg_install \
        libprotobuf-dev \
        ffmpeg
#        g++-12 \
fi

# Install uv tool for managing Python packages
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

ln -sf /usr/bin/FileCheck-17 /usr/bin/FileCheck
