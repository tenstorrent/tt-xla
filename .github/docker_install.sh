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
    # For Mulitlinux based images
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

ln -sf /usr/bin/FileCheck-17 /usr/bin/FileCheck

# Now do the python requirements installation
python3.11 -m pip install --upgrade pip --no-cache-dir
python3.11 -m pip install --index-url https://download.pytorch.org/whl/cpu $(grep 'torch==' /tmp/requirements.txt)
python3.11 -m pip install --index-url https://download.pytorch.org/whl/cpu $(grep 'torchvision==' /tmp/requirements-dev.txt)
python3.11 -m pip install -r /tmp/requirements.txt --no-cache-dir
python3.11 -m pip install -r /tmp/requirements-dev.txt --no-cache-dir
