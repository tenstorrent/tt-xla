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
    protobuf-compiler \
    libprotobuf-dev \
    ffmpeg \
    g++-12

# Install uv tool for managing Python packages
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

ln -sf /usr/bin/FileCheck-20 /usr/bin/FileCheck

# --- ttop / exabox multihost worker support -----------------------------------
# The image is used verbatim as the worker image of a ttop SSH environment (see
# .github/workflows/call-test-multihost.yml), which SSHes in as `user` on port
# 2223. ULFM OpenMPI is already provided by the tt-mlir base image.
# libatomic1 is a runtime dep of tt-smi's tt_umd native extension.
pkg_install openssh-server sudo libatomic1
if ! id -u user > /dev/null 2>&1; then
    # uid 1001 matches the CI runner's uid so files written to the shared NFS
    # volume stay writable from both sides. Fall back to any free uid if taken.
    adduser --uid 1001 --shell /bin/bash --disabled-password --gecos "" user \
        || adduser --shell /bin/bash --disabled-password --gecos "" user
fi
usermod -aG sudo user
echo 'user ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/user
chmod 0440 /etc/sudoers.d/user
mkdir -p /run/sshd
grep -q '^StrictModes no' /etc/ssh/sshd_config || echo "StrictModes no" >> /etc/ssh/sshd_config

# tt-smi is used to reset and health check the Galaxies before a multihost run.
# Installed into its own venv (like /opt/tt-triage-venv) because installing it
# system-wide fails: it pulls a newer Pygments and pip cannot uninstall the
# dpkg-owned one ("Cannot uninstall Pygments, RECORD file not found").
uv venv --python 3.12 /opt/tt-smi-venv
uv pip install --python /opt/tt-smi-venv/bin/python --no-cache tt-smi
ln -sf /opt/tt-smi-venv/bin/tt-smi /usr/local/bin/tt-smi
