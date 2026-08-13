#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Makes the image usable as the ttop worker image on the exabox Galaxy cluster
# Requires `pkg_install` to be exported and uv to be on PATH, so run this after
# .github/docker_install.sh.

set -e

# opensssh-server is a deop of sshd
# libatomic1 is a runtime dep of tt-smi's tt_umd native extension
pkg_install openssh-server sudo libatomic1

if ! id -u user > /dev/null 2>&1; then
    # uid 1001 matches the CI runner's uid so files on the shared NFS volume stay
    # writable from both sides
    adduser --uid 1001 --shell /bin/bash --disabled-password --gecos "" user \
        || adduser --shell /bin/bash --disabled-password --gecos "" user
fi
usermod -aG sudo user
echo 'user ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/user
chmod 0440 /etc/sudoers.d/user
mkdir -p /run/sshd
grep -q '^StrictModes no' /etc/ssh/sshd_config || echo "StrictModes no" >> /etc/ssh/sshd_config

uv venv --python 3.12 /opt/tt-smi-venv
uv pip install --python /opt/tt-smi-venv/bin/python --no-cache tt-smi
ln -sf /opt/tt-smi-venv/bin/tt-smi /usr/local/bin/tt-smi
