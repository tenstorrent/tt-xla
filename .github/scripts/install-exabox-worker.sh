#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Makes the image usable as the ttop worker image on the exabox Galaxy cluster
# (see .github/workflows/call-test-exabox.yml). Only applied to the base image,
# which is the one the exabox tests run in.
#
# Requires `pkg_install` to be exported and uv to be on PATH, so run this after
# .github/docker_install.sh.

set -e

# The ttop-ssh chart starts sshd in this image and its generated client config
# connects as `user` on port 2223, so both must exist here.
# libatomic1 is a runtime dep of tt-smi's tt_umd native extension.
pkg_install openssh-server sudo libatomic1

if ! id -u user > /dev/null 2>&1; then
    # uid 1001 matches the CI runner's uid so files on the shared NFS volume stay
    # writable from both sides. Fall back to any free uid if 1001 is taken.
    adduser --uid 1001 --shell /bin/bash --disabled-password --gecos "" user \
        || adduser --shell /bin/bash --disabled-password --gecos "" user
fi
usermod -aG sudo user
echo 'user ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/user
chmod 0440 /etc/sudoers.d/user
mkdir -p /run/sshd
# Must come after the apt install, which writes a fresh sshd_config. Without it
# sshd rejects key auth on the chart-injected authorized_keys.
grep -q '^StrictModes no' /etc/ssh/sshd_config || echo "StrictModes no" >> /etc/ssh/sshd_config

# tt-smi resets and health checks the Galaxies before a test run. Installed into
# its own venv (like /opt/tt-triage-venv) because installing it system-wide fails:
# it pulls a newer Pygments and pip cannot uninstall the dpkg-owned one
# ("Cannot uninstall Pygments, RECORD file not found").
uv venv --python 3.12 /opt/tt-smi-venv
uv pip install --python /opt/tt-smi-venv/bin/python --no-cache tt-smi
ln -sf /opt/tt-smi-venv/bin/tt-smi /usr/local/bin/tt-smi
