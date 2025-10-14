#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

# Start SSHD in the background
service ssh start

# Exec the passed command (replace shell with target command)
exec "$@"
