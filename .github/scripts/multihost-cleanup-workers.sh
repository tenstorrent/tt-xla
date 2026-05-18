#!/usr/bin/env bash
# Stops and removes multihost worker containers on all remote worker hosts.
# Baked into the tt-xla-multihost-controller image.
#
# Usage:
#   multihost-cleanup-workers.sh [hostfile]

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly HOSTFILE="${1:-/etc/mpirun/hostfile}"

WORKER_HOSTFILE="/tmp/mpirun-workers-hostfile"
CONTROLLER_HOSTNAME=$(hostname -s)

awk -v controller="${CONTROLLER_HOSTNAME}" '
  /^[[:space:]]*#/ {next}
  NF == 0 {next}
  {
    host=$1; short=host
    sub(/\..*$/, "", short)
    if (host != controller && short != controller) print
  }
' "${HOSTFILE}" > "${WORKER_HOSTFILE}"

if [[ ! -s "${WORKER_HOSTFILE}" ]]; then
  echo "No worker hosts to clean up"
  exit 0
fi

echo "Stopping worker containers on:"
cat "${WORKER_HOSTFILE}"

mpirun --allow-run-as-root \
  --hostfile "${WORKER_HOSTFILE}" \
  --mca btl_tcp_if_exclude docker0,lo \
  --mca plm_rsh_agent "ssh -A -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -l ubuntu" \
  --tag-output \
  bash -c "docker stop ubuntu-host-mapped 2>/dev/null || true; docker rm ubuntu-host-mapped 2>/dev/null || true; echo \"\$(hostname): cleaned up\"" \
  || true

echo "Worker container cleanup complete"
