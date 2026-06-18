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
readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly BUILD_WORKER_HOSTFILE_SH="${REPO_ROOT}/.github/scripts/multihost-build-worker-hostfile.sh"

WORKER_HOSTFILE="/tmp/mpirun-workers-hostfile"
CONTROLLER_HOSTNAME=$(hostname -s)

if [[ ! -f "${BUILD_WORKER_HOSTFILE_SH}" ]]; then
  echo "worker hostfile builder script not found at ${BUILD_WORKER_HOSTFILE_SH}" >&2
  exit 1
fi

"${BUILD_WORKER_HOSTFILE_SH}" "${HOSTFILE}" "${WORKER_HOSTFILE}"

if [[ ! -s "${WORKER_HOSTFILE}" ]]; then
  echo "No worker hosts to clean up"
  exit 0
fi

echo "Stopping worker containers on:"
cat "${WORKER_HOSTFILE}"

DOCKER_CLEANUP_CMD="
  set -euo pipefail
  CONTAINER_NAME=ubuntu-host-mapped
  if ! command -v docker >/dev/null 2>&1; then
    echo \"\$(hostname): docker not found in PATH, skipping cleanup\" >&2
    exit 0
  fi
  if docker ps -a --filter 'name=^\${CONTAINER_NAME}\$' --format '{{.Names}}' \
     | grep -q \"\${CONTAINER_NAME}\"; then
    docker rm -f \"\${CONTAINER_NAME}\" >/dev/null
    echo \"\$(hostname): removed \${CONTAINER_NAME}\"
  else
    echo \"\$(hostname): no \${CONTAINER_NAME} container found\"
  fi
"

mpirun --allow-run-as-root \
  --hostfile "${WORKER_HOSTFILE}" \
  --mca btl_tcp_if_exclude docker0,lo \
  --mca plm_rsh_agent "ssh -A -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -l ubuntu" \
  --tag-output \
  bash -c "${DOCKER_CLEANUP_CMD}"

echo "Worker container cleanup complete"
