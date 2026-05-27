#!/usr/bin/env bash
# Starts multihost worker containers on all remote worker hosts.
#
# Usage:
#   multihost-start-workers.sh <worker-image> [hostfile]
#
# Required env vars:
#   HOST_WORKSPACE_PATH  - workspace path as seen from the worker's host kernel
#                          (i.e. ${{ github.workspace }} in the calling workflow)
#
# Optional env vars:
#   CONTAINER_WORKSPACE  - mount target inside worker containers (default: $PWD)
#   SSH_AUTH_SOCK        - forwarded SSH agent socket (picked up automatically
#                          when passed via `docker run -e SSH_AUTH_SOCK=...`)

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly WORKER_IMAGE="${1:?Usage: $0 <worker-image> [hostfile]}"
readonly HOSTFILE="${2:-/etc/mpirun/hostfile}"
readonly CONTAINER_WORKSPACE="${CONTAINER_WORKSPACE:-$(pwd)}"
readonly HOST_WORKSPACE_PATH="${HOST_WORKSPACE_PATH:?HOST_WORKSPACE_PATH must be set to the host-side workspace path}"

WORKER_HOSTFILE="/tmp/mpirun-workers-hostfile"
CONTROLLER_HOSTNAME=$(hostname -s)

# Build the filtered worker-only hostfile (exclude the controller)
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
  echo "No worker hosts found after filtering controller (${CONTROLLER_HOSTNAME}), skipping"
  exit 0
fi

echo "Starting worker containers on:"
cat "${WORKER_HOSTFILE}"


# Shell fragment executed on each worker host via mpirun+SSH.
# Uses single-quotes inside the outer double-quoted heredoc deliberately:
# the variables are expanded HERE (on the controller) before being sent.
DOCKER_RUN_CMD="
  CONTAINER_NAME=ubuntu-host-mapped
  if docker ps --filter 'name=^\${CONTAINER_NAME}\$' --format '{{.Names}}' \
     | grep -q \"\${CONTAINER_NAME}\"; then
    echo \"\$(hostname): container \${CONTAINER_NAME} already running, skipping\"
    exit 0
  fi
  docker run --rm -d \\
    --name \"\${CONTAINER_NAME}\" \\
    --pid=host --network=host \\
    --device /dev/tenstorrent \\
    -v /dev/hugepages:/dev/hugepages \\
    -v /dev/hugepages-1G:/dev/hugepages-1G \\
    -v /etc/udev/rules.d:/etc/udev/rules.d \\
    -v /lib/modules:/lib/modules \\
    -v /opt/tt_metal_infra/provisioning/provisioning_env:/opt/tt_metal_infra/provisioning/provisioning_env \\
    -v /mnt/dockercache:/mnt/dockercache \\
    -v /etc/mpirun:/etc/mpirun:ro \\
    -v '${HOST_WORKSPACE_PATH}:${CONTAINER_WORKSPACE}' \\
    -w '${CONTAINER_WORKSPACE}' \\
    '${WORKER_IMAGE}' sleep infinity
  echo \"\$(hostname): container \${CONTAINER_NAME} started\"
"

mpirun --allow-run-as-root \
  --hostfile "${WORKER_HOSTFILE}" \
  --mca btl_tcp_if_exclude docker0,lo \
  --mca plm_rsh_agent "ssh -A -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -l ubuntu" \
  --tag-output \
  bash -c "${DOCKER_RUN_CMD}"

echo "All worker containers started successfully"


echo "=== SSH connectivity check (via remote_docker.sh) ==="
mpirun --allow-run-as-root --hostfile /etc/mpirun/hostfile \
  --mca btl_tcp_if_exclude docker0,lo \
  --mca plm_rsh_agent ${{ steps.strings.outputs.work-dir }}/tests/torch/multi_host/experimental/remote_docker.sh \
  --tag-output \
  bash -c "
    echo \"\$(hostname): hostname OK\"
    if [ -f /.dockerenv ]; then
      echo \"\$(hostname): /.dockerenv EXISTS (running inside container)\"
    else
      echo \"\$(hostname): /.dockerenv NOT FOUND (running on bare metal)\"
    fi
  "
echo "=== End SSH connectivity check (via remote_docker.sh) ==="
