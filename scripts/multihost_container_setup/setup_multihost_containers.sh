#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Multihost CI helper (invoked only from .github/workflows/call-test.yml).
# Args: <subcommand> <docker-image> <host-workspace> <container-workspace>
#
# host-workspace and container-workspace are the two sides of the repo bind mount
# (they differ; both come from the workflow and are the same on controller and workers).

set -euo pipefail

readonly MULTIHOST_CONTAINER_NAME="${MULTIHOST_CONTAINER_NAME:-ubuntu-host-mapped}"

readonly MPI_HOSTFILE="${MPI_HOSTFILE:-/etc/mpirun/hostfile}"
readonly SSH_AUTH_SOCK_PATH="${SSH_AUTH_SOCK_PATH:-/var/run/mpirun/id_rsa_multihost_ssh_agent_sock}"

readonly MPIRUN_BTL_TCP_IF_EXCLUDE="${MPIRUN_BTL_TCP_IF_EXCLUDE:-docker0,lo}"
readonly MPIRUN_PLM_RSH_AGENT="${MPIRUN_PLM_RSH_AGENT:-ssh -A -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -l ubuntu}"

run_multihost_container() {
  docker run --rm -d \
    --name "${MULTIHOST_CONTAINER_NAME}" \
    --privileged --pid=host --network=host \
    --device /dev/tenstorrent \
    -v /dev/hugepages:/dev/hugepages \
    -v /dev/hugepages-1G:/dev/hugepages-1G \
    -v /etc/udev/rules.d:/etc/udev/rules.d \
    -v /lib/modules:/lib/modules \
    -v /opt/tt_metal_infra/provisioning/provisioning_env:/opt/tt_metal_infra/provisioning/provisioning_env \
    -v /mnt/dockercache:/mnt/dockercache \
    -v /etc/mpirun:/etc/mpirun:ro \
    -v "${SSH_AUTH_SOCK_PATH}:${SSH_AUTH_SOCK_PATH}" \
    -e "SSH_AUTH_SOCK=${SSH_AUTH_SOCK_PATH}" \
    -v "${HOST_WS}:${CTR_WS}" \
    -w "${CTR_WS}" \
    "${IMAGE}" sleep infinity
}

run_worker_container() {
  if docker ps --filter "name=${MULTIHOST_CONTAINER_NAME}" --format '{{.Names}}' | grep -q "${MULTIHOST_CONTAINER_NAME}"; then
    echo "$(hostname): Container already running, skipping"
    return 0
  fi
  run_multihost_container
  echo "$(hostname): Container started"
}

start_workers() {
  local setup_script inner

  setup_script="${HOST_WS}/scripts/multihost_container_setup/setup_multihost_containers.sh"
  inner="$(printf 'bash %q run-worker-container %q %q %q' "${setup_script}" "${IMAGE}" "${HOST_WS}" "${CTR_WS}")"

  docker exec -i "${MULTIHOST_CONTAINER_NAME}" bash -s <<MULTIHOST_SETUP_EOF
set -euo pipefail
mpirun --allow-run-as-root --hostfile ${MPI_HOSTFILE} \\
  --mca btl_tcp_if_exclude ${MPIRUN_BTL_TCP_IF_EXCLUDE} \\
  --mca plm_rsh_agent $(printf %q "${MPIRUN_PLM_RSH_AGENT}") \\
  --tag-output \\
  bash -c $(printf %q "${inner}")
MULTIHOST_SETUP_EOF

  echo "All multihost containers ready"
}

[[ $# -eq 4 ]] || exit 1

IMAGE="$2"
HOST_WS="$3"
CTR_WS="$4"

case "$1" in
start-controller)
  run_multihost_container
  ;;
start-workers)
  start_workers
  ;;
run-worker-container)
  run_worker_container
  ;;
*)
  exit 1
  ;;
esac
