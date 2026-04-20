#!/usr/bin/env bash
set -euo pipefail

readonly DOCKER_IMAGE="${1:-}"
readonly HOST_WORKSPACE_PATH="${2:-}"
readonly IN_CONTAINER_WORKSPACE="${3:-}"
readonly CONTAINER_NAME="${4:-ubuntu-host-mapped}"
readonly SSH_AUTH_SOCK_PATH="${5:-}"

if [[ -z "${DOCKER_IMAGE}" || -z "${HOST_WORKSPACE_PATH}" || -z "${IN_CONTAINER_WORKSPACE}" ]]; then
  echo "Usage: $0 <docker_image> <host_workspace_path> <in_container_workspace> [container_name] [ssh_auth_sock_path]" >&2
  exit 1
fi

if [[ "$(docker ps --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}')" == "${CONTAINER_NAME}" ]]; then
  echo "$(hostname): Container ${CONTAINER_NAME} already running, skipping"
  exit 0
fi

docker_args=(
  run --rm -d
  --name "${CONTAINER_NAME}"
  --privileged --pid=host --network=host
  --device /dev/tenstorrent
  -v /dev/hugepages:/dev/hugepages
  -v /dev/hugepages-1G:/dev/hugepages-1G
  -v /etc/udev/rules.d:/etc/udev/rules.d
  -v /lib/modules:/lib/modules
  -v /opt/tt_metal_infra/provisioning/provisioning_env:/opt/tt_metal_infra/provisioning/provisioning_env
  -v /mnt/dockercache:/mnt/dockercache
  -v /etc/mpirun:/etc/mpirun:ro
  -v "${HOST_WORKSPACE_PATH}:${IN_CONTAINER_WORKSPACE}"
  -w "${IN_CONTAINER_WORKSPACE}"
)

if [[ -n "${SSH_AUTH_SOCK_PATH}" ]]; then
  docker_args+=(
    -v "${SSH_AUTH_SOCK_PATH}:${SSH_AUTH_SOCK_PATH}"
    -e "SSH_AUTH_SOCK=${SSH_AUTH_SOCK_PATH}"
  )
fi

docker "${docker_args[@]}" "${DOCKER_IMAGE}" sleep infinity
echo "$(hostname): Container ${CONTAINER_NAME} started"
