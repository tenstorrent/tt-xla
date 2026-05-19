#!/usr/bin/env bash
# Starts multihost worker containers on remote worker hosts, then establishes
# a WireGuard overlay network (L3, inside the containers) so that PRTE/MPI
# can advertise a stable, routable IP address (10.200.0.1 for the controller)
# to all worker daemons.  This avoids the GitHub Actions Docker bridge-network
# problem where --network=host conflicts with the injected --network flag.
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
#   SSH_AUTH_SOCK        - forwarded SSH agent socket

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly WORKER_IMAGE="${1:?Usage: $0 <worker-image> [hostfile]}"
readonly HOSTFILE="${2:-/etc/mpirun/hostfile}"
readonly CONTAINER_WORKSPACE="${CONTAINER_WORKSPACE:-$(pwd)}"
readonly HOST_WORKSPACE_PATH="${HOST_WORKSPACE_PATH:?HOST_WORKSPACE_PATH must be set to the host-side workspace path}"

# WireGuard overlay network parameters.
# Controller always gets .1; workers get .2, .3, …
readonly WG_SUBNET="10.200.0"
readonly WG_CTRL_IP="${WG_SUBNET}.1"
readonly WG_PORT="51820"

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

SSH_OPTS="-A -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# ---------------------------------------------------------------------------
# Step 1 – generate WireGuard keypairs for controller + all workers
# ---------------------------------------------------------------------------

CTRL_PRIVKEY=$(wg genkey)
CTRL_PUBKEY=$(echo "${CTRL_PRIVKEY}" | wg pubkey)

declare -a WORKER_HOSTS=()
# Use indexed arrays manually to stay compatible with bash 4.x
declare -a WORKER_WG_IPS=()
declare -a WORKER_PRIVKEYS=()
declare -a WORKER_PUBKEYS=()

idx=2
while IFS= read -r line; do
  host=$(awk '{print $1}' <<< "${line}")
  [[ -z "${host}" ]] && continue
  WORKER_HOSTS+=("${host}")
  WORKER_WG_IPS+=("${WG_SUBNET}.${idx}")
  privkey=$(wg genkey)
  WORKER_PRIVKEYS+=("${privkey}")
  WORKER_PUBKEYS+=("$(echo "${privkey}" | wg pubkey)")
  idx=$((idx + 1))
done < "${WORKER_HOSTFILE}"

# ---------------------------------------------------------------------------
# Step 2 – start worker containers in parallel (plain SSH loop, same as before)
# ---------------------------------------------------------------------------
pids=()

for i in "${!WORKER_HOSTS[@]}"; do
  host="${WORKER_HOSTS[$i]}"

  ssh ${SSH_OPTS} -l ubuntu "${host}" bash << EOF &
  set -euo pipefail
  CONTAINER_NAME=ubuntu-host-mapped
  if docker ps -a --filter "name=^\${CONTAINER_NAME}\$" --format '{{.Names}}' \
     | grep -q "\${CONTAINER_NAME}"; then
    echo "\$(hostname): removing existing container \${CONTAINER_NAME}"
    docker stop "\${CONTAINER_NAME}" 2>/dev/null || true
    docker rm "\${CONTAINER_NAME}" 2>/dev/null || true
  fi
  docker run --rm -d \\
    --name "\${CONTAINER_NAME}" \\
    --pid=host --network=host \\
    --cap-add NET_ADMIN \\
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
  echo "\$(hostname): container \${CONTAINER_NAME} started"
EOF
  pids+=("$!")
done

# Wait for all SSH commands and collect failures
failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=$((failed + 1))
done
if [[ ${failed} -gt 0 ]]; then
  echo "ERROR: ${failed} worker container(s) failed to start" >&2
  exit 1
fi

echo "All worker containers started successfully"

# ---------------------------------------------------------------------------
# Step 3 – set up WireGuard on the controller (this container)
# ---------------------------------------------------------------------------

echo "Setting up WireGuard overlay network…"

# Create the interface (ignore error if it already exists from a previous run).
ip link add wg0 type wireguard 2>/dev/null || true
echo "${CTRL_PRIVKEY}" | wg set wg0 private-key /dev/stdin listen-port "${WG_PORT}"
ip addr replace "${WG_CTRL_IP}/24" dev wg0
ip link set up dev wg0

# Add each worker as a WireGuard peer on the controller.
for i in "${!WORKER_HOSTS[@]}"; do
  host="${WORKER_HOSTS[$i]}"
  wg_ip="${WORKER_WG_IPS[$i]}"
  pubkey="${WORKER_PUBKEYS[$i]}"
  wg set wg0 peer "${pubkey}" \
    allowed-ips "${wg_ip}/32" \
    endpoint "${host}:${WG_PORT}" \
    persistent-keepalive 25
  echo "Controller: added WireGuard peer ${wg_ip} (${host})"
done

# ---------------------------------------------------------------------------
# Step 4 – configure WireGuard inside each worker container via docker exec
# Workers use --network=host so wg0 is created in the HOST network namespace;
# the tunnel endpoints are the physical host IPs (DNS-resolvable hostnames).
# ---------------------------------------------------------------------------
wg_pids=()

for i in "${!WORKER_HOSTS[@]}"; do
  host="${WORKER_HOSTS[$i]}"
  wg_ip="${WORKER_WG_IPS[$i]}"
  privkey="${WORKER_PRIVKEYS[$i]}"

  # Build the peer lines for this worker: controller + all other workers.
  peer_cmds=""

  # Peer: controller
  peer_cmds+="wg set wg0 peer ${CTRL_PUBKEY} allowed-ips ${WG_CTRL_IP}/32 endpoint ${CONTROLLER_HOSTNAME}:${WG_PORT} persistent-keepalive 25; "

  # Peers: sibling workers (for direct worker-to-worker streams if needed)
  for j in "${!WORKER_HOSTS[@]}"; do
    [[ ${j} -eq ${i} ]] && continue
    peer_cmds+="wg set wg0 peer ${WORKER_PUBKEYS[$j]} allowed-ips ${WORKER_WG_IPS[$j]}/32 endpoint ${WORKER_HOSTS[$j]}:${WG_PORT} persistent-keepalive 25; "
  done

  ssh ${SSH_OPTS} -l ubuntu "${host}" \
    docker exec ubuntu-host-mapped bash -c "
      set -euo pipefail
      ip link add wg0 type wireguard 2>/dev/null || true
      echo '${privkey}' | wg set wg0 private-key /dev/stdin listen-port ${WG_PORT}
      ${peer_cmds}
      ip addr replace ${wg_ip}/24 dev wg0
      ip link set up dev wg0
      echo \"\$(hostname): WireGuard wg0 = ${wg_ip}\"
    " &
  wg_pids+=("$!")
done

failed=0
for pid in "${wg_pids[@]}"; do
  wait "${pid}" || failed=$((failed + 1))
done
if [[ ${failed} -gt 0 ]]; then
  echo "ERROR: WireGuard setup failed on ${failed} worker(s)" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Step 5 – verify reachability (brief wait for handshakes to complete)
# ---------------------------------------------------------------------------
sleep 2

all_ok=true
for i in "${!WORKER_HOSTS[@]}"; do
  wg_ip="${WORKER_WG_IPS[$i]}"
  host="${WORKER_HOSTS[$i]}"
  if ping -c 1 -W 3 "${wg_ip}" > /dev/null 2>&1; then
    echo "WireGuard peer ${wg_ip} (${host}): reachable"
  else
    echo "WARNING: WireGuard peer ${wg_ip} (${host}): NOT reachable" >&2
    all_ok=false
  fi
done

${all_ok} || echo "WARNING: some WireGuard peers are unreachable; tests may fail" >&2

echo "WireGuard overlay network ready. Controller IP: ${WG_CTRL_IP}"
