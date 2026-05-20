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
# Controller uses 51821 (published via Docker -p 51821:51821/udp).
# Workers use 51820 (on host network, default WireGuard port).
readonly WG_CTRL_PORT="51821"
readonly WG_WORKER_PORT="51820"

# Discover the controller's physical hostname via the Docker socket.
# GitHub Actions bind-mounts /var/run/docker.sock into every container.
# `docker info .Name` returns the HOST's hostname (e.g. "f10cs02").
CONTROLLER_PHYSICAL_HOST=$(python3 -c "
import socket, json, sys
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect('/var/run/docker.sock')
s.sendall(b'GET /info HTTP/1.0\r\nHost: localhost\r\n\r\n')
d = b''
while True:
    c = s.recv(4096)
    if not c: break
    d += c
s.close()
print(json.loads(d.split(b'\r\n\r\n', 1)[1])['Name'])
")
echo "Controller physical hostname: ${CONTROLLER_PHYSICAL_HOST}"

# Container hostname (Docker container ID) — used for /etc/hosts on workers.
CONTROLLER_CONTAINER_HOSTNAME=$(hostname)

WORKER_HOSTFILE="/tmp/mpirun-workers-hostfile"

# Build the filtered worker-only hostfile (exclude the controller).
# Use the PHYSICAL hostname (not the container ID) for matching.
awk -v controller="${CONTROLLER_PHYSICAL_HOST}" '
  /^[[:space:]]*#/ {next}
  NF == 0 {next}
  {
    host=$1; short=host
    sub(/\..*$/, "", short)
    if (host != controller && short != controller) print
  }
' "${HOSTFILE}" > "${WORKER_HOSTFILE}"

if [[ ! -s "${WORKER_HOSTFILE}" ]]; then
  echo "No worker hosts found after filtering controller (${CONTROLLER_PHYSICAL_HOST}), skipping"
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
  # Kill stale wireguard-go processes from previous runs.  With --pid=host
  # --network=host, wireguard-go daemonizes into the host PID namespace and
  # survives docker stop, leaving port 51820 bound.
  pkill -f 'wireguard-go' 2>/dev/null || true
  ip link del wg0 2>/dev/null || true
  docker run --rm -d \\
    --name "\${CONTAINER_NAME}" \\
    --pid=host --network=host \\
    --cap-add NET_ADMIN \\
    --device /dev/net/tun \\
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

# Kill any stale wireguard-go on the controller container from a previous run.
pkill -f 'wireguard-go' 2>/dev/null || true
ip link del wg0 2>/dev/null || true

# Start a userspace WireGuard daemon (wireguard-go). It only requires
# CAP_NET_ADMIN and /dev/net/tun — no kernel module or CAP_SYS_MODULE needed.
# wireguard-go forks into the background and creates the wg0 TUN interface.
wireguard-go wg0
echo "${CTRL_PRIVKEY}" | wg set wg0 private-key /dev/stdin listen-port "${WG_CTRL_PORT}"
ip addr replace "${WG_CTRL_IP}/24" dev wg0
ip link set up dev wg0

# Add each worker as a WireGuard peer on the controller.
for i in "${!WORKER_HOSTS[@]}"; do
  host="${WORKER_HOSTS[$i]}"
  wg_ip="${WORKER_WG_IPS[$i]}"
  pubkey="${WORKER_PUBKEYS[$i]}"
  wg set wg0 peer "${pubkey}" \
    allowed-ips "${wg_ip}/32" \
    endpoint "${host}:${WG_WORKER_PORT}" \
    persistent-keepalive 25
  echo "Controller: added WireGuard peer ${wg_ip} (${host})"
done

# ---------------------------------------------------------------------------
# Step 4 – configure WireGuard inside each worker container
# Write per-worker setup scripts locally, then stream them to the remote host
# via SSH stdin.  The remote shell saves them to a host-side temp file and
# feeds that file to `docker exec -i bash` (host-side redirect, reliable).
# ---------------------------------------------------------------------------
wg_pids=()
WG_SCRIPT_DIR="$(mktemp -d)"

for i in "${!WORKER_HOSTS[@]}"; do
  host="${WORKER_HOSTS[$i]}"
  wg_ip="${WORKER_WG_IPS[$i]}"
  privkey="${WORKER_PRIVKEYS[$i]}"

  SETUP_SCRIPT="${WG_SCRIPT_DIR}/wg-setup-worker-${i}.sh"

  cat > "${SETUP_SCRIPT}" << EOF_SCRIPT
#!/bin/bash
set -euo pipefail

# Map the controller's Docker hostname to its WireGuard IP so that PRTE's
# remote daemons can resolve the HNP node name (which is the container ID,
# e.g. "cde67301ec05") and connect back via the WireGuard tunnel.
echo "${WG_CTRL_IP} ${CONTROLLER_CONTAINER_HOSTNAME}" >> /etc/hosts

wireguard-go wg0
printf '%s' '${privkey}' | wg set wg0 private-key /dev/stdin listen-port ${WG_WORKER_PORT}
# Explicit endpoint to the controller via its published WireGuard port.
# Workers can always reach the controller host at CONTROLLER_PHYSICAL_HOST:51821
# because Docker publishes that UDP port from the controller container.
wg set wg0 peer ${CTRL_PUBKEY} allowed-ips ${WG_CTRL_IP}/32 endpoint ${CONTROLLER_PHYSICAL_HOST}:${WG_CTRL_PORT} persistent-keepalive 25
EOF_SCRIPT

  for j in "${!WORKER_HOSTS[@]}"; do
    [[ ${j} -eq ${i} ]] && continue
    echo "wg set wg0 peer ${WORKER_PUBKEYS[$j]} allowed-ips ${WORKER_WG_IPS[$j]}/32 endpoint ${WORKER_HOSTS[$j]}:${WG_WORKER_PORT} persistent-keepalive 25" >> "${SETUP_SCRIPT}"
  done

  cat >> "${SETUP_SCRIPT}" << EOF_SCRIPT
ip addr replace ${wg_ip}/24 dev wg0
ip link set up dev wg0
echo "\$(hostname): WireGuard wg0 = ${wg_ip}"
EOF_SCRIPT

  # Stream the script to the remote host: save it to a host-side temp file,
  # then feed that file to docker exec via a host-side redirect.
  # Using a host-side `< /tmp/...` avoids stdin-forwarding issues through SSH.
  ssh ${SSH_OPTS} -l ubuntu "${host}" \
    "cat > /tmp/wg-setup-${i}.sh && docker exec -i ubuntu-host-mapped bash < /tmp/wg-setup-${i}.sh; rm -f /tmp/wg-setup-${i}.sh" \
    < "${SETUP_SCRIPT}" &
  wg_pids+=("$!")
done

failed=0
for pid in "${wg_pids[@]}"; do
  wait "${pid}" || failed=$((failed + 1))
done
rm -rf "${WG_SCRIPT_DIR}"
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

# ---------------------------------------------------------------------------
# Step 6 – generate WireGuard hostfile and update /etc/hosts
# ---------------------------------------------------------------------------

# Generate a hostfile using WireGuard IPs (workers only — no controller entry).
# PRTE will use this instead of the physical hostfile.  Because the controller's
# physical hostname (f10csXX) is NOT listed, PRTE won't try to launch a daemon
# on the controller's host (where there's no "ubuntu-host-mapped" container).
WG_HOSTFILE="${CONTAINER_WORKSPACE}/wg_mpi_hostfile"
rm -f "${WG_HOSTFILE}"
for i in "${!WORKER_HOSTS[@]}"; do
  host="${WORKER_HOSTS[$i]}"
  wg_ip="${WORKER_WG_IPS[$i]}"
  # Preserve slots= and other fields from the original hostfile entry.
  rest=$(grep "^${host}" "${HOSTFILE}" | head -1 | sed "s/^${host}//")
  echo "${wg_ip}${rest}" >> "${WG_HOSTFILE}"
done
echo "Generated WireGuard hostfile: ${WG_HOSTFILE}"
cat "${WG_HOSTFILE}"

# Add /etc/hosts entries on the controller mapping worker hostnames → WireGuard IPs.
# This lets PRTE (and SSH from remote_docker.sh) resolve worker names via wg0.
for i in "${!WORKER_HOSTS[@]}"; do
  echo "${WORKER_WG_IPS[$i]} ${WORKER_HOSTS[$i]}" >> /etc/hosts
done

echo "WireGuard overlay network ready. Controller IP: ${WG_CTRL_IP}"
