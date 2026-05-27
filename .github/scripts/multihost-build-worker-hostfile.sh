#!/usr/bin/env bash
# Builds a worker-only hostfile by removing the controller entry.
#
# Usage:
#   multihost-build-worker-hostfile.sh [input_hostfile] [output_hostfile]

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly INPUT_HOSTFILE="${1:-/etc/mpirun/hostfile}"
readonly OUTPUT_HOSTFILE="${2:-/tmp/mpirun-workers-hostfile}"

CONTROLLER_HOSTNAME=$(hostname -s)
CONTROLLER_FQDN=$(hostname -f 2>/dev/null || true)
CONTROLLER_IPS=$(hostname -I 2>/dev/null | tr ' ' '\n' | sed '/^$/d' | paste -sd, -)

awk -v controller="${CONTROLLER_HOSTNAME}" \
    -v controller_fqdn="${CONTROLLER_FQDN}" \
    -v controller_ips="${CONTROLLER_IPS}" '
  BEGIN {
    n = split(controller_ips, ip_arr, ",")
    for (i = 1; i <= n; i++) {
      if (ip_arr[i] != "") {
        controller_ip_map[ip_arr[i]] = 1
      }
    }
  }
  /^[[:space:]]*#/ {next}
  NF == 0 {next}
  {
    host=$1; short=host
    sub(/\..*$/, "", short)
    if (
      host != controller &&
      short != controller &&
      host != controller_fqdn &&
      !(host in controller_ip_map)
    ) print
  }
' "${INPUT_HOSTFILE}" > "${OUTPUT_HOSTFILE}"
