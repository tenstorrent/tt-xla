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

echo "Controller hostname: ${CONTROLLER_HOSTNAME}, FQDN: ${CONTROLLER_FQDN}, IPs: ${CONTROLLER_IPS}"

awk -v controller="${CONTROLLER_HOSTNAME}" \
    -v controller_fqdn="${CONTROLLER_FQDN}" \
    -v controller_ips="${CONTROLLER_IPS}" '
  BEGIN {
    # Wrap in commas so we can check exact IP tokens via index().
    controller_ips_csv = "," controller_ips ","
  }
  /^[[:space:]]*#/ {next}
  NF == 0 {next}
  {
    host=$1; short=host
    sub(/\..*$/, "", short)
    is_controller_ip = index(controller_ips_csv, "," host ",") > 0
    if (host != controller && short != controller && host != controller_fqdn && !is_controller_ip) print
  }
' "${INPUT_HOSTFILE}" > "${OUTPUT_HOSTFILE}"
