#!/usr/bin/env bash
# Filters an MPI hostfile by removing the controller host entry.

set -euo pipefail

readonly SOURCE_HOSTFILE="${1:-/etc/mpirun/hostfile}"
readonly OUTPUT_HOSTFILE="${2:-/tmp/mpirun-workers-hostfile}"
readonly CONTROLLER_HOSTNAME="${CONTROLLER_HOSTNAME:-$(hostname -s)}"

awk -v controller="${CONTROLLER_HOSTNAME}" '
  /^[[:space:]]*#/ {next}
  NF == 0 {next}
  {
    host=$1
    short=host
    sub(/\..*$/, "", short)
    if (host != controller && short != controller) {
      print
    }
  }
' "${SOURCE_HOSTFILE}" > "${OUTPUT_HOSTFILE}"
