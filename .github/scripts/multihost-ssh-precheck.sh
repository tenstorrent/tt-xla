#!/usr/bin/env bash
# Multihost SSH pre-flight check.
#
# Fails fast, before any test setup, if the multihost SSH path is broken so we
# don't burn a full job only to have mpirun hang on a dead agent socket or a
# controller that cannot reach the worker. It verifies that:
#   1. the forwarded SSH agent socket is alive and has at least one key loaded,
#   2. the controller can keyless-SSH into the target worker with a bare ssh, and
#   3. the same connection works through the MPI plm_rsh_agent wrapper
#      (remote_docker.sh), which is how mpirun actually reaches the worker.
#
# Usage:
#   multihost-ssh-precheck.sh <target-host> <ssh-user> [rsh-agent]
#
# Arguments:
#   target-host  - hostname/IP of the worker to reach (e.g. f10cs04)
#   ssh-user     - user to SSH as on the worker (e.g. ubuntu)
#   rsh-agent    - optional path to the plm_rsh_agent script; defaults to
#                  $TT_DISTRIBUTED_PLM_RSH_AGENT or the repo's remote_docker.sh
#
# Optional env vars:
#   SSH_AUTH_SOCK              - forwarded SSH agent socket (checked if set)
#   TT_DISTRIBUTED_PLM_RSH_AGENT - default rsh-agent script path

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly HOST="${1:?Usage: $0 <target-host> <ssh-user> [rsh-agent]}"
readonly USER_NAME="${2:?Usage: $0 <target-host> <ssh-user> [rsh-agent]}"
readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly DEFAULT_AGENT="${REPO_ROOT}/tests/torch/multi_host/experimental/remote_docker.sh"
readonly RSH_AGENT="${3:-${TT_DISTRIBUTED_PLM_RSH_AGENT:-${DEFAULT_AGENT}}}"

# Match the ssh options the plm_rsh_agent uses so the two checks exercise the
# same transport behaviour. BatchMode=yes turns a would-be password prompt into
# an immediate failure, so a missing/expired key fails fast instead of hanging.
readonly SSH_OPTS=(
  -A
  -o BatchMode=yes
  -o ConnectTimeout=10
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o LogLevel=ERROR
)

echo "=== Multihost SSH pre-flight check ==="
echo "Controller : $(hostname -s)"
echo "Target     : ${USER_NAME}@${HOST}"
echo "RSH agent  : ${RSH_AGENT}"
echo

# ---------------------------------------------------------------------------
# 1. SSH agent socket
# ---------------------------------------------------------------------------
echo "--- [1/3] SSH agent socket ---"
if [[ -z "${SSH_AUTH_SOCK:-}" ]]; then
  echo "WARNING: SSH_AUTH_SOCK is not set; relying on on-disk keys."
elif [[ ! -S "${SSH_AUTH_SOCK}" ]]; then
  # `-S` only checks the inode TYPE (stat()); it never connects. So a failure
  # here does NOT mean the agent process crashed -- it means one of:
  #   * the path does not exist at all: the agent never created the socket, it
  #     was cleaned up, or (common in containers) /var/run/mpirun was not
  #     mounted into this container; or
  #   * the path exists but is some other file type (regular file, dir, ...),
  #     i.e. a misconfiguration.
  # A crashed-but-stale socket (file lingers after the agent dies) still has
  # socket type, so `-S` would PASS and it is the ssh-add probe below that
  # catches a dead agent behind a present socket.
  if [[ -e "${SSH_AUTH_SOCK}" ]]; then
    echo "ERROR: SSH_AUTH_SOCK=${SSH_AUTH_SOCK} exists but is not a socket" \
         "(wrong file type -- misconfiguration)." >&2
  else
    echo "ERROR: SSH_AUTH_SOCK=${SSH_AUTH_SOCK} does not exist (socket never" \
         "created, removed, or /var/run/mpirun not mounted into this container)." >&2
  fi
  exit 1
else
  # The socket file exists and IS a socket, but that alone says nothing about
  # whether an agent is actually listening on it -- ssh-add opens the socket and
  # talks to the agent, which is the real liveness probe. Exit codes:
  #   0 = agent alive, identities listed
  #   1 = agent reachable but no identities loaded
  #   2 = cannot connect to the agent (dead/stale socket: file present, no
  #       process behind it)
  if ssh-add -l >/tmp/ssh-precheck-agent-keys 2>&1; then
    echo "SSH agent socket OK; loaded identities:"
    sed 's/^/  /' /tmp/ssh-precheck-agent-keys
  else
    status=$?
    if [[ ${status} -eq 1 ]]; then
      echo "WARNING: SSH agent is reachable but has no identities loaded."
    else
      echo "ERROR: socket ${SSH_AUTH_SOCK} exists but no agent is listening on" \
           "it (dead or stale socket -- the agent process is gone)." >&2
      exit 1
    fi
  fi
fi
echo

# ---------------------------------------------------------------------------
# 2. Bare keyless SSH
# ---------------------------------------------------------------------------
echo "--- [2/3] Bare keyless SSH to ${USER_NAME}@${HOST} ---"
if ssh "${SSH_OPTS[@]}" -l "${USER_NAME}" "${HOST}" "echo connected to \$(hostname -s)"; then
  echo "Bare SSH OK."
else
  echo "ERROR: keyless SSH to ${USER_NAME}@${HOST} failed." >&2
  echo "       The controller cannot reach the worker without a password." >&2
  exit 1
fi
echo

# ---------------------------------------------------------------------------
# 3. SSH via the plm_rsh_agent wrapper
# ---------------------------------------------------------------------------
echo "--- [3/3] SSH via plm_rsh_agent (${RSH_AGENT##*/}) ---"
if [[ ! -x "${RSH_AGENT}" ]]; then
  echo "ERROR: rsh agent script not found or not executable: ${RSH_AGENT}" >&2
  exit 1
fi

# mpirun invokes the agent as: <agent> <host> <remote command...>. The agent
# wraps the command in `docker exec <container>`, so a full success also proves
# the worker container is up. During a pre-flight the container may not exist
# yet; in that case ssh still connects and only the inner docker exec fails.
# ssh reports connection/auth failures with exit code 255, while docker errors
# surface as 125/126/127 -- so a non-255 exit still confirms the SSH transport.
set +e
"${RSH_AGENT}" "${HOST}" "echo rsh-agent reached \$(hostname -s)"
agent_status=$?
set -e

if [[ ${agent_status} -eq 0 ]]; then
  echo "plm_rsh_agent SSH + docker exec OK."
elif [[ ${agent_status} -eq 255 ]]; then
  echo "ERROR: plm_rsh_agent could not SSH into ${HOST} (ssh exit 255)." >&2
  exit 1
else
  echo "plm_rsh_agent SSH transport OK (inner command exit ${agent_status};" \
       "worker container is likely not started yet -- expected pre-setup)."
fi
echo

echo "=== Multihost SSH pre-flight check passed ==="
