#!/bin/bash
# Reference from metal: https://github.com/tenstorrent/tt-metal/commit/8c8b385e58a4a262dd98
# To be used for mca option plm_rsh_agent script path

# MPI passes the hostname as the first argument
HOST=$1
shift

# Capture the entire remaining command as one block
REMOTE_COMMAND="$*"

# SSH Options:
# StrictHostKeyChecking=no: Don't ask to verify the host
# UserKnownHostsFile=/dev/null: Don't write to or check the global known_hosts file
# LogLevel=ERROR: Suppress the "Warning: Permanently added..." message for a cleaner MPI output
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# Use bash -c inside docker exec to handle the complex MPI environment string
ssh -A $SSH_OPTS -l ubuntu "$HOST" sudo docker exec \
  -u root \
  -e LD_LIBRARY_PATH=/opt/ttmlir-toolchain/lib:/lib/x86_64-linux-gnu \
  jzx-host-mapped bash -c "'$REMOTE_COMMAND'"
