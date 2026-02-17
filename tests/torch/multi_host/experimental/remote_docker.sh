#!/bin/bash
HOST=$1
shift

# Capture the entire remaining command as one block
REMOTE_COMMAND="$*"

# Get SSH user from environment or default to ttuser
SSH_USER=${SSH_USER:-ttuser}

# SSH Options:
# StrictHostKeyChecking=no: Don't ask to verify the host
# UserKnownHostsFile=/dev/null: Don't write to or check the global known_hosts file
# LogLevel=ERROR: Suppress the "Warning: Permanently added..." message for a cleaner MPI output
SSH_CONFIG_OPT="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

# Use bash -c inside docker exec to handle the complex MPI environment string
ssh $SSH_CONFIG_OPT -l $SSH_USER "$HOST" docker exec \
  ubuntu-host-mapped bash -c "'$REMOTE_COMMAND'"
