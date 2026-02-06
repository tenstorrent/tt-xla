#!/bin/bash

# MPI passes the hostname as the first argument
HOST=$1
shift

# Capture the entire remaining command as one block
REMOTE_COMMAND="$*"

# Use bash -c inside docker exec to handle the complex MPI environment string
ssh -A -l jameszianxu "$HOST" sudo docker exec \
  -u root \
  -e LD_LIBRARY_PATH=/opt/ttmlir-toolchain/lib:/lib/x86_64-linux-gnu \
  jzx-host-mapped bash -c "'$REMOTE_COMMAND'"