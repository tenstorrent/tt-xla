#!/bin/bash

# MPI passes the hostname as the first argument
HOST=$1
shift

# The rest of the arguments are the command and its flags
COMMAND="$@"

# Log for debugging (highly recommended right now)
echo "$(date): SSH to $HOST running: docker exec jzx-host-mapped $COMMAND"

# Execute
ssh -A -l jameszianxu "$HOST" sudo docker exec \
  -u root \
  -e LD_LIBRARY_PATH=/opt/ttmlir-toolchain/lib:/lib/x86_64-linux-gnu \
  jzx-host-mapped $COMMAND