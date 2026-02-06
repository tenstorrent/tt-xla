#!/bin/bash
# reference from metal: https://github.com/tenstorrent/tt-metal/commit/8c8b385e58a4a262dd98748287ab1801f8dd7ad8#diff-f677f80355394556f316d00c2ec1e127d2e01b6df3bbfd1b3cb598d21d9f0818
# to be used for mca option plm_rsh_agent script path
# MPI passes the hostname as the first argument
HOST=$1
shift

# Capture the entire remaining command as one block
REMOTE_COMMAND="$*"

# Use bash -c inside docker exec to handle the complex MPI environment string
# "ubuntu" user is preconfigured for keyless ssh in aus cluster
ssh -A -l ubuntu "$HOST" sudo docker exec \
  -u root \
  -e LD_LIBRARY_PATH=/opt/ttmlir-toolchain/lib:/lib/x86_64-linux-gnu \
  tt-xla-ci-worker bash -c "'$REMOTE_COMMAND'"
