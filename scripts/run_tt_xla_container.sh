#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# This script is meant to be used to run a docker container on a bare metal machine.

set -e
CONTAINER_NAME="tt-xla-ird-$USER"
IMAGE_ID="ghcr.io/tenstorrent/tt-xla/tt-xla-ird-ubuntu-22-04:latest"

# Remove old container if exists
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Directory where your repos will be stored. Make sure to create your ssh key under a
# .ssh folder in this directory.
HOST_HOME_DIRECTORY="/data/$USER"

docker run -dit \
  --name "$CONTAINER_NAME" \
  -v "$HOST_HOME_DIRECTORY:/home/$USER" \
  -e "HOME=/home/$USER" \
  --device /dev/tenstorrent \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  "$IMAGE_ID" \
  sleep infinity

echo "Container '$CONTAINER_NAME' started and running in background."

# Create matching user inside container
HOST_UID=$(id -u)
HOST_GID=$(id -g)
HOST_USER=$(whoami)

docker exec "$CONTAINER_NAME" /bin/bash -c "
  if ! id $HOST_UID &>/dev/null; then
    groupadd -g $HOST_GID $HOST_USER 2>/dev/null || true
    useradd -u $HOST_UID -g $HOST_GID -m -s /bin/bash $HOST_USER 2>/dev/null || true
    echo \"$HOST_USER ALL=(ALL) NOPASSWD:ALL\" >> /etc/sudoers
  fi
"
echo "Attach with: docker exec -it --user $HOST_UID:$HOST_GID $CONTAINER_NAME /bin/bash"
