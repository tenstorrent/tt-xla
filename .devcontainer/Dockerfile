FROM ghcr.io/tenstorrent/tt-xla/tt-xla-ird-ubuntu-22-04:latest

ARG USERNAME
ARG USER_UID
ARG USER_GID

# Create group and user matching host UID/GID so bind-mounted files have correct ownership
RUN groupadd -g ${USER_GID} ${USERNAME} \
 && useradd -u ${USER_UID} -g ${USER_GID} -m ${USERNAME} -s /bin/bash \
 && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
 && chmod 0440 /etc/sudoers.d/$USERNAME

USER $USERNAME
