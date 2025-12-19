ARG MLIR_TAG=latest

FROM ghcr.io/tenstorrent/tt-mlir/tt-mlir-base-ubuntu-22-04:${MLIR_TAG} as base
SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install requirements.txt as global Python 3.11 packages
COPY python_package/requirements.txt /tmp/requirements.txt
COPY venv/requirements-dev.txt /tmp/requirements-dev.txt
COPY --chown=root:root --chmod=777 .github/docker_install.sh /tmp/script.sh
RUN pkg_install() { apt-get update && apt-get install -y "$@"; } && export -f pkg_install && /tmp/script.sh && \
    rm /tmp/requirements.txt /tmp/requirements-dev.txt /tmp/script.sh

FROM ghcr.io/tenstorrent/tt-mlir/tt-mlir-manylinux-2-34:${MLIR_TAG} AS ci
SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install requirements.txt as global Python 3.11 packages
COPY python_package/requirements.txt /tmp/requirements.txt
COPY venv/requirements-dev.txt /tmp/requirements-dev.txt
COPY --chown=root:root --chmod=777 .github/docker_install.sh /tmp/script.sh
RUN dnf check-update
RUN dnf install --refresh -y g++-12 git-lfs
RUN dnf install --refresh -y g++-12 patchelf
RUN dnf install --refresh -y g++-12 protobuf-compiler
RUN dnf install --refresh -y g++-12 libprotobuf-dev
RUN dnf install --refresh -y g++-12 xxd
RUN dnf install --refresh -y g++-12 ffmpeg

# RUN pkg_install() { dnf check-update && dnf install --refresh -y "$@"; } && export -f pkg_install && /tmp/script.sh && \
#     rm /tmp/requirements.txt /tmp/requirements-dev.txt /tmp/script.sh

FROM ghcr.io/tenstorrent/tt-mlir/tt-mlir-ird-ubuntu-22-04:${MLIR_TAG} AS ird
SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install requirements.txt as global Python 3.11 packages
COPY python_package/requirements.txt /tmp/requirements.txt
COPY venv/requirements-dev.txt /tmp/requirements-dev.txt
COPY --chown=root:root --chmod=777 .github/docker_install.sh /tmp/script.sh
RUN pkg_install() { apt-get update && apt-get install -y "$@"; } && export -f pkg_install && /tmp/script.sh && \
    rm /tmp/requirements.txt /tmp/requirements-dev.txt /tmp/script.sh

# Add entrypoint script
COPY .github/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
