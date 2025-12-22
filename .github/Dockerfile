ARG MLIR_TAG=latest

FROM ghcr.io/tenstorrent/tt-mlir/tt-mlir-base-ubuntu-22-04:${MLIR_TAG} as base
SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install requirements.txt as global Python 3.11 packages
COPY python_package/requirements.txt /tmp/requirements.txt
COPY venv/requirements-dev.txt /tmp/requirements-dev.txt
COPY --chown=root:root --chmod=777 .github/docker_install.sh /tmp/script.sh
RUN pkg_install() { apt-get update && apt-get install -y "$@"; } && export -f pkg_install && /tmp/script.sh ubuntu && \
    rm /tmp/requirements.txt /tmp/requirements-dev.txt /tmp/script.sh

FROM ghcr.io/tenstorrent/tt-mlir/tt-mlir-ci-ubuntu-22-04:${MLIR_TAG} as ci
SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install requirements.txt as global Python 3.11 packages
COPY python_package/requirements.txt /tmp/requirements.txt
COPY venv/requirements-dev.txt /tmp/requirements-dev.txt
COPY --chown=root:root --chmod=777 .github/docker_install.sh /tmp/script.sh
RUN pkg_install() { apt-get update && apt-get install -y "$@"; } && export -f pkg_install && /tmp/script.sh ubuntu && \
    rm /tmp/requirements.txt /tmp/requirements-dev.txt /tmp/script.sh

FROM ghcr.io/tenstorrent/tt-mlir/tt-mlir-manylinux-2-34:${MLIR_TAG} AS cibw
SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install clang 17
RUN dnf install --refresh -y clang llvm llvm-devel llvm-toolset

# Install requirements.txt as global Python 3.11 packages
COPY python_package/requirements.txt /tmp/requirements.txt
COPY venv/requirements-dev.txt /tmp/requirements-dev.txt
COPY --chown=root:root --chmod=777 .github/docker_install.sh /tmp/script.sh
RUN pkg_install() { dnf install --refresh -y "$@"; } && export -f pkg_install && /tmp/script.sh dnf && \
    rm /tmp/requirements.txt /tmp/requirements-dev.txt /tmp/script.sh

FROM ghcr.io/tenstorrent/tt-mlir/tt-mlir-ird-ubuntu-22-04:${MLIR_TAG} AS ird
SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install requirements.txt as global Python 3.11 packages
COPY python_package/requirements.txt /tmp/requirements.txt
COPY venv/requirements-dev.txt /tmp/requirements-dev.txt
COPY --chown=root:root --chmod=777 .github/docker_install.sh /tmp/script.sh
RUN pkg_install() { apt-get update && apt-get install -y "$@"; } && export -f pkg_install && /tmp/script.sh ubuntu && \
    rm /tmp/requirements.txt /tmp/requirements-dev.txt /tmp/script.sh

# Add entrypoint script
COPY .github/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
