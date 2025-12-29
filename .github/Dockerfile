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
# RUN dnf install --refresh -y clang llvm llvm-devel llvm-toolset && \
#     ln -s /usr/bin/clang-17 /usr/bin/clang && \
#     ln -s /usr/bin/clang++-17 /usr/bin/clang++ && \
#     ln -s /usr/bin/clangd-17 /usr/bin/clangd

# Build and install LLVM/Clang 17 from source
# RUN dnf install --refresh -y \
#     wget \
#     cmake \
#     ninja-build \
#     gcc \
#     gcc-c++ \
#     zlib-devel \
#     libxml2-devel \
#     && cd /tmp \
#     && wget https://github.com/llvm/llvm-project/releases/download/llvmorg-17.0.6/llvm-project-17.0.6.src.tar.xz \
#     && tar -xf llvm-project-17.0.6.src.tar.xz \
#     && cd llvm-project-17.0.6.src \
#     && mkdir build && cd build \
#     && cmake -G Ninja ../llvm \
#         -DCMAKE_BUILD_TYPE=Release \
#         -DCMAKE_INSTALL_PREFIX=/usr/local \
#         -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;lld" \
#         -DLLVM_TARGETS_TO_BUILD="X86" \
#         -DLLVM_ENABLE_ASSERTIONS=OFF \
#         -DLLVM_INCLUDE_TESTS=OFF \
#         -DLLVM_INCLUDE_EXAMPLES=OFF \
#         -DLLVM_INCLUDE_BENCHMARKS=OFF \
#         -DLLVM_ENABLE_BINDINGS=OFF \
#     && ninja -j32 \
#     && ninja install \
#     && cd /tmp \
#     && rm -rf llvm-project-17.0.6.src*

ENV CC=clang
ENV CXX=clang++
ENV CMAKE_C_COMPILER=clang
ENV CMAKE_CXX_COMPILER=clang++

# Install ccache for build caching and set up compiler wrappers
# RUN dnf install --refresh -y ccache \
#     && mkdir -p /usr/lib/ccache \
#     && ln -sf /usr/bin/ccache /usr/lib/ccache/gcc \
#     && ln -sf /usr/bin/ccache /usr/lib/ccache/g++ \
#     && ln -sf /usr/bin/ccache /usr/lib/ccache/cc \
#     && ln -sf /usr/bin/ccache /usr/lib/ccache/c++ \
#     && ln -sf /usr/bin/ccache /usr/lib/ccache/clang \
#     && ln -sf /usr/bin/ccache /usr/lib/ccache/clang++

# ENV PATH="/usr/lib/ccache:${PATH}"
# ENV CCACHE_DIR="/root/.ccache"
# ENV CCACHE_MAXSIZE="5G"
# ENV CCACHE_COMPRESS="true"

# Install requirements.txt as global Python 3.11 packages
# RUN dnf install --refresh -y \
#     git-lfs \
#     patchelf \
#     protobuf-compiler \
    # protobuf-devel \
    # ffmpeg-free

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
