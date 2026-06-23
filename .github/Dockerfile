ARG MLIR_TAG=latest

#============================================================================
# Base image
#============================================================================

FROM ghcr.io/tenstorrent/tt-mlir/tt-mlir-base-ubuntu-24-04:${MLIR_TAG} as base
SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

COPY --chown=root:root --chmod=777 .github/docker_install.sh /tmp/script.sh
RUN pkg_install() { apt-get update && apt-get install -y "$@"; } && export -f pkg_install && /tmp/script.sh ubuntu && \
    rm /tmp/script.sh

COPY --chown=root:root --chmod=777 .github/scripts/install-tt-triage.sh /tmp/install-tt-triage.sh
COPY third_party/CMakeLists.txt third_party/CMakeLists.txt
RUN /tmp/install-tt-triage.sh || true

#============================================================================
# CI image
#============================================================================

FROM ghcr.io/tenstorrent/tt-mlir/tt-mlir-ci-ubuntu-24-04:${MLIR_TAG} as ci
SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install requirements.txt as global Python 3.11 packages
COPY --chown=root:root --chmod=777 .github/docker_install.sh /tmp/script.sh
RUN pkg_install() { apt-get update && apt-get install -y "$@"; } && export -f pkg_install && /tmp/script.sh ubuntu && \
    rm /tmp/script.sh

ARG OMPI_TAG=v5.0.7
ARG OMPI_PREFIX=/opt/openmpi-${OMPI_TAG}-ulfm
ENV PATH=${OMPI_PREFIX}/bin:$PATH
ENV LD_LIBRARY_PATH=${OMPI_PREFIX}/lib:$LD_LIBRARY_PATH
ENV CPATH=${OMPI_PREFIX}/include
ENV PKG_CONFIG_PATH=${OMPI_PREFIX}/lib/pkgconfig:$PKG_CONFIG_PATH

#============================================================================
# Manylinux image
#============================================================================

FROM ghcr.io/tenstorrent/tt-mlir/tt-mlir-manylinux-2-34:${MLIR_TAG} AS cibw
SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Uninstall OpenMPI from EPEL to avoid conflicts
RUN dnf remove -y epel-release openmpi openmpi-devel || true

# Install build requirements
RUN dnf install --refresh -y \
    protobuf-compiler \
    protobuf-devel \
    flex \
    unzip

# Replace patchelf with LIEF's patchelf (lief-patchelf). The stock patchelf
# (including >= 0.18) corrupts the TT libraries when rewriting RPATH during
# `auditwheel repair`. auditwheel (and our own setup.py /
# CMakeLists.txt patchelf calls) invoke whichever `patchelf` is first on PATH;
# the manylinux base ships it at /usr/local/bin/patchelf, so we shadow it there.
# The musl build is statically linked and runs regardless of the container glibc.
# LIEF 0.17.6 is the latest release at time of writing.
#
# The shadow is a wrapper rather than a symlink because auditwheel verifies
# `patchelf --version` against the regex /^patchelf\s+\d+/ and requires >= 0.14;
# lief-patchelf's own version banner would not match. The wrapper reports a
# compatible version for that probe and delegates every real call to lief-patchelf.
ARG LIEF_VERSION=0.17.6
RUN wget -qO /tmp/lief-tools.zip \
      "https://github.com/lief-project/LIEF/releases/download/${LIEF_VERSION}/lief-tools-x86_64-unknown-linux-musl.zip" && \
    unzip -q /tmp/lief-tools.zip -d /tmp/lief-tools && \
    install -m 0755 /tmp/lief-tools/bin/lief-patchelf /usr/local/bin/lief-patchelf && \
    rm -rf /tmp/lief-tools /tmp/lief-tools.zip && \
    printf '%s\n' \
      '#!/bin/sh' \
      'if [ "$1" = "--version" ]; then echo "patchelf 0.18.0 (lief-patchelf '"${LIEF_VERSION}"')"; exit 0; fi' \
      'exec /usr/local/bin/lief-patchelf "$@"' \
      > /usr/local/bin/patchelf && \
    chmod 0755 /usr/local/bin/patchelf && \
    hash -r && \
    echo "Active patchelf: $(command -v patchelf)" && patchelf --version && \
    test "$(command -v patchelf)" = /usr/local/bin/patchelf && \
    patchelf --version | grep -q lief-patchelf

# Install ccache latest version from GitHub releases
RUN CCACHE_VERSION="4.7.4" && \
    wget -qO ccache.tar.xz https://github.com/ccache/ccache/releases/download/v$CCACHE_VERSION/ccache-$CCACHE_VERSION-linux-x86_64.tar.xz && \
    mkdir ccache-temp && \
    tar xf ccache.tar.xz --strip-components=1 -C ccache-temp && \
    mv ccache-temp/ccache /usr/local/bin && \
    chmod a+x /usr/local/bin/ccache && \
    rm -rf ccache.tar.xz ccache-temp

# Set up compiler wrappers
RUN mkdir -p /usr/lib/ccache \
    && ln -sf /usr/bin/ccache /usr/lib/ccache/gcc \
    && ln -sf /usr/bin/ccache /usr/lib/ccache/g++ \
    && ln -sf /usr/bin/ccache /usr/lib/ccache/cc \
    && ln -sf /usr/bin/ccache /usr/lib/ccache/c++ \
    && ln -sf /usr/bin/ccache /usr/lib/ccache/clang \
    && ln -sf /usr/bin/ccache /usr/lib/ccache/clang++

# Build OpenMPI 5.0.7 with ULFM support from source
ARG OMPI_TAG=v5.0.7
ARG OMPI_PREFIX=/opt/openmpi-${OMPI_TAG}-ulfm
WORKDIR /tmp
RUN git clone --branch ${OMPI_TAG} --depth 1 https://github.com/open-mpi/ompi.git ompi-src && \
    cd ompi-src && \
    git submodule update --init --recursive && \
    grep -rl 'req_complete = false;' --include='*.h' --include='*.c' . | xargs sed -i 's/req_complete = false;/req_complete = REQUEST_PENDING;/g' && \
    ./autogen.pl && \
    ./configure \
        --prefix=${OMPI_PREFIX} \
        --with-ft=ulfm \
        --enable-wrapper-rpath \
        --enable-mpirun-prefix-by-default \
        --disable-mca-dso \
        --disable-dlopen && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf ompi-src

# ENV needed
ENV PATH=${OMPI_PREFIX}/bin:$PATH
ENV LD_LIBRARY_PATH=${OMPI_PREFIX}/lib:$LD_LIBRARY_PATH
ENV CPATH=${OMPI_PREFIX}/include
ENV PKG_CONFIG_PATH=${OMPI_PREFIX}/lib/pkgconfig:$PKG_CONFIG_PATH
ENV CC=clang
ENV CXX=clang++
ENV TRACY_NO_INVARIANT_CHECK=1
ENV TRACY_NO_ISA_EXTENSIONS=1

ENV CCACHE_MAXSIZE="5G"
ENV CCACHE_COMPRESS="true"

#============================================================================
# IRD image
#============================================================================

FROM ghcr.io/tenstorrent/tt-mlir/tt-mlir-ird-ubuntu-24-04:${MLIR_TAG} AS ird
SHELL ["/bin/bash", "-c"]

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

COPY --chown=root:root --chmod=777 .github/docker_install.sh /tmp/script.sh
RUN pkg_install() { apt-get update && apt-get install -y "$@"; } && export -f pkg_install && /tmp/script.sh ubuntu && \
    rm /tmp/script.sh

# Add entrypoint script
COPY .github/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
