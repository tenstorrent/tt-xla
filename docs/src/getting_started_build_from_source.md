# Getting Started with Building from Source

This document describes how to build the TT-XLA project on your local machine. You must build from source if you want to develop for TT-XLA. If you only want to run models, please choose one of the following sets of instructions instead:
* [Installing a Wheel and Running an Example](getting_started.md#installing-a-wheel-and-running-an-example) - You should choose this option if you want to run models.
* [Using a Docker Container to Run an Example](getting_started_docker.md) - Choose this option if you want to keep the environment for running models separate from your existing environment.

The following topics are covered:

* [Configuring Hardware](#configuring-hardware)
* [System Dependencies](#system-dependencies)
* [Build Process](#build-process)
* [Testing](#testing)
* [Common Build Errors](#common-build-errors)

> **NOTE:** If you encounter issues, please request assistance on the
>[TT-XLA Issues](https://github.com/tenstorrent/tt-xla/issues) page.

## Configuring Hardware
Before setup can happen, you must configure your hardware. You can skip this section if you already completed the configuration steps. Otherwise, follow the instructions on the [Getting Started page](getting_started.md#configuring-hardware).

## System Dependencies

TT-XLA has the following system dependencies:
* Ubuntu 22.04
* Python 3.11
* python3.11-venv
* Clang 17
* GCC 12
* Ninja
* CMake 4.0.3

### Installing Python
If your system already has Python installed, make sure it is Python 3.11:

```bash
python3 --version
```

If not, install Python:

```bash
sudo apt install python3.11
```

### Installing CMake 4.0.3
To install CMake 4 or higher, do the following:

1. Install CMake 4.0.3:

```bash
pip install cmake==4.0.3
```

2. Check that the correct version of CMake is installed:

```bash
cmake --version
```

If you see ```cmake version 4.0.3``` you are ready for the next section.

### Installing Clang 17
To install Clang 17, do the following:

1. Install Clang 17:

```bash
wget https://apt.llvm.org/llvm.sh
chmod u+x llvm.sh
sudo ./llvm.sh 17
sudo apt install -y libc++-17-dev libc++abi-17-dev
sudo ln -s /usr/bin/clang-17 /usr/bin/clang
sudo ln -s /usr/bin/clang++-17 /usr/bin/clang++
```

2. Check that the selected GCC candidate using Clang 17 is GCC 12:

```bash
clang -v
```

3. Look for the line that starts with: `Selected GCC installation:`. If it is something other than GCC 12, and you do not see GCC 12 listed as an option, please install GCC 12 using:

```bash
sudo apt install g++-12 # installs gcc-12 and libstdc++-12-dev as dependencies
```

### Installing Ninja
To install Ninja, do the following:

```bash
sudo apt install ninja-build
```

### Installing OpenMPI
To install OpenMPI, do the following:

```bash
sudo wget -q https://github.com/dmakoviichuk-tt/mpi-ulfm/releases/download/v5.0.7-ulfm/openmpi-ulfm_5.0.7-1_amd64.deb -O /tmp/openmpi-ulfm.deb && sudo apt install /tmp/openmpi-ulfm.deb
```

### Installing Additional Dependencies

TT-XLA additionally requires the following libraries:

```bash
sudo apt install protobuf-compiler libprotobuf-dev
sudo apt install ccache
sudo apt install libnuma-dev
sudo apt install libhwloc-dev
sudo apt install libboost-all-dev
```

## Build Process
TT-XLA integration with the TT-MLIR compiler is still in progress. Currently TT-XLA depends on the TT-MLIR toolchain to build from source. This build flow provides an easy way to experiment with TT-XLA, StableHLO, and the TT-MLIR infrastructure. The build process will be updated in the future to enhance the user experience.

### Building the TT-MLIR Toolchain
Before compiling TT-XLA, the TT-MLIR toolchain needs to be built:
- Clone the [tt-mlir](https://github.com/tenstorrent/tt-mlir) repo.
- Follow the TT-MLIR [build instructions](https://docs.tenstorrent.com/tt-mlir/getting-started.html#setting-up-the-environment-manually) to set up the environment and build the toolchain.

### Building TT-XLA
Before running these commands to build TT-XLA, please ensure that the environment variable `TTMLIR_TOOLCHAIN_DIR` is set to point to the TT-MLIR toolchain directory created above as part of the TT-MLIR environment setup (for example `export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain/`). You can also set `export LOGGER_LEVEL=DEBUG` in order to enable debug logs, or `export LOGGER_LEVEL=VERBOSE` to enable even more verbose logs like printing intermediate IR in compiler passes. To build TT-XLA do the following:

1. Make sure you are not in the TT-MLIR build directory, and you are in the location where you want TT-XLA to install.

2. Clone TT-XLA:

```bash
git clone https://github.com/tenstorrent/tt-xla.git
```

3. Navigate into the TT-XLA folder:

```bash
cd tt-xla
```

4. Initialize third-party submodules:

```bash
git submodule update --init --recursive
```

5. Run the following set of commands to build TT-XLA (this will build the PJRT plugin and install it into `venv`):

```bash
source venv/activate
cmake -G Ninja -B build # -DCMAKE_BUILD_TYPE=Debug in case you want debug build
cmake --build build
```

6. To verify that everything is working correctly, run the following command:

```bash
python -c "import jax; print(jax.devices('tt'))"
```

The command should output all available TT devices, e.g. `[TTDevice(id=0, arch=Wormhole_b0)]`

7. (optional) If you want to build the TT-XLA wheel, run the following command:

```bash
cd python_package
python setup.py bdist_wheel
```

The above command outputs a `python_package/dist/pjrt_plugin_tt*.whl` file which is self-contained. To install the created wheel, run:

```bash
pip install dist/pjrt_plugin_tt*.whl
```

The wheel has the following structure:
```plaintext
pjrt_plugin_tt/                     # PJRT plugin package
    |-- __init__.py
    |-- pjrt_plugin_tt.so               # PJRT plugin binary
    |-- tt-metal/                       # tt-metal runtime dependencies (kernels, riscv compiler/linker, etc.)
    `-- lib/                            # shared library dependencies (tt-mlir, tt-metal)
jax_plugin_tt/                      # Thin JAX wrapper
    `-- __init__.py                     # imports and sets up pjrt_plugin_tt for XLA
torch_plugin_tt                     # Thin PyTorch/XLA wrapper
    `-- __init__.py                     # imports and sets up pjrt_plugin_tt for PyTorch/XLA
```

It contains a custom Tenstorrent PJRT plugin (`pjrt_plugin_tt.so`) and its dependencies (`tt-mlir` and `tt-metal`). Additionally, there are thin wrappers for JAX (`jax_plugin_tt`) and PyTorch/XLA (`torch_plugin_tt`) that import the PJRT plugin and set it up for use with the respective frameworks.

## Testing
The TT-XLA repo contains various tests in the **tests** directory. To run an individual test, `pytest -svv` is recommended in order to capture all potential error messages down the line. Multi-chip tests can be run only on specific Tenstorrent hardware, therefore these tests are structured in folders named by the Tenstorrent cards/systems they can be run on. For example, you can run `pytest -v tests/jax/multi_chip/n300` only on a system with an n300 Tenstorrent card. Single-chip tests can be run on any system with the command `pytest -v tests/jax/single_chip`.

## Common Build Errors
- Building TT-XLA requires `clang-17`. Please make sure that `clang-17` is installed on the system and `clang/clang++` links to the correct version of the respective tools.
- Please also see the TT-MLIR [docs](https://github.com/tenstorrent/tt-mlir/blob/main/docs/src/getting-started.md#common-build-errors) for common build errors.

### Pre-commit
Pre-commit applies a git hook to the local repository such that linting is checked and applied on every `git commit` action. Install it from the root of the repository using:

```bash
source venv/activate
pre-commit install
```

If you have already committed something locally before installing the pre-commit hooks, you can run this command to check all files:

```bash
pre-commit run --all-files
```

For more information please visit [pre-commit](https://pre-commit.com/).
