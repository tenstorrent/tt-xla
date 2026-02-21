# Getting Started
This document walks you through how to set up TT-XLA. TT-XLA is a front end for TT-Forge that is primarily used to ingest JAX models via jit compile, providing a StableHLO (SHLO) graph to the TT-MLIR compiler. TT-XLA leverages [PJRT](https://github.com/openxla/xla/tree/main/xla/pjrt/c#pjrt---uniform-device-api) to integrate JAX, [TT-MLIR](https://github.com/tenstorrent/tt-mlir) and Tenstorrent hardware. Please see [this](https://opensource.googleblog.com/2023/05/pjrt-simplifying-ml-hardware-and-framework-integration.html) blog post for more information about PJRT project. This project started as a fork of [iree-pjrt](https://github.com/stellaraccident/iree-pjrt), but has since been refactored and diverged.

> **NOTE:** If you encounter issues, please request assistance on the
>[TT-XLA Issues](https://github.com/tenstorrent/tt-xla/issues) page.

## Prerequisites

### 1. Set Up the Hardware
- Follow the instructions for the Tenstorrent device you are using at: [Hardware Setup](https://docs.tenstorrent.com)

### 2. Install Software (choose one)
- **Option 1: Quick path:** Use TT-Installer using: [Software Installation](https://docs.tenstorrent.com/getting-started/README.html#software-installation)

- **Option 2: Manual path:** For more control, follow the [manual software dependencies installation guide.](https://docs.tenstorrent.com/getting-started/manual-software-install.html)

## TT-XLA Installation Options

- [Option 1: Installing a Wheel and Running an Example](#installing-a-wheel-and-running-an-example)

   You should choose this option if you want to run models.

- [Option 2: Using a Docker Container to Run an Example](#using-a-docker-container-to-run-an-example)

   Choose this option if you want to keep the environment for running models separate from your existing environment.

- [Option 3: Building from Source](#building-from-source)

   This option is best if you want to develop TT-XLA further. It's a more complex process you are unlikely to need if you want to stick with running a model.

---

### Installing a Wheel and Running an Example

To install a wheel and run an example model, do the following:

#### Step 1. Install the Latest Wheel:

```bash
pip install pjrt-plugin-tt --extra-index-url https://pypi.eng.aws.tenstorrent.com/
```

#### Step 2. Run a Model:

- Navigate to the section of the [TT-Forge repo that contains TT-XLA demos](https://github.com/tenstorrent/tt-forge/tree/main/demos/tt-xla)

- For this walkthrough, the [demo in the **TT-Forge** repo](https://github.com/tenstorrent/tt-forge/blob/main/demos/tt-xla/nlp/jax/gpt_demo.py) is used. In the **jax** folder, in the **requirements.txt** file, you can see that **flax** and **transformers** are necessary to run the demo. Install them:

   ```bash
   pip install flax transformers
   ```

- Download the [**gpt_demo.py** file ](https://github.com/tenstorrent/tt-forge/blob/main/demos/tt-xla/nlp/jax/gpt_demo.py ) The demo you are about to run takes a piece of text and tries to predict the next word that logically follows.

- Run the model:

   ```bash
   python gpt_demo.py
   ```

- If all goes well you should see the prompt "The capital of France is", the predicted next token, the probability it will occur, and a list of other ranked options that could follow instead.

---

### Using a Docker Container to Run an Example

This section walks through the installation steps for using a Docker container for your project.
- Prerequisite: Docker must be installed. See the [official Docker installation guide](https://docs.docker.com/engine/install/ubuntu/) if needed.

#### Step 1. Run the Docker container:

   ```bash
   docker run -it --rm \
   --device /dev/tenstorrent \
   -v /dev/hugepages-1G:/dev/hugepages-1G \
   ghcr.io/tenstorrent/tt-xla-slim:latest
   ```

   >**NOTE:** You cannot isolate devices in containers. You must pass through all devices even if you are only using one. You can do this by passing ```--device /dev/tenstorrent```. Do not try to pass ```--device /dev/tenstorrent/1``` or similar, as this type of device-in-container isolation will result in fatal errors later on during execution.

- If you want to check that it is running, open a new tab with the **Same Command** option and run the following:

   ```bash
   docker ps
   ```

#### Step 2: Running Models in Docker

- Inside your running Docker container, clone the TT-Forge repo:

   ```bash
   git clone https://github.com/tenstorrent/tt-forge.git
   ```

- Set the path for Python:

   ```bash
   export PYTHONPATH=/tt-forge:$PYTHONPATH
   ```

- Navigate into TT-Forge and run the following command:

   ```bash
   git submodule update --init --recursive
   ```

- Run a model. For this example, the **demo.py** for **opt_125m** is used. Similar to **gpt2**, this model predicts what the next word in a sentence is likely to be.  The **requirements.txt** file shows that you need to install **flax** and **transformers**:

   ```bash
   pip install flax transformers
   ```

- After completing installation, run the following:

   ```bash
   python demos/tt-xla/nlp/pytorch/opt_demo.py
   ```
- If all goes well, you should get an example prompt saying 'The capital of France is.' The prediction for the next term is listed, along with the probability it will occur. This is followed by a table of other likely choices.

---

### Building from Source

Install from source if you are a developer who wants to develop for TT-XLA.

#### Step 1: Prerequisites

- TT-XLA has the following system dependencies:
   * Ubuntu 22.04
   * Python 3.11
   * python3.11-venv
   * Clang 17
   * GCC 12
   * Ninja
   * CMake 4.0.3

- TT-XLA additionally requires the following libraries:

   ```bash
   sudo apt install protobuf-compiler libprotobuf-dev
   sudo apt install ccache
   sudo apt install libnuma-dev
   sudo apt install libhwloc-dev
   sudo apt install libboost-all-dev
   ```

#### Step 2: Building the TT-MLIR Toolchain

- Before compiling TT-XLA, the TT-MLIR toolchain needs to be built:
   - Clone the [tt-mlir](https://github.com/tenstorrent/tt-mlir) repo.
   - Follow the TT-MLIR [build instructions](https://docs.tenstorrent.com/tt-mlir/getting-started.html#setting-up-the-environment-manually) to set up the environment and build the toolchain.

- After building the toolchain, set the following environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `TTMLIR_TOOLCHAIN_DIR` | Yes | Path to TT-MLIR toolchain (e.g., `/opt/ttmlir-toolchain/`) |
| `TTXLA_LOGGER_LEVEL` | No | Set to `DEBUG` or `VERBOSE` for detailed logs |

#### Step 3: Building TT-XLA

   Make sure you are not in the TT-MLIR build directory, and you are in the location where you want TT-XLA to install.

1. Clone TT-XLA:

   ```bash
   git clone https://github.com/tenstorrent/tt-xla.git
   ```

2. Navigate into the TT-XLA folder:
   ```bash
   cd tt-xla
   ```

3. Initialize third-party submodules:

   ```bash
   git submodule update --init --recursive
   ```

4. Run the following set of commands to build TT-XLA (this will build the PJRT plugin and install it into `venv`):

   ```bash
   source venv/activate
   cmake -G Ninja -B build # -DCMAKE_BUILD_TYPE=Debug in case you want debug build
   cmake --build build
   ```

5. To verify that everything is working correctly, run the following command:

   ```bash
   python -c "import jax; print(jax.devices('tt'))"
   ```

   The command should output all available TT devices, e.g. `[TTDevice(id=0, arch=Wormhole_b0)]`

6. (optional) If you want to build the TT-XLA wheel, run the following command:

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

## Pre-commit
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

## Where to Go Next

- Try more demos in the [TT-XLA folder in the TT-Forge repo](https://github.com/tenstorrent/tt-forge/tree/main/demos/tt-xla)
- Learn about [Improving Model Performance](./performance.md)
- Explore [Code Generation](./getting_started_codegen.md) to convert models into standalone code
