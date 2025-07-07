# Getting Started with Building from Source

This document describes how to build the TT-XLA project on your local machine. You must build from source if you want to develop for TT-XLA. If you only want to run models, please choose one of the following sets of instructions instead:
* [Installing a Wheel and Running an Example](getting_started.md) - You should choose this option if you want to run models.
* [Using a Docker Container to Run an Example](getting_started_docker.md) - Choose this option if you want to keep the environment for running models separate from your existing environment.

## Build Process
TT-XLA integration with the TT-MLIR compiler is still in progress. Currently TT-XLA depends on the TT-MLIR toolchain to build from source. This build flow provides an easy way to experiment with TT-XLA, StableHLO, and the TT-MLIR infrastructure. The build process will be updated in the future to enhance the user experience.

### TT-MLIR Toolchain
Before compiling TT-XLA, the TT-MLIR toolchain needs to be built:
- Clone [tt-mlir](https://github.com/tenstorrent/tt-mlir) repo
- Follow the TT-MLIR [build instructions](https://docs.tenstorrent.com/tt-mlir/getting-started.html) to set up the environment and build the toolchain

### TT-XLA
Before running these commands to build TT-XLA, please ensure that the environtment variable `TTMLIR_TOOLCHAIN_DIR` is set to point to the TT-MLIR toolchain directory created above as part of the TT-MLIR environment setup (for example `export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain/`). You can also set `export LOGGER_LEVEL=DEBUG` in order to enable debug logs.

```bash
git clone git@github.com:tenstorrent/tt-xla.git
cd tt-xla
source venv/activate
cmake -G Ninja -B build # -DCMAKE_BUILD_TYPE=Debug in case you want debug build
cmake --build build
```

### Wheel Build
To build a wheel run

```bash
>> cd python_package
>> python setup.py bdist_wheel
```

this will output a `python_package/dist/pjrt_plugin_tt*.whl` file which is self-contained and can be installed using:

```bash
pip install pjrt_plugin_tt*.whl
```

The wheel has the following structure:

```bash
jax_plugins
`-- pjrt_plugin_tt
    |-- __init__.py
    |-- pjrt_plugin_tt.so   # Plugin itself.
    `-- tt-mlir             # Entire tt-mlir installation folder
        `-- install
            |-- include
            |   `-- ...
            |-- lib
            |   |-- libTTMLIRCompiler.so
            |   |-- libTTMLIRRuntime.so
            |   `-- ...
            `-- tt-metal    # We need to set TT_METAL_HOME to this dir when loading plugin
                |-- runtime
                |   `-- ...
                |-- tt_metal
                |   `-- ...
                `-- ttnn
                    `-- ...
```

It contains a custom Tenstorrent PJRT plugin (an `.so` file), `__init__.py` file which holds a python function for registering the PJRT plugin with `JAX` and the `tt-mlir` installation dir. This is needed in order to dynamically link TT-MLIR libs in runtime and to resolve various `tt-metal` dependencies without which the plugin does not work.

Structuring wheel/folders this way allows JAX to automatically register the plugin upon usage. Do the following:

```bash
>> pip install pjrt_plugin_tt*.whl
>> python
# Python console
>>>> import jax
>>>> tt_device = jax.devices("tt") # this will trigger plugin registration.
```

## Testing
The TT-XLA repo contains various tests in the **tests** directory. To run them all, please run `pytest -v tests` from the project root directory. To run an individual test, `pytest -svv` is recommended in order to capture all potential error messages down the line.

## Common Build Errors
- Building TT-XLA requires `clang-17`. Please make sure that `clang-17` is installed on the system and `clang/clang++` links to the correct version of the respective tools.
- Please also see the TT-MLIR [docs](https://docs.tenstorrent.com/tt-mlir/getting-started.html#common-build-errors) for common build errors.

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
