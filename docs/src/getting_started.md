# Getting Started with tt-xla
tt-xla leverages [PJRT](https://github.com/openxla/xla/tree/main/xla/pjrt/c#pjrt---uniform-device-api) to integrate JAX (and in the future other frameworks), [tt-mlir](https://github.com/tenstorrent/tt-mlir) and Tenstorrent hardware. Please see [this](https://opensource.googleblog.com/2023/05/pjrt-simplifying-ml-hardware-and-framework-integration.html) blog post for more information about PJRT project. This project is a fork of [iree-pjrt](https://github.com/stellaraccident/iree-pjrt).

> **Note:** Currently only Tenstorrent `nebula` boards are supported and `galaxy` boards are not yet supported.

## Build Process
tt-xla integration with tt-mlir compiler is still in progress. Currently tt-xla it depends on tt-mlir toolchain for build. This build flow provides an easy way to experiment with tt-xla, StableHLO, and the tt-mlir infrastructure. The build process will be updated in the future to enhance the user experience.

### tt-mlir toolchain
Before compiling tt-xla, the tt-mlir toolchain needs to be built:
- Clone [tt-mlir](https://github.com/tenstorrent/tt-mlir) repo
- Follow tt-mlir [build instructions](https://docs.tenstorrent.com/tt-mlir/getting-started.html) to setup the environment and build the toolchain

### tt-xla
Before running these commands to build tt-xla, please ensure that the environtment variable `TTMLIR_TOOLCHAIN_DIR` is set to point to the tt-mlir toolchain directory created above as part of tt-mlir environment setup (for example `export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain/`). You can also set `export LOGGER_LEVEL=DEBUG` in order to enable debug logs.

```bash
git clone git@github.com:tenstorrent/tt-xla.git
cd tt-xla
source venv/activate
cmake -G Ninja -B build # -DCMAKE_BUILD_TYPE=Debug in case you want debug build
cmake --build build
```

### Wheel build
To build a wheel run

```bash
>> cd python_package
>> python setup.py bdist_wheel
```

this will output `python_package/jax_plugins/pjrt_plugin_tt/dist/pjrt_plugin_tt*.whl` file which is self-contained and can be installed using

```bash
pip install pjrt_plugin_tt*.whl
```

Wheel has the following structure

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

It contains custom Tenstorrent PJRT plugin (an `.so` file), `__init__.py` file which holds recipe (a python function) to register PJRT plugin with `JAX` and `tt-mlir` installation dir, which is needed to
be able to dynamically link tt-mlir libs in runtime and to resolve various `tt-metal` dependencies
without which plugin won't work.

Structuring wheel/folders this way allows jax to automatically register plugin upon usage. User just needs to do

```bash
>> pip install pjrt_plugin_tt*.whl
>> python
# Python console
>>>> import jax
>>>> tt_device = jax.devices("tt") # this will trigger plugin registration.
```

## Testing
The tt-xla repo contains various tests in the **tests** directory. To run them all, please run `pytest -v tests` from the project root directory. To run an individual test, `pytest -svv` is recommended in order to capture all potential error messages down the line.

## Common Build Errors
- Building tt-xla requires `clang-17`. Please make sure that `clang-17` is installed on the system and `clang/clang++` links to the correct version of the respective tools.
- Please also see the tt-mlir [docs](https://docs.tenstorrent.com/tt-mlir/getting-started.html#common-build-errors) for common build errors.

### Pre-Commit
Pre-Commit applies a git hook to the local repository such that linting is checked and applied on every `git commit` action. Install from the root of the repository using:

```bash
source venv/activate
pre-commit install
```

If you have already committed something locally before installing the pre-commit hooks, you can run this command to "catch up" on all files:

```bash
pre-commit run --all-files
```

For more information please visit [pre-commit](https://pre-commit.com/).
