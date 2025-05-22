[![Tests][tests badge]][tests]
[![Codecov][codecov badge]][codecov]

# tt-xla
tt-xla leverages [PJRT](https://github.com/openxla/xla/tree/main/xla/pjrt/c#pjrt---uniform-device-api) to integrate JAX (and in the future other frameworks), [tt-mlir](https://github.com/tenstorrent/tt-mlir) and Tenstorrent hardware. Please see [this](https://opensource.googleblog.com/2023/05/pjrt-simplifying-ml-hardware-and-framework-integration.html) blog post for more information about PJRT project. This project is a fork of [iree-pjrt](https://github.com/stellaraccident/iree-pjrt).

**Note:** Currently only Tenstorrent `nebula` boards are supported and `galaxy` boards are not yet supported.

## Build Process
tt-xla integration with tt-mlir compiler is still under progress and currently it depends on tt-mlir toolchain for build. This build flow provides an easy way to experiment with tt-xla, StableHLO and tt-mlir infrastructure. Build process will be updated in the future to enhance the user experience.

### tt-mlir toolchain
Before compiling tt-xla, tt-mlir toolchain needs to be built:
- Clone [tt-mlir](https://github.com/tenstorrent/tt-mlir) repo
- Follow tt-mlir [build instructions](https://docs.tenstorrent.com/tt-mlir/build.html) to setup the environment and build the toolchain

### tt-xla
Before running these commands to build tt-xla, please ensure that the environtment variable `TTMLIR_TOOLCHAIN_DIR` is set to point to the tt-mlir toolchain directory created above as part of tt-mlir environment setup (for example `export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain/`). You can also set `export LOGGER_LEVEL=DEBUG` in order to enable debug logs.

```bash
git clone git@github.com:tenstorrent/tt-xla.git
cd tt-xla
source venv/activate
cmake -G Ninja -B build # -DCMAKE_BUILD_TYPE=Debug in case you want debug build
cmake --build build
```

## Testing
tt-xla repo contains various tests in `tests` directory. To run them all, please run `pytest -v tests` from project root directory. To run individual test, `pytest -svv` is recommended in order to capture all potential error messages down the line.

## Common Build Errors
- Building `tt-xla` requires `clang-17`. Please make sure that `clang-17` is installed on the system and `clang/clang++` link to correct version of respective tools.
- `tt-xla` also builds `tt-metal` and it may cause `sfpi-trisc-ncrisc-build-failure`. Please use [this](https://docs.tenstorrent.com/tt-mlir/build.html#sfpi-trisc-ncrisc-build-failure) fix.
- Please see tt-mlir [docs](https://docs.tenstorrent.com/tt-mlir/build.html#common-build-errors) for common build errors.

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


[codecov]: https://codecov.io/gh/tenstorrent/tt-xla
[tests]: https://github.com/tenstorrent/tt-xla/actions/workflows/on-push.yml?query=branch%3Amain
[codecov badge]: https://codecov.io/gh/tenstorrent/tt-xla/graph/badge.svg?token=XQJ3JVKIRI
[tests badge]: https://github.com/tenstorrent/tt-xla/actions/workflows/on-push.yml/badge.svg?query=branch%3Amain