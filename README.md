# tt-xla
tt-xla leverages pjrt to provide a hardware and framework independent interface for compilers and runtimes and simplifies the integration of hardware with frameworks. Please see the [blog](https://opensource.googleblog.com/2023/05/pjrt-simplifying-ml-hardware-and-framework-integration.html) for more information about PJRT project. This project is a fork of [iree-pjrt](https://github.com/stellaraccident/iree-pjrt)

**Note** This project is currently only supported for `nebula` boards and does not provide support for `galaxy` boards.

## Build Process
tt-xla and stableHLO integration with tt-mlir compiler is still under progress. This build flow provides an easy way to experiment with tt-xla, stableHLO, and tt-mlir infrastructure. This build process will be updated to enhance the user experience. Before compiling, please ensure environtment variable `TTMLIR_TOOLCHAIN_DIR` is set.

### tt-mlir environment
- clone tt-mlir [repo](https://github.com/tenstorrent/tt-mlir).
- Follow tt-mlir build [instructions](https://docs.tenstorrent.com/tt-mlir/build.html) to build tt-mlir environment and install all dependencies.

### tt-xla
This `tt-xla` repo is updated to use cmake instead of bazel and made compatible with tt-mlir compiler.
```
git clone git@github.com:tenstorrent/tt-xla.git
cd tt-xla
source venv/activate
cmake -G Ninja -B build
cmake --build build
```

## Testing
tt-xla repo contains various tests in `tests` directory. To run them, please run `pytest -v tests` from project root directory.

## Common Build Errors (tt-xla repo)
- Building `tt-xla` requires `clang-17`. Please make sure that `clang-17` is installed on the system and `clang/clang++` link to correct version of respective tools.
- `tt-xla` also builds `tt-metal` and it may cause `sfpi-trisc-ncrisc-build-failure`. Please use this [fix](https://docs.tenstorrent.com/tt-mlir/build.html#sfpi-trisc-ncrisc-build-failure).

### Pre-Commit
Pre-Commit applies a git hook to the local repository such that linting is checked and applied on every `git commit` action. Install from the root of the repository using:

```bash
source venv/activate
pre-commit install
```

If you have already committed before installing the pre-commit hooks, you can run on all files to "catch up":

```bash
pre-commit run --all-files
```

For more information visit [pre-commit](https://pre-commit.com/)
