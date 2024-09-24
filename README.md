# tt-xla
tt-xla provides a hardware- and framework-independent interface for compilers and runtimes and simplifies the integration of hardware with frameworks. Please see the [blog](https://opensource.googleblog.com/2023/05/pjrt-simplifying-ml-hardware-and-framework-integration.html) for more information about PJRT project.

**Note** This project is currently only supported for `nebula` boards and does not provide support for `galaxy` boards.

## Build Process
tt-xla and stableHLO integration with tt-mlir compiler is still under progress. This build flow provides an easy way to experiment with tt-xla, stableHLO, and tt-mlir infrastructure. This build process will be updated for better user experience.

### tt-mlir environment
- clone tt-mlir [repo](https://github.com/tenstorrent/tt-mlir).
- Follow tt-mlir build [instructions](https://docs.tenstorrent.com/tt-mlir/build.html) to build tt-mlir environment and install all dependencies.

### StableHLO
Please make sure that you are running the following commands with in tt-mlir `virtual environment` (execute `source env/activate` in `tt-mlir` directory).
```
cd /opt/ttmlir-toolchain/src/stablehlo-build/
cmake ../stablehlo -G Ninja -DMLIR_DIR=${PWD}/../llvm-build/lib/cmake/mlir
cmake --build .
cp /opt/ttmlir-toolchain/src/stablehlo-build/lib/lib* /opt/ttmlir-toolchain/lib/
```

### tt-xla
This `tt-xla` repo is updated to use cmake instead of bazel and made compatiable with tt-mlir compiler.
```
git clone git@github.com:tenstorrent/tt-xla.git // Clone tt-xla repo
cd pjrt
source venv/activate
cmake -G Ninja -B build
cmake –-build build
```

## Testing
tt-xla repo contains various tests in `tests` directory. To run them, please run `pytest -v tests` from project root directory.

## Common Build Errors (tt-xla repo)
- Building `tt-xla` requires `clang-17`. Please make sure that `clang-17` is installed on the system and `clang/clang++` link to correct version of respective tools.
- `tt-xla` also builds `tt-metal` and it may cause `sfpi-trisc-ncrisc-build-failure`. Please use this [fix](https://docs.tenstorrent.com/tt-mlir/build.html#sfpi-trisc-ncrisc-build-failure).
