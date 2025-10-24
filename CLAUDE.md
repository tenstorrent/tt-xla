# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TT-XLA is a PJRT-based backend integration that enables JAX and PyTorch/XLA to run on Tenstorrent AI hardware. It leverages the PJRT (Portable JAX Runtime) interface to connect these frameworks, TT-MLIR compiler, and Tenstorrent hardware, providing StableHLO graphs to the TT-MLIR compiler.

## Architecture

### Core Components
- **PJRT Plugin (`pjrt_plugin_tt.so`)**: Main dynamic library that JAX imports as a plugin
- **TT-MLIR Integration**: Depends on TT-MLIR toolchain for compilation (environment variable `TTMLIR_TOOLCHAIN_DIR` required)
- **Multi-framework Support**:
  - `jax_plugin_tt/`: JAX wrapper package that registers the TT PJRT plugin
  - `torch_plugin_tt/`: PyTorch/XLA wrapper package with TTPlugin class that enables torch.compile with "tt" backend
  - `pjrt_plugin_tt/`: Shared PJRT plugin package containing the core binary
  - `tt_torch/`: PyTorch backend integration that registers the "tt" backend for torch.compile

### Build System
- **CMake-based**: Uses CMake with Ninja generator
- **Python wheel packaging**: Custom `setup.py` with `CMakeBuildPy` class that builds CMake project during wheel creation
- **Dependency hierarchy** (see CMakeLists.txt:10-23):
  ```
  pjrt_plugin_tt.so (TTPJRTTTDylib)
  └── TTPJRTTT
      └── TTPJRTCommon
          ├── TTPJRTCommonDylibPlatform
          ├── TTMLIRCompiler (from tt-mlir)
          ├── TTMLIRRuntime (from tt-mlir)
          └── loguru
  ```

## Common Development Commands

### Building from Source
```bash
# Ensure TT-XLA environment is activated
source venv/activate

# Build (requires TTMLIR_TOOLCHAIN_DIR environment variable)
cmake -G Ninja -B build  # Add -DCMAKE_BUILD_TYPE=Debug for debug build
cmake --build build
```

### Testing
```bash
# JAX tests
pytest -v tests/jax/single_chip          # Single-chip JAX tests
pytest -v tests/jax/multi_chip/n300      # Multi-chip JAX tests (n300 card)

# PyTorch tests
pytest -v tests/torch/single_chip        # Single-chip PyTorch tests

# Individual test with verbose output
pytest -svv path/to/test

# Memory tracking
pytest --log-memory path/to/test
```

### Python Wheel
```bash
cd python_package
python setup.py bdist_wheel
pip install dist/pjrt_plugin_tt*.whl
```

### Code Quality
```bash
# Install and run pre-commit hooks
pre-commit install
pre-commit run --all-files

# Manual linting (uses black for Python, clang-format for C++)
black python_package/
clang-format -i src/**/*.cpp
```

## Environment Setup

### Required Environment Variables
- `TTMLIR_TOOLCHAIN_DIR`: Path to TT-MLIR toolchain directory
- `TTXLA_ENV_ACTIVATED`: Must be set (checked by CMake)

### Optional Environment Variables
- `LOGGER_LEVEL`: Set to `DEBUG` or `VERBOSE` for detailed logging
- `TTMLIR_ENABLE_PERF_TRACE`: Enable performance tracing in tt-mlir

### Testing Device Connectivity
```bash
# Verify TT-XLA JAX installation
python -c "import jax; print(jax.devices('tt'))"

# Verify PyTorch/XLA with TT plugin
python -c "import torch_xla.core.xla_model as xm; print(xm.get_xla_supported_devices('tt'))"

## Test Structure and Markers

### Test Categories (pytest markers)
- `push`: Tests for PR pipeline
- `nightly`: Nightly pipeline tests
- `model_test`: Model tests (separate job)
- `large`: Resource-intensive model tests
- `known_failure_xfail`: Known failing tests
- `not_supported_skip`: Architecture-unsupported tests

### Test Properties
Use `@pytest.mark.record_test_properties()` to tag tests with:
- `category`, `jax_op_name`, `shlo_op_name` (for op tests)
- `model_name`, `model_group`, `model_info`, `run_mode`, `parallelism`, `bringup_status`, `pcc`, `atol` (for model tests)

## Key Directories

- `src/`: C++ source code (common utilities, TT-specific PJRT implementation)
- `python_package/`: Python packages and setup configuration
- `tests/`: Test suites organized by framework (jax/, torch/) and chip count
- `third_party/`: Git submodules and external dependencies
- `docs/`: Documentation and getting started guides
- `examples/`: Usage examples

## Dependencies

### System Requirements
- Ubuntu 22.04, Python 3.11, Clang 17, GCC 11, Ninja, CMake 4.0.3
- OpenMPI, protobuf-compiler, ccache, libnuma-dev, libhwloc-dev, libboost-all-dev

### Python Requirements
- jax==0.7.1, jaxlib==0.7.1, torch==2.7.0
- torch-xla (custom build from Tenstorrent PyPI)
