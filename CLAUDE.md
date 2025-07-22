# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

tt-xla is a PJRT plugin that enables running JAX models on Tenstorrent AI hardware. It serves as a bridge between the JAX ecosystem and Tenstorrent's ML accelerators using the PJRT (Portable JAX Runtime) interface. The project ingests JAX models via JIT compilation and provides StableHLO graphs to the tt-mlir compiler.

## Build System & Environment Setup

This project uses CMake with Ninja as the build system and requires a specific environment setup:

### Prerequisites
- Requires tt-mlir toolchain to be built first
- Environment variable `TTMLIR_TOOLCHAIN_DIR` must point to tt-mlir toolchain directory
- Uses clang-17 as compiler (set in CMakeLists.txt)
- Python 3.10 required

### Environment Activation
Always run `source venv/activate` before executing any other commands. The build system checks for `TTXLA_ENV_ACTIVATED` environment variable.

### Build Commands
```bash
# Configure build
cmake -G Ninja -B build  # Add -DCMAKE_BUILD_TYPE=Debug for debug build

# Build the project
cmake --build build

# Build wheel package
cd python_package
python setup.py bdist_wheel
```

### Testing
```bash
# Run all tests
pytest -v tests

# Run individual test with verbose output
pytest -svv path/to/test.py

# Tests are organized by markers in pytest.ini:
# - push: PR pipeline tests
# - nightly: nightly pipeline tests  
# - model_test: separate job in nightly tests
```

### Development Tools
```bash
# Install pre-commit hooks
pre-commit install

# Run linting on all files
pre-commit run --all-files
```

## Architecture Overview

### Core Components Structure
```
src/
├── common/           # Common PJRT implementation
│   └── pjrt_implementation/  # PJRT C API bindings
├── tt/              # TT-specific client implementation
└── CMakeLists.txt
```

### Key Classes and Dependencies
- `ClientInstance`: Base PJRT client implementation in src/common/pjrt_implementation/
- `TTClientInstance`: TT-specific client in src/tt/client.h
- `ModuleBuilder`: Handles compilation pipeline in src/common/module_builder.h
- Plugin architecture produces `pjrt_plugin_tt.so` dynamic library

### Dependency Hierarchy (from CMakeLists.txt)
```
pjrt_plugin_tt.so (final JAX plugin)
├── TTPJRTTT
│   └── TTPJRTCommon
│       ├── TTPJRTCommonDylibPlatform
│       ├── TTMLIRCompiler (from tt-mlir)
│       ├── TTMLIRRuntime (from tt-mlir)
│       └── loguru
└── coverage_config
```

## Test Organization

### Test Structure
- `tests/jax/single_chip/`: Single-chip JAX tests
  - `ops/`: Individual operation tests
  - `models/`: Full model tests (BERT, GPT-2, etc.)
  - `graphs/`: Graph-level tests
- `tests/jax/multi_chip/`: Multi-chip and distributed tests
- `tests/torch/`: PyTorch tests
- `tests/infra/`: Test infrastructure and utilities

### Test Infrastructure
- `tests/infra/comparators/`: Compare outputs between frameworks
- `tests/infra/connectors/`: Device connection abstractions
- `tests/infra/runners/`: Test execution engines
- `tests/infra/testers/`: Test orchestration classes

## Package Distribution

The wheel package structure bundles everything needed for JAX plugin registration:
```
jax_plugins/pjrt_plugin_tt/
├── __init__.py              # Plugin registration
├── pjrt_plugin_tt.so       # Compiled plugin
└── tt-mlir/install/        # Complete tt-mlir installation
```

## Development Workflow

1. Ensure tt-mlir toolchain is built and `TTMLIR_TOOLCHAIN_DIR` is set
2. Activate environment: `source venv/activate`
3. Configure: `cmake -G Ninja -B build`
4. Build: `cmake --build build`
5. Test: `pytest -v tests`
6. For wheel: `cd python_package && python setup.py bdist_wheel`

## Debug Configuration

For debug builds:
- Use `-DCMAKE_BUILD_TYPE=Debug` in cmake configure step
- Set `export LOGGER_LEVEL=DEBUG` for debug logs
- Debug builds enable `LOGURU_DEBUG_LOGGING=1`