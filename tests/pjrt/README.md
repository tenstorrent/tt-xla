# PJRT Unit and Mock Tests

## Goals and Guidelines

#### Complement existing integration tests (PyTorch, JAX) with unit/mock tests.

- Avoid writing tests for scenarios that are well-covered by integration tests.

#### Write tests for paths not easily reproducible in client (framework) code:

- Internal logic not visible externally to the PJRT API.
- Edge cases and exception / error code handling paths.
- Critical utility functions unit tests.
- Scenarios that require controlled / mock environment (e.g. thread safety).

## Build Tests

Tests are built by default during a full `tt-xla` build. This behavior can be
disabled during CMake configuration through the dedicated option:

```bash
cmake -B build -G Ninja -DTTXLA_ENABLE_PJRT_TESTS=OFF
```

To build or clean only the test-specific target, run:

```bash
# build
cmake --build build/ --target TTPJRTTests

# clean
cd build
ninja -t clean TTPJRTTests
```

## Run and Debug Tests

Use CTest (CMake's test driver) to run all tests:

```bash
ctest --test-dir build/ -R PJRT -V
```

To run within a debugger, follow the official instructions [here](https://docs.tenstorrent.com/tt-xla/getting_started_debugging.html#debugging-pjrt-unit-tests).

## CI

PJRT unit tests are run as part of every PR's [debug build workflow](/.github/workflows/call-build-debug.yml).
