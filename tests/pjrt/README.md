// TODO(acicovic): review.

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

Enabled by default during a full `tt-xla` build:

```bash
cmake -B build -G Ninja
cmake --build build/
```

Build can be disabled during CMake configuration through the dedicated option:

```bash
cmake -B build -G Ninja -DTTXLA_ENABLE_PJRT_TESTS=OFF
```

To build or clean the test-specific target, run:

```bash
# build
cmake --build build/ --target TTPJRTTests

# clean
cd build
ninja -t clean TTPJRTTests
```

## Run Tests

Use CTest (CMake's test driver) to run all tests:

```bash
ctest --test-dir build/ -R PJRT -V
```

> TODO(acicovic): We should alias these commands somehow in venv.

## CI

TODO(acicovic).

## Not Yet Implemented

These tests don't require TT hardware, and could be built and executed locally:

- Would help speed-up PJRT PRs, including refactors.
- Would help speed-up testing / investigation of specific error situations
that are hard to trigger in integration tests.
