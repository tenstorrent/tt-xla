# Common Antipatterns — tt-xla

## C++ Antipatterns

### 1. Manual Memory Management

**Wrong:**
```cpp
BufferInstance *buffer = new BufferInstance(data_type, dims);
// ... later ...
delete buffer;  // Leak if exception or early return
```

**Right:**
```cpp
auto buffer = BufferInstance::createInstance(data_type, dims, num_dims, device, memory);
// Automatically cleaned up when unique_ptr goes out of scope
```

**Why**: The project exclusively uses `std::unique_ptr` with factory methods. Manual `new`/`delete` bypasses RAII and risks leaks, especially across PJRT API boundaries.

---

### 2. Using Exceptions or RTTI

**Wrong:**
```cpp
try {
    auto result = compile(graph);
} catch (const std::exception &e) {
    // ...
}

if (dynamic_cast<BufferInstance *>(ptr)) { ... }
```

**Right:**
```cpp
tt_pjrt_status status = compile(graph);
if (!tt_pjrt_status_is_ok(status)) {
    LOG_F(ERROR, "Compilation failed");
    return ErrorInstance::makeError(status);
}
```

**Why**: The project compiles with `-fno-rtti`. Exceptions cannot cross the PJRT C API boundary. All error handling uses `tt_pjrt_status` return codes.

---

### 3. Unprotected Shared Mutable State

**Wrong:**
```cpp
static int counter = 0;

void increment() {
    counter++;  // Data race
}
```

**Right:**
```cpp
static std::mutex s_mutex;
static int counter = 0;

void increment() {
    std::lock_guard<std::mutex> lock(s_mutex);
    counter++;
}
```

**Why**: PJRT callbacks can be invoked from multiple threads. See `BufferInstance::s_copy_to_host_internal_mutex` for the established pattern.

---

### 4. Copying Large Objects Instead of Moving

**Wrong:**
```cpp
std::vector<uint32_t> dimensions = getDimensions();
auto buffer = BufferInstance::createInstance(type, dimensions, ...);
```

**Right:**
```cpp
std::vector<uint32_t> dimensions = getDimensions();
auto buffer = BufferInstance::createInstance(type, std::move(dimensions), ...);
```

**Why**: Factory methods accept rvalue references for vectors and strings. Unnecessary copies waste memory and CPU, especially for large tensor metadata.

---

### 5. Silent Error Swallowing

**Wrong:**
```cpp
tt_pjrt_status status = doSomething();
// Just continue regardless
doNextThing();
```

**Right:**
```cpp
tt_pjrt_status status = doSomething();
if (!tt_pjrt_status_is_ok(status)) {
    LOG_F(ERROR, "doSomething failed with status %d", static_cast<int>(status));
    return status;
}
doNextThing();
```

**Why**: Silent failures cascade into harder-to-debug issues downstream (e.g., corrupt tensor data, segfaults in tt-mlir runtime).

---

### 6. Wrong Include Guard Format

**Wrong:**
```cpp
#pragma once
// or
#ifndef BUFFER_H
```

**Right:**
```cpp
#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_BUFFER_INSTANCE_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_BUFFER_INSTANCE_H_
// ...
#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_BUFFER_INSTANCE_H_
```

**Why**: The project convention uses full path-based include guards, not `#pragma once`. This ensures uniqueness across the build tree.

---

### 7. Logging to stderr/stdout

**Wrong:**
```cpp
std::cerr << "Error: buffer is null" << std::endl;
printf("Debug: entering function\n");
```

**Right:**
```cpp
LOG_F(ERROR, "Buffer is null");
DLOG_F(LOG_DEBUG, "Entering function");
```

**Why**: The project uses loguru for all logging. Direct stderr/stdout output bypasses log level filtering (`TTXLA_LOGGER_LEVEL`) and log formatting.

---

## Python Antipatterns

### 8. Missing Test Markers

**Wrong:**
```python
def test_add():
    result = jnp.add(jnp.array([1]), jnp.array([2]))
    assert result == jnp.array([3])
```

**Right:**
```python
@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP,
    jax_op_name="jnp.add",
    shlo_op_name="stablehlo.add",
)
def test_add():
    result = jnp.add(jnp.array([1]), jnp.array([2]))
    assert result == jnp.array([3])
```

**Why**: Unmarked tests are invisible to CI pipelines and test reporting dashboards. They won't run in `push` or `nightly` pipelines and won't show up in op/model coverage tracking.

---

### 9. Skipping Instead of xfail for Known Failures

**Wrong:**
```python
@pytest.mark.skip(reason="Fails on N300")
def test_something():
    ...
```

**Right:**
```python
@pytest.mark.known_failure_xfail
def test_something():
    ...
```

**Why**: `skip` hides the test entirely — you'll never know when it starts passing. `known_failure_xfail` runs the test and alerts you when it unexpectedly passes, enabling you to remove the marker.

---

### 10. Hardcoded Device Paths or Library Paths

**Wrong:**
```python
lib_path = "/home/user/tt-xla/build/lib/pjrt_plugin_tt.so"
```

**Right:**
```python
from pjrt_plugin_tt import wrapper
library_path = wrapper.get_library_path()
```

**Why**: Paths vary between development environments, CI, and installed wheels. Use the package's discovery mechanism.

---

### 11. Missing SPDX Header

**Wrong:**
```python
import jax
# ... code ...
```

**Right:**
```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
# ... code ...
```

**Why**: The SPDX copyright check is a pre-commit hook. PRs without it will fail CI.

---

## CMake Antipatterns

### 12. Adding Sources to Wrong Target

**Wrong:** Adding a new API implementation file but forgetting to list it in the correct CMake target.

**Right:** Add to the appropriate target in `pjrt_implementation/src/CMakeLists.txt`:
- API layer files → `TTPJRTApi`
- Utility files → `TTPJRTUtils`
- Binding layer files → `TTPJRTBindings`

**Why**: Files not listed in a CMake target won't compile. Files in the wrong target break the dependency graph and may cause link errors.

---

### 13. Using PUBLIC Include for Internal Headers

**Wrong:**
```cmake
target_include_directories(TTPJRTApi PUBLIC ${CMAKE_SOURCE_DIR}/pjrt_implementation/src/internal/)
```

**Right:**
```cmake
target_include_directories(TTPJRTApi PRIVATE ${CMAKE_SOURCE_DIR}/pjrt_implementation/src/internal/)
```

**Why**: Internal headers exposed as PUBLIC leak implementation details to downstream targets and increase coupling.
