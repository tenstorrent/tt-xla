# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# This file incorporates work covered by the following copyright and permission notice:
# SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# https://llvm.org/LICENSE.txt

# Here is the dependency graph to easily keep track of components involved:
#
# pjrt_plugin_tt.so (TTPJRTTTDylib) # pjrt_implementation/src/CMakeLists.txt
# |                                 # This is the final product of the build, a dynamic
# |                                 # lib imported by JAX as a plugin
# │
# ├── TTPJRTBindings  # pjrt_implementation/src/CMakeLists.txt
# │   └── TTPJRTApi  # pjrt_implementation/src/api/CMakeLists.txt
# │       ├── TTMLIRCompiler (tt-mlir dynamic lib, product of tt-mlir build)
# │       ├── TTMLIRRuntime (tt-mlir dynamic lib, product of tt-mlir build)
# │       └── TTPJRTUtils  # pjrt_implementation/src/utils/CMakeLists.txt
# │           └── loguru
# │
# └── coverage_config   # CMakeLists.txt

# ----- Project, cmake, compiler settings -----

cmake_minimum_required(VERSION 3.20)

if (NOT DEFINED ENV{TTXLA_ENV_ACTIVATED})
  message(FATAL_ERROR "tt-xla environment not activated. Please run 'source venv/activate'.")
endif()

project(TT_PJRT)
cmake_policy(SET CMP0069 NEW)
set(TTPJRT_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(TTMLIR_TOOLCHAIN_DIR $ENV{TTMLIR_TOOLCHAIN_DIR})

set(CMAKE_C_COMPILER clang-17)
set(CMAKE_CXX_COMPILER clang++-17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

option(TTMLIR_ENABLE_PERF_TRACE "Enable performance tracing in tt-mlir" OFF)
option(TTXLA_ENABLE_EXPLORER "Enable Explorer feature in tt-mlir" OFF)

if(TTXLA_ENABLE_EXPLORER)
    set(TTMLIR_ENABLE_PERF_TRACE ON)
endif()

# ----- coverage_config -----

add_library(coverage_config INTERFACE)
option(CODE_COVERAGE "Enable coverage reporting" OFF)
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Add required flags (GCC & LLVM/Clang)
  target_compile_options(coverage_config INTERFACE
    -O0        # no optimization
    -g         # generate debug info
    --coverage # sets all required flags
  )
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.13)
    target_link_options(coverage_config INTERFACE --coverage)
  else()
    target_link_libraries(coverage_config INTERFACE --coverage)
  endif()
endif(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")


# set(ASAN_FLAGS "-fsanitize=address -fno-omit-frame-pointer")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ASAN_FLAGS}")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_definitions(-DDEBUG)
endif()

link_directories(${TTMLIR_TOOLCHAIN_DIR}//lib)

if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    add_definitions(-DLOGURU_DEBUG_LOGGING=1)
endif()

add_subdirectory(pjrt_implementation/src)
add_subdirectory(debug_tools/src)
add_subdirectory(third_party)
add_subdirectory(docs)
