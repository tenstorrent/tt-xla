# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# This file incorporates work covered by the following copyright and permission notice:
# SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# https://llvm.org/LICENSE.txt


cmake_minimum_required(VERSION 3.20)

if (NOT DEFINED ENV{TTXLA_ENV_ACTIVATED})
  message(FATAL_ERROR "tt-xla environment not activated. Please run 'source venv/activate'.")
endif()

project(TT_PJRT)
cmake_policy(SET CMP0069 NEW)
set(TTPJRT_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(TTMLIR_TOOLCHAIN_DIR $ENV{TTMLIR_TOOLCHAIN_DIR})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

if(NOT TTMLIR_BUILD_TYPE)
    set(TTMLIR_BUILD_TYPE Release)
endif()

set(CMAKE_INSTALL_PREFIX ${TTPJRT_SOURCE_DIR}/install)

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

add_subdirectory(src)
add_subdirectory(third_party)
