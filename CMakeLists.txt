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

set(CMAKE_C_COMPILER clang-17)
set(DCMAKE_CXX_COMPILER clang++-17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

set(CMAKE_INSTALL_PREFIX ${TTPJRT_SOURCE_DIR}/install)

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
