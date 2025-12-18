# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# This file incorporates work covered by the following copyright and permission notice:
# SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# https://llvm.org/LICENSE.txt

include(FetchContent)

# TODO(acicovic): Investigate why this is needed here and in /third_party/.
if(CMAKE_C_COMPILER MATCHES "clang-17")
    set(CMAKE_C_COMPILER "/usr/bin/clang-17" CACHE STRING "" FORCE)
endif()
if(CMAKE_CXX_COMPILER MATCHES "clang\\+\\+-17")
    set(CMAKE_CXX_COMPILER "/usr/bin/clang++-17" CACHE STRING "" FORCE)
endif()

# DEVNOTE: googletest is a CMake project with well-exposed targets and a
# dependency needed only for testing, so it will not be added to this repository
# as an external source tree in /third_party.
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
    GIT_SHALLOW ON
)

FetchContent_MakeAvailable(googletest)

add_executable(TTPJRTTests test_main.cc)

# Gather all test sources (various unit test and mock test directories.)

get_target_property(API_UNIT_TESTS_DIR TTPJRTApi UNIT_TESTS_DIR)
if(API_UNIT_TESTS_DIR)
    file(GLOB API_UNIT_TESTS_SRC CONFIGURE_DEPENDS ${API_UNIT_TESTS_DIR}/*.cc)
    target_sources(TTPJRTTests
        PRIVATE
            ${API_UNIT_TESTS_SRC}
    )
    # TODO(acicovic): Can we avoid enforcing a directory name for include files?
    target_include_directories(TTPJRTTests
        PRIVATE
            ${API_UNIT_TESTS_DIR}/inc
    )
endif()

get_target_property(UTILS_UNIT_TESTS_DIR TTPJRTUtils UNIT_TESTS_DIR)
if(UTILS_UNIT_TESTS_DIR)
    file(GLOB UTILS_UNIT_TESTS_SRC CONFIGURE_DEPENDS ${UTILS_UNIT_TESTS_DIR}/*.cc)
    target_sources(TTPJRTTests
        PRIVATE
            ${UTILS_UNIT_TESTS_SRC}
    )
endif()

# Link with gtest and TT's PJRT libraries.
target_link_libraries(TTPJRTTests
    PRIVATE
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
    PUBLIC
        TTPJRTApi
        TTPJRTUtils
)

# Register a test invocation command.
add_test(
    NAME
        PJRTUnitTests
    COMMAND
        TTPJRTTests --gtest_output=xml:${CMAKE_BINARY_DIR}/gtest_report.xml
)
