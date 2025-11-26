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
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
    GIT_SHALLOW ON
)

FetchContent_MakeAvailable(googletest)

add_executable(TTPJRTTests
    test_utils.cc
)

# Link with gtest and TT's PJRT libraries.
# DEVNOTE: TTPJRTApi has a dependency on TTPJRTUtils, which will not be linked
# explicitly, but it will be covered by a test suite.
target_link_libraries(TTPJRTTests
    PRIVATE
        GTest::gtest
        GTest::gtest_main
        GTest::gmock
    PUBLIC
        TTPJRTApi
)

# Register tests with CTest.
add_test(NAME PJRT COMMAND TTPJRTTests)
