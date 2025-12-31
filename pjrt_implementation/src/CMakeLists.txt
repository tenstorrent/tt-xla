# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# This file incorporates work covered by the following copyright and permission notice:
# SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# https://llvm.org/LICENSE.txt

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(api)
add_subdirectory(utils)

include(ExternalProject)

# ----- TTPJRTBindings -----
# Static lib that implements api bindings logic.
# Depends on: TTPJRTApi.

add_library(TTPJRTBindings
    "api_bindings.cc"
)

target_link_libraries(TTPJRTBindings PUBLIC
    TTPJRTApi
)

# ----- TTPJRTTTDylib -----
# Final dynamic library plugin `pjrt_plugin_tt.so`.
# Depends on: TTPJRTBindings, coverage_config

add_library(TTPJRTTTDylib SHARED
    "dylib_entry_point.cc"
)

target_link_libraries(TTPJRTTTDylib PUBLIC
    TTPJRTBindings
    coverage_config
)

# Don't link _XLAC.so directly - it contains LLVM symbols that conflict with libTTMLIRCompiler.so
# Instead, use dynamic loading (dlopen/dlsym) at runtime to avoid symbol conflicts
# Allow undefined symbols - they will be resolved via dlopen/dlsym at runtime
target_link_options(TTPJRTTTDylib PRIVATE "-Wl,--allow-shlib-undefined")

# Setting RPATH to a path involving $ORIGIN (resolved location of the pjrt_plugin_tt.so
# file) sets its search paths in which it will search for libs it dynamically links.
# Include venv site-packages path for _XLAC.so
set_target_properties(TTPJRTTTDylib PROPERTIES
    PREFIX "" # Disable "lib" prefix.
    LIBRARY_OUTPUT_NAME pjrt_plugin_tt
    BUILD_RPATH "$ORIGIN:$ORIGIN/lib:$ORIGIN/../../venv/lib/python3.11/site-packages"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/lib:$ORIGIN/../../venv/lib/python3.11/site-packages"
)

install(TARGETS TTPJRTTTDylib DESTINATION ${CMAKE_INSTALL_PREFIX})

if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    # Provide options for the build type
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Print the current build type
message(STATUS "Current build type: ${CMAKE_BUILD_TYPE}")
# Set compiler flags based on the build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(TTPJRTTTDylib PRIVATE DEBUG)
    target_compile_options(TTPJRTTTDylib PRIVATE -g -O0)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_definitions(TTPJRTTTDylib PRIVATE NDEBUG)
    target_compile_options(TTPJRTTTDylib PRIVATE -O3)
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_definitions(TTPJRTTTDylib PRIVATE NDEBUG)
    target_compile_options(TTPJRTTTDylib PRIVATE -O2 -g)
elseif(CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
    target_compile_definitions(TTPJRTTTDylib PRIVATE NDEBUG)
    target_compile_options(TTPJRTTTDylib PRIVATE -Os)
endif()

add_custom_target(install-dylib-editable ALL
    COMMAND ln -sf $<TARGET_FILE:TTPJRTTTDylib> ${CMAKE_SOURCE_DIR}/python_package/pjrt_plugin_tt/pjrt_plugin_tt.so
    COMMENT "Installing pjrt_plugin_tt.so to python_package/pjrt_plugin_tt"
    COMMAND pip install -e ${CMAKE_SOURCE_DIR}/python_package --no-deps --no-build-isolation
    COMMENT "Running editable installation..."
    USES_TERMINAL
)

add_dependencies(install-dylib-editable TTPJRTTTDylib)
