# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# This file incorporates work covered by the following copyright and permission notice:
# SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# https://llvm.org/LICENSE.txt

include(ExternalProject)

# ----- TTPJRTTT -----
# Static lib that implements client logic.
# Depends on: loguru, TTPJRTCommon.

add_library(TTPJRTTT
    "client.cc"
)

target_include_directories(TTPJRTTT PUBLIC
    ${PROJECT_SOURCE_DIR}/src/common
    ${PROJECT_SOURCE_DIR}/third_party/loguru/src/loguru-install/include/
)

target_link_libraries(TTPJRTTT PUBLIC
    loguru
    TTPJRTCommon
)

# ----- TTPJRTTTDylib -----
# Final dynamic library plugin `pjrt_plugin_tt.so`.
# Depends on: TTPJRTTT, coverage_config

add_library(TTPJRTTTDylib SHARED
    "dylib_entry_point.cc"
)

target_include_directories(TTPJRTTTDylib PUBLIC
    ${PROJECT_SOURCE_DIR}/third_party/pjrt_c_api
)

target_link_libraries(TTPJRTTTDylib PUBLIC
    TTPJRTTT
    coverage_config
)

target_link_options(TTPJRTTTDylib PRIVATE "-Wl,--no-undefined")

# Setting RPATH to a path involving $ORIGIN (resolved location of the pjrt_plugin_tt.so
# file) sets its search paths in which it will search for libs it dynamically links.
set_target_properties(TTPJRTTTDylib PROPERTIES
    PREFIX "" # Disable "lib" prefix.
    LIBRARY_OUTPUT_NAME pjrt_plugin_tt
    BUILD_RPATH "$ORIGIN:$ORIGIN/lib"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/lib"
)

install(TARGETS TTPJRTTTDylib DESTINATION ${CMAKE_INSTALL_PREFIX})

# Install protoc binary from tt-mlir
# Use CODE block to check at install time rather than configure time
install(CODE "
    # Check multiple possible locations for protoc at install time
    if(EXISTS \"${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/build/bin/protoc-3.21.12.0\")
        file(INSTALL \"${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/build/bin/protoc-3.21.12.0\"
             DESTINATION \"${CMAKE_INSTALL_PREFIX}/bin\"
             FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
             RENAME \"protoc\")
        message(STATUS \"Installed protoc from tt-metal build directory\")
    elseif(EXISTS \"${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/build/_deps/protobuf-build/protoc-3.21.12.0\")
        file(INSTALL \"${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/build/_deps/protobuf-build/protoc-3.21.12.0\"
             DESTINATION \"${CMAKE_INSTALL_PREFIX}/bin\"
             FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
             RENAME \"protoc\")
        message(STATUS \"Installed protoc from protobuf-build directory\")
    elseif(EXISTS \"${PROJECT_SOURCE_DIR}/third_party/tt-mlir/install/bin/protoc\")
        # Check if it's a valid file (not a broken symlink)
        execute_process(
            COMMAND test -f \"${PROJECT_SOURCE_DIR}/third_party/tt-mlir/install/bin/protoc\"
            RESULT_VARIABLE is_valid_file
            OUTPUT_QUIET
            ERROR_QUIET
        )
        if(is_valid_file EQUAL 0)
            file(INSTALL \"${PROJECT_SOURCE_DIR}/third_party/tt-mlir/install/bin/protoc\"
                 DESTINATION \"${CMAKE_INSTALL_PREFIX}/bin\"
                 FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
            message(STATUS \"Installed protoc from tt-mlir install directory\")
        else()
            message(WARNING \"protoc at ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/install/bin/protoc is a broken symlink or doesn't exist, skipping\")
        endif()
    else()
        message(WARNING \"protoc binary not found in any expected location, skipping installation\")
    endif()
")

# Install protobuf libraries from tt-mlir
# Use OPTIONAL flag to avoid errors if files don't exist
install(FILES ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/install/lib/libprotobuf.a
              ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/install/lib/libprotobuf-lite.a
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        OPTIONAL)

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
