# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

option(USE_CUSTOM_TT_MLIR_VERSION "Flag to use TT_MLIR_VERSION set by the user" OFF)

if (NOT DEFINED TT_MLIR_VERSION OR NOT USE_CUSTOM_TT_MLIR_VERSION)
    set(TT_MLIR_VERSION "9124f27e6b0bc567aa7bbd97a7bd24c1e928a772")
endif()

set(LOGURU_VERSION "4adaa185883e3c04da25913579c451d3c32cfac1")

if (TOOLCHAIN STREQUAL "ON")
    cmake_minimum_required(VERSION 3.20)
    project(ttmlir-toolchain LANGUAGES CXX C)
    execute_process(
        COMMAND git clone --recursive https://github.com/tenstorrent/tt-mlir.git ${PROJECT_SOURCE_DIR}/tt-mlir/src/tt-mlir
        RESULT_VARIABLE result1
        OUTPUT_VARIABLE output1
        ERROR_VARIABLE error_output1
    )
    if (NOT ${result1} EQUAL 0)
        message(FATAL_ERROR "Failed to clone tt-mlir: ${error_output1}")
    endif()
    execute_process(
        COMMAND /bin/bash -c "cd ${PROJECT_SOURCE_DIR}/tt-mlir/src/tt-mlir && git checkout ${TT_MLIR_VERSION}"
    )
else()
    include(ExternalProject)
    add_subdirectory(pjrt_c_api)

    # ----- tt-mlir -----

    set(TTMLIR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tt-mlir")
    set(TTMLIR_BUILD_DIR "${TTMLIR_SOURCE_DIR}/build")
    set(TTMLIR_LIB_DIR ${TTMLIR_SOURCE_DIR}/install)
    set(TT_RUNTIME_DEBUG OFF CACHE BOOL "Enable ttmlir runtime debugging")
    message(STATUS "ttmlir runtime debugging is set to: ${TT_RUNTIME_DEBUG}")

    ExternalProject_Add(
        tt-mlir
        PREFIX ${TTPJRT_SOURCE_DIR}/third_party/tt-mlir
        # Installing the python dependencies before the build and fixing cmake config
        PATCH_COMMAND TTPJRT_SOURCE_DIR=${TTPJRT_SOURCE_DIR} bash ${TTPJRT_SOURCE_DIR}/venv/install_ttmlir_requirements.sh &&
                      cp ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/TTMLIRConfig.cmake.in ${CMAKE_CURRENT_SOURCE_DIR}/tt-mlir/cmake/ &&
                      echo "set(PROJECT_VERSION \"1.0.0\")" > ${CMAKE_CURRENT_SOURCE_DIR}/tt-mlir/cmake/version_fix.cmake &&
                      patch -p1 < ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/fix_protobuf_exports.patch || true
        CMAKE_GENERATOR Ninja
        BINARY_DIR ${TTMLIR_BUILD_DIR}
        INSTALL_COMMAND ${CMAKE_COMMAND} --install ${TTMLIR_BUILD_DIR} --component SharedLib
        CMAKE_ARGS
          -DCMAKE_BUILD_TYPE=Release
          -DCMAKE_C_COMPILER=clang-17
          -DCMAKE_CXX_COMPILER=clang++-17
          -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
          -DTT_RUNTIME_ENABLE_TTNN=ON
          -DTTMLIR_ENABLE_STABLEHLO=ON
          -DTTMLIR_ENABLE_RUNTIME=ON
          -DTT_RUNTIME_DEBUG=${TT_RUNTIME_DEBUG}
          -DTTMLIR_ENABLE_BINDINGS_PYTHON=OFF
          -DCMAKE_INSTALL_PREFIX=${TTMLIR_LIB_DIR}
          -DTT_RUNTIME_ENABLE_PERF_TRACE=${TTMLIR_ENABLE_PERF_TRACE}
          -DTTMLIR_ENABLE_OPMODEL=ON
          -DTTMLIR_ENABLE_SHARED_LIB=ON
        GIT_REPOSITORY https://github.com/tenstorrent/tt-mlir.git
        GIT_TAG ${TT_MLIR_VERSION}
        GIT_PROGRESS ON
    )

    set_target_properties(tt-mlir PROPERTIES EXCLUDE_FROM_ALL TRUE)

    install(DIRECTORY ${TTMLIR_LIB_DIR}/ DESTINATION "${CMAKE_INSTALL_PREFIX}" USE_SOURCE_PERMISSIONS)

    # Expose all tt-mlir .so files as shared imported libraries.
    set(TTMLIR_LIB_DIR ${TTMLIR_LIB_DIR} PARENT_SCOPE)
    message(STATUS "Linking to TTMLIR libraries in ${TTMLIR_LIB_DIR}")
    file(GLOB TTMLIR_LIBRARIES "${TTMLIR_LIB_DIR}/*.so")
    foreach(TTMLIR_LIBRARY ${TTMLIR_LIBRARIES})
        # Remove `lib` prefix and `.so` extention from shared lib name.
        get_filename_component(lib_name ${TTMLIR_LIBRARY} NAME_WE)
        string(REPLACE "lib" "" lib_name ${lib_name})

        add_library(${lib_name} SHARED IMPORTED GLOBAL)
        set_target_properties(${lib_name} PROPERTIES
            EXCLUDE_FROM_ALL TRUE
            IMPORTED_LOCATION ${TTMLIR_LIBRARY}
        )
        add_dependencies(${lib_name} tt-mlir)
    endforeach()

    # Since tt-mlir (commit 25883ede7ce89f96085b3fdc806be5852119d6f9) now exports protobuf
    # targets when runtime/opmodel is enabled, we can use those instead of hardcoded protobuf
    # The protobuf library and headers will come from tt-metal via tt-mlir
    #
    # However, we need to provide a fallback libprotobuf target for compatibility
    # during the build process before TTMLIR config is available

    # Create a fallback libprotobuf target that points to the system protobuf
    # This will be overridden if TTMLIR provides protobuf targets
    find_library(SYSTEM_PROTOBUF_LIB NAMES protobuf libprotobuf
                 PATHS ${TTMLIR_TOOLCHAIN_DIR}/lib
                 NO_DEFAULT_PATH)

    if(SYSTEM_PROTOBUF_LIB)
        add_library(libprotobuf STATIC IMPORTED GLOBAL)
        set_target_properties(libprotobuf PROPERTIES
            IMPORTED_LOCATION ${SYSTEM_PROTOBUF_LIB}
            INTERFACE_INCLUDE_DIRECTORIES "${TTMLIR_TOOLCHAIN_DIR}/include"
        )
        message(STATUS "Created fallback libprotobuf target using: ${SYSTEM_PROTOBUF_LIB}")
    else()
        message(WARNING "Protobuf library not found in toolchain. TTMLIR must provide protobuf targets.")
    endif()

    set(FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wno-undef -Wno-unsafe-buffer-usage -Wno-disabled-macro-expansion")

    ExternalProject_Add(
        loguru
        PREFIX ${TTPJRT_SOURCE_DIR}/third_party/loguru
        CMAKE_GENERATOR Ninja
        CMAKE_ARGS
          -DCMAKE_BUILD_TYPE=Release
          -DCMAKE_C_COMPILER=clang-17
          -DCMAKE_CXX_COMPILER=clang++-17
          -DCMAKE_CXX_FLAGS=${FLAGS}
          -DCMAKE_INSTALL_PREFIX=${TTPJRT_SOURCE_DIR}/third_party/loguru/src/loguru-install
        GIT_REPOSITORY https://github.com/emilk/loguru.git
        GIT_TAG ${LOGURU_VERSION}
        GIT_PROGRESS ON
    )

endif()
