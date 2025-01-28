# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#

set(TT_MLIR_VERSION "e581e6a815de6b0ce3c4ceedc70620d19be29969")
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
    set(TTMLIR_LIB_DIR ${TTPJRT_SOURCE_DIR}/install)
    ExternalProject_Add(
        tt-mlir
        PREFIX ${TTPJRT_SOURCE_DIR}/third_party/tt-mlir
        CMAKE_GENERATOR Ninja
        CMAKE_ARGS
          -DCMAKE_BUILD_TYPE=Release
          -DCMAKE_C_COMPILER=clang-17
          -DCMAKE_CXX_COMPILER=clang++-17
          -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
          -DTT_RUNTIME_ENABLE_TTNN=ON
          -DTTMLIR_ENABLE_STABLEHLO=ON
          -DTTMLIR_ENABLE_RUNTIME=ON
          -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
        GIT_REPOSITORY https://github.com/tenstorrent/tt-mlir.git
        GIT_TAG ${TT_MLIR_VERSION}
        GIT_PROGRESS ON
    )

    # #TODO: Remove this once we have a proper install target
    # message(STATUS "Copying ttmetal libraries from ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal-build/lib to ${TTMLIR_LIB_DIR}/lib")
    # execute_process(
    #     COMMAND /bin/bash -c "mkdir -p ${TTMLIR_LIB_DIR}/lib/ && cp ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal-build/lib/* ${TTMLIR_LIB_DIR}/lib/"
    # )

    set_target_properties(tt-mlir PROPERTIES EXCLUDE_FROM_ALL TRUE)

    # Link to all TTMLIR libraries in ${TTMLIR_INSTALL_DIR}/install/lib
    set(TTMLIR_LIB_DIR ${TTMLIR_LIB_DIR} PARENT_SCOPE)
    message(STATUS "Linking to TTMLIR libraries in ${TTMLIR_LIB_DIR}")
    file(GLOB TTMLIR_LIBRARIES "${TTMLIR_LIB_DIR}/*.so")
    foreach(TTMLIR_LIBRARY ${TTMLIR_LIBRARIES})
        get_filename_component(lib_name ${TTMLIR_LIBRARY} NAME_WE)
        string(REPLACE "lib" "" lib_name ${lib_name}) # Remove the "lib" prefix if it exists
        add_library(${lib_name} SHARED IMPORTED GLOBAL)
        set_target_properties(${lib_name} PROPERTIES EXCLUDE_FROM_ALL TRUE IMPORTED_LOCATION ${TTMLIR_LIBRARY})
        add_dependencies(${lib_name} tt-mlir)
    endforeach()

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
