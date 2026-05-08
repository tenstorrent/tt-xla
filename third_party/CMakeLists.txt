# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

option(USE_CUSTOM_TT_MLIR_VERSION "Flag to use TT_MLIR_VERSION set by the user" OFF)

if (NOT DEFINED TT_MLIR_VERSION OR NOT USE_CUSTOM_TT_MLIR_VERSION)
    set(TT_MLIR_VERSION "7a32a979c5fcf19203015f6baa93dbd1effab6ac")
endif()

# Local-development override: point ExternalProject_Add at a file:// URL of a
# sibling tt-mlir checkout instead of cloning from GitHub. Set TT_MLIR_REPO and
# TT_MLIR_VERSION (branch or SHA in that repo) when iterating against an
# uncommitted tt-mlir branch. Default falls back to the public GitHub repo.
if (NOT DEFINED TT_MLIR_REPO)
    set(TT_MLIR_REPO "https://github.com/tenstorrent/tt-mlir.git")
endif()

set(LOGURU_VERSION "4adaa185883e3c04da25913579c451d3c32cfac1")

if (TOOLCHAIN STREQUAL "ON")
    cmake_minimum_required(VERSION 3.20)
    project(ttmlir-toolchain LANGUAGES CXX C)
    execute_process(
        COMMAND git clone --recursive ${TT_MLIR_REPO} ${PROJECT_SOURCE_DIR}/tt-mlir/src/tt-mlir
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
    include(GNUInstallDirs)
    add_subdirectory(pjrt_c_api)

    # ----- tt-mlir -----
    set(TTMLIR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tt-mlir")
    set(TTMLIR_BUILD_DIR "${TTMLIR_SOURCE_DIR}/src/tt-mlir/build")
    set(WITH_METAL_RUNTIME_ROOT_SET "")
    if(NOT DEFINED ENV{TT_METAL_RUNTIME_ROOT})
        set(WITH_METAL_RUNTIME_ROOT_SET "TT_METAL_RUNTIME_ROOT=${TTPJRT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal")
    endif()
    set(TT_RUNTIME_DEBUG ${TT_RUNTIME_DEBUG} CACHE BOOL "Enable ttmlir runtime debugging")
    if(TTXLA_ENABLE_EXPLORER)
        set(TT_RUNTIME_DEBUG ON)
    endif()

    message(STATUS "ttmlir runtime debugging is set to: ${TT_RUNTIME_DEBUG}")

    ExternalProject_Add(
        tt-mlir
        PREFIX ${TTPJRT_SOURCE_DIR}/third_party/tt-mlir

        # Installing the python dependencies before the build
        PATCH_COMMAND mkdir -p ${TTMLIR_BUILD_DIR}
        COMMAND TTPJRT_SOURCE_DIR=${TTPJRT_SOURCE_DIR} bash ${TTPJRT_SOURCE_DIR}/venv/install_ttmlir_requirements.sh

        CMAKE_GENERATOR Ninja
        BINARY_DIR ${TTMLIR_BUILD_DIR}

        BUILD_COMMAND env ${WITH_METAL_RUNTIME_ROOT_SET} ${CMAKE_COMMAND} --build <BINARY_DIR>

        INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR> --component SharedLib
        COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR> --component DistributedRuntime

        CMAKE_ARGS
          -DCMAKE_BUILD_TYPE=${TTMLIR_BUILD_TYPE}
          -DCMAKE_C_COMPILER=clang
          -DCMAKE_CXX_COMPILER=clang++
          -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
          # On dev hosts where the toolchain isn't at /opt/ttmlir-toolchain, the
          # tt-mlir cmake's FindMLIR can't auto-locate MLIRConfig.cmake. Make the
          # ExternalProject pass these explicitly from the parent's TTMLIR_TOOLCHAIN_DIR.
          -DMLIR_DIR=${TTMLIR_TOOLCHAIN_DIR}/lib/cmake/mlir
          -DCMAKE_PREFIX_PATH=${TTMLIR_TOOLCHAIN_DIR}
          # Force the toolchain's python (3.12 with pybind11 2.10+) instead
          # of CMake auto-discovering the system /bin/python3 (3.10 with
          # pybind11 2.9.1, which MLIRDetectPythonEnv rejects).
          -DPython3_EXECUTABLE=${TTMLIR_TOOLCHAIN_DIR}/venv/bin/python3
          -DTT_RUNTIME_ENABLE_TTNN=ON
          -DTTMLIR_ENABLE_STABLEHLO=ON
          -DTTMLIR_ENABLE_RUNTIME=ON
          -DTT_RUNTIME_DEBUG=${TT_RUNTIME_DEBUG}
          -DTTMLIR_ENABLE_BINDINGS_PYTHON=${TTMLIR_ENABLE_BINDINGS_PYTHON}
          -DCMAKE_INSTALL_PREFIX=${TTMLIR_INSTALL_PREFIX}
          -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
          -DTT_RUNTIME_ENABLE_PERF_TRACE=${TTMLIR_ENABLE_PERF_TRACE}
          -DTTMLIR_ENABLE_OPMODEL=ON
          -DTTMLIR_ENABLE_EXPLORER=${TTXLA_ENABLE_EXPLORER}
          -DTTMLIR_ENABLE_TESTS=OFF
          -DTTMLIR_ENABLE_TOOLS=${TTXLA_ENABLE_TOOLS}
          -DTTMLIR_ENABLE_DOCS=OFF
          -DTT_USE_SYSTEM_SFPI=${TT_USE_SYSTEM_SFPI}
        GIT_REPOSITORY ${TT_MLIR_REPO}
        GIT_TAG ${TT_MLIR_VERSION}
        GIT_PROGRESS ON
    )

    if(TTXLA_ENABLE_EXPLORER)
        ExternalProject_Add_Step(tt-mlir build_explorer
            COMMAND env ${WITH_METAL_RUNTIME_ROOT_SET} ${CMAKE_COMMAND} --build <BINARY_DIR> -- explorer
            DEPENDEES build
            WORKING_DIRECTORY ${TTMLIR_BUILD_DIR}
        )
    endif()

    set_target_properties(tt-mlir PROPERTIES EXCLUDE_FROM_ALL TRUE)

    install(DIRECTORY ${TTMLIR_INSTALL_PREFIX}/ DESTINATION "${CMAKE_INSTALL_PREFIX}" USE_SOURCE_PERMISSIONS)

    # Expose all tt-mlir .so files as shared imported libraries.
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

    set(FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wno-undef -Wno-unsafe-buffer-usage -Wno-disabled-macro-expansion")

    ExternalProject_Add(
        loguru
        PREFIX ${TTPJRT_SOURCE_DIR}/third_party/loguru
        CMAKE_GENERATOR Ninja
        CMAKE_ARGS
          -DCMAKE_BUILD_TYPE=Release
          -DCMAKE_C_COMPILER=clang
          -DCMAKE_CXX_COMPILER=clang++
          -DCMAKE_CXX_FLAGS=${FLAGS}
          -DCMAKE_INSTALL_PREFIX=${LOGURU_INSTALL_PREFIX}
          -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
        GIT_REPOSITORY https://github.com/emilk/loguru.git
        GIT_TAG ${LOGURU_VERSION}
        GIT_PROGRESS ON
    )

endif()
