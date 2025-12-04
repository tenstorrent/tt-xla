# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

option(USE_CUSTOM_TT_MLIR_VERSION "Flag to use TT_MLIR_VERSION set by the user" OFF)

if (NOT DEFINED TT_MLIR_VERSION OR NOT USE_CUSTOM_TT_MLIR_VERSION)
    set(TT_MLIR_VERSION "dc338264791f9de5719e7f08bb5a93c0fd4b19a0")
endif()

set(PROTOBUF_VERSION "v21.12") # same version as tt-metal uses
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

    # ----- protobuf -----

    include(FetchContent)

    FetchContent_Declare(
        protobuf
        GIT_REPOSITORY https://github.com/protocolbuffers/protobuf.git
        GIT_TAG ${PROTOBUF_VERSION}
        GIT_SHALLOW ON
    )

    # match options from metal
    set(protobuf_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(protobuf_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

    # Use full compiler paths since it can't find it during build config phase
    if(CMAKE_C_COMPILER MATCHES "clang-17")
        set(CMAKE_C_COMPILER "/usr/bin/clang-17" CACHE STRING "" FORCE)
    endif()
    if(CMAKE_CXX_COMPILER MATCHES "clang\\+\\+-17")
        set(CMAKE_CXX_COMPILER "/usr/bin/clang++-17" CACHE STRING "" FORCE)
    endif()

    FetchContent_MakeAvailable(protobuf)

    # ----- tt-mlir -----

    set(TTMLIR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tt-mlir")
    set(TTMLIR_BUILD_DIR "${TTMLIR_SOURCE_DIR}/src/tt-mlir/build")
    set(TTMLIR_LIB_DIR "${TTMLIR_SOURCE_DIR}/install")
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
        COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target tt-alchemist

        INSTALL_COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR> --component SharedLib
        COMMAND ${CMAKE_COMMAND} --install <BINARY_DIR> --component DistributedRuntime

        CMAKE_ARGS
          -DCMAKE_BUILD_TYPE=${TTMLIR_BUILD_TYPE}
          -DCMAKE_C_COMPILER=clang-17
          -DCMAKE_CXX_COMPILER=clang++-17
          -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
          -DTT_RUNTIME_ENABLE_TTNN=ON
          -DTTMLIR_ENABLE_STABLEHLO=ON
          -DTTMLIR_ENABLE_RUNTIME=ON
          -DTT_RUNTIME_DEBUG=${TT_RUNTIME_DEBUG}
          -DTTMLIR_ENABLE_BINDINGS_PYTHON=${TTXLA_ENABLE_EXPLORER}
          -DCMAKE_INSTALL_PREFIX=${TTMLIR_LIB_DIR}
          -DTT_RUNTIME_ENABLE_PERF_TRACE=${TTMLIR_ENABLE_PERF_TRACE}
          -DTTMLIR_ENABLE_OPMODEL=ON
          -DTTMLIR_ENABLE_EXPLORER=${TTXLA_ENABLE_EXPLORER}
        GIT_REPOSITORY https://github.com/tenstorrent/tt-mlir.git
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
