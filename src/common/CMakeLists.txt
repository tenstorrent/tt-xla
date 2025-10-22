# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# This file incorporates work covered by the following copyright and permission notice:
# SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# https://llvm.org/LICENSE.txt

# ----- TTPJRTCommonDylibPlatform -----
# Static lib containing platform specific logic.
# Depends on: loguru

add_library(TTPJRTCommonDylibPlatform
    "dylib_platform.cc"
)

add_dependencies(TTPJRTCommonDylibPlatform
    loguru
)

target_include_directories(TTPJRTCommonDylibPlatform PUBLIC
    ${PROJECT_SOURCE_DIR}/third_party/loguru/src/loguru-install/include/
)

# ----- TTPJRTCommon -----
# Static lib containing core logic implementation, handling device/client/buffer/etc.
# Depends on: TTMLIRCompiler, TTMLIRRuntime, TTPJRTCommonDylibPlatform, loguru, coverage_config

add_library(TTPJRTCommon
    "platform.cc"
    "plugin_attributes.cc"
    "pjrt_implementation/api_bindings.cc"
    "pjrt_implementation/buffer_instance.cc"
    "pjrt_implementation/client_instance.cc"
    "pjrt_implementation/data_type_utils.cc"
    "pjrt_implementation/device_description.cc"
    "pjrt_implementation/device_instance.cc"
    "pjrt_implementation/error_instance.cc"
    "pjrt_implementation/event_instance.cc"
    "pjrt_implementation/executable_image.cc"
    "pjrt_implementation/executable_instance.cc"
    "pjrt_implementation/loaded_executable_instance.cc"
    "pjrt_implementation/flatbuffer_loaded_executable_instance.cc"
    "pjrt_implementation/so_loaded_executable_instance.cc"
    "pjrt_implementation/memory_instance.cc"
    "pjrt_implementation/compile_options.cc"
    "pjrt_implementation/module_builder/frontend_passes/shlo_input_role_propagation.cc"
    "pjrt_implementation/module_builder/frontend_passes/shlo_set_proper_sdy_mesh_attribute.cc"
    "pjrt_implementation/module_builder/module_builder.cc"
    "pjrt_implementation/serialized_executable_instance.cc"
)

add_dependencies(TTPJRTCommon
    tt-mlir
    loguru
)

target_include_directories(TTPJRTCommon PUBLIC
    ${PROJECT_SOURCE_DIR}/inc
    ${PROJECT_SOURCE_DIR}/third_party/loguru/src/loguru-install/include/
    ${PROJECT_SOURCE_DIR}/third_party/pjrt_c_api
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/install/include
    ${TTMLIR_TOOLCHAIN_DIR}/include
    ${TTMLIR_TOOLCHAIN_DIR}/src/shardy
    ${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo
)

target_link_libraries(TTPJRTCommon PUBLIC
    dl
    libprotobuf
    loguru
    coverage_config
    TTPJRTCommonDylibPlatform
    TTMLIRCompiler
    TTMLIRRuntime
)

target_link_directories(TTPJRTCommon PUBLIC
    ${TTMLIR_TOOLCHAIN_DIR}/lib
    ${PROJECT_SOURCE_DIR}/third_party/loguru/src/loguru-install/lib/
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/install/lib
)
