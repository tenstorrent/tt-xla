# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# This file incorporates work covered by the following copyright and permission notice:
# SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# https://llvm.org/LICENSE.txt

# ----- TTPJRTApi -----
# Static lib containing core PJRT API implementation: handling device/client/buffer/etc.
# Depends on: TTMLIRCompiler, TTMLIRRuntime, TTPJRTUtils, coverage_config, protobuf::libprotobuf

add_library(TTPJRTApi
    "buffer_instance.cc"
    "client_instance.cc"
    "device_description.cc"
    "device_instance.cc"
    "error_instance.cc"
    "event_instance.cc"
    "executable_image.cc"
    "executable_instance.cc"
    "loaded_executable_instance.cc"
    "flatbuffer_loaded_executable_instance.cc"
    "so_loaded_executable_instance.cc"
    "memory_instance.cc"
    "compile_options.cc"
    "module_builder/frontend_passes/shlo_input_role_propagation.cc"
    "module_builder/frontend_passes/shlo_set_proper_sdy_mesh_attribute.cc"
    "module_builder/module_builder.cc"
    "plugin_attributes.cc"
    "serialized_executable_instance.cc"
)

target_include_directories(TTPJRTApi PUBLIC
    ${Protobuf_INCLUDE_DIRS}
    ${TTMLIR_TOOLCHAIN_DIR}/src/shardy
    ${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo
)

target_link_libraries(TTPJRTApi
PRIVATE
    dl
    coverage_config
    libprotobuf
    TTMLIRCompiler
    TTMLIRRuntime
PUBLIC
    TTPJRTUtils
)

target_link_directories(TTPJRTApi PUBLIC
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/install/lib
)
