# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# This file incorporates work covered by the following copyright and permission notice:
# SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# https://llvm.org/LICENSE.txt

# ----- TTPJRTUtils -----
# Static lib containing utility logic: logging initialization and data type utilities.
# Depends on: loguru, coverage_config

add_library(TTPJRTUtils
    "data_type_utils.cc"
    "logging.cc"
)

target_include_directories(TTPJRTUtils PUBLIC
    ${PROJECT_SOURCE_DIR}/pjrt_implementation/inc
    ${PROJECT_SOURCE_DIR}/third_party/loguru/src/loguru-install/include/
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/install/include
    ${PROJECT_SOURCE_DIR}/third_party/pjrt_c_api
    ${TTMLIR_TOOLCHAIN_DIR}/include
)

add_dependencies(TTPJRTUtils
    tt-mlir
)

target_link_libraries(TTPJRTUtils PUBLIC
    loguru
)

target_link_directories(TTPJRTUtils PUBLIC
    ${PROJECT_SOURCE_DIR}/third_party/loguru/src/loguru-install/lib/
)
