# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_library(TTPJRTCommonDylibPlatform
"dylib_platform.cc"
)

add_library(TTPJRTCommon
    "api_impl.cc"
    "platform.cc"
    "module_builder.cc"
)
add_dependencies(TTPJRTCommon tt-mlir)
target_include_directories(TTPJRTCommon PUBLIC 
    ${PROJECT_SOURCE_DIR}/third_party/pjrt_c_api
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/include
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/include/ttmlir/Target/Common
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir/include
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir/runtime/include
    /opt/ttmlir-toolchain/include/
    /opt/ttmlir-toolchain/src/stablehlo/
    /opt/ttmlir-toolchain/src/stablehlo-build/
)

set(STABLEHLO_LIBS
	StablehloBase
    CheckOps
    ChloCAPI
    ChloOps
    InterpreterOps
    StablehloAssemblyFormat
    StablehloBase
    StablehloBroadcastUtils
    StablehloCAPI
    StablehloLinalgTransforms
    StablehloOps
    StablehloPasses
    StablehloPortableApi
    StablehloRegister
    StablehloSerialization
    StablehloTOSATransforms
    StablehloTestUtils
    StablehloTypeInference
    Version
    VhloCAPI
    VhloOps
    VhloTypes
    StablehloBase
    )
    
    target_link_libraries(TTPJRTCommon PUBLIC
    LLVM
    MLIR
    TTMLIR
    TTPJRTCommonDylibPlatform
    ${STABLEHLO_LIBS}
)

target_link_directories(TTPJRTCommon PUBLIC 
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/lib/SharedLib
    /opt/ttmlir-toolchain/src/stablehlo-build/lib
)

