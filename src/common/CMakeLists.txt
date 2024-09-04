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
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/stablehlo/
    /opt/ttmlir-toolchain/include
    /opt/ttmlir-toolchain/src/stablehlo
)

set(STABLEHLO_LIBS
StablehloBase
StablehloReferenceIndex
StablehloReferenceErrors
StablehloReferenceElement
StablehloReferenceAxes
StablehloReferenceValue
StablehloReferenceTypes
StablehloReferenceToken
StablehloReferenceTensor
StablehloReferenceScope
StablehloReferenceProcessGrid
StablehloReferenceProcess
StablehloReferenceOps
StablehloPasses
ChloOps
VhloOps
VhloTypes
StablehloOps
StablehloRegister
StablehloReferenceToken
StablehloLinalgTransforms
StablehloReferenceValue
StablehloReferenceScope
StablehloPasses
StablehloReferenceOps
StablehloSerialization
InterpreterOps
StablehloPortableApi
StablehloPasses
StablehloAssemblyFormat    StablehloOps            StablehloReferenceElement  StablehloReferenceOps          StablehloReferenceTensor  StablehloRegister
StablehloBase              StablehloPasses         StablehloReferenceErrors   StablehloReferenceProcess      StablehloReferenceToken   StablehloSerialization
StablehloBroadcastUtils    StablehloPortableApi    StablehloReferenceIndex    StablehloReferenceProcessGrid  StablehloReferenceTypes   StablehloTypeInference
StablehloLinalgTransforms  StablehloReferenceAxes  StablehloReferenceNumPy    StablehloReferenceScope        StablehloReferenceValue
    )
    
    target_link_libraries(TTPJRTCommon PUBLIC
    LLVM
    MLIR
    TTMLIR
    TTPJRTCommonDylibPlatform
    TTMLIRStatic
    TTMLIRStableHLOToTTIR
    ${STABLEHLO_LIBS}
)

target_link_directories(TTPJRTCommon PUBLIC
    /opt/ttmlir-toolchain/lib
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/lib
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/lib/SharedLib
)

