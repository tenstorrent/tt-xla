# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# This file incorporates work covered by the following copyright and permission notice:
# SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# https://llvm.org/LICENSE.txt

add_library(TTPJRTCommonDylibPlatform
"dylib_platform.cc"
)
target_include_directories(TTPJRTCommonDylibPlatform PUBLIC
    ${PROJECT_SOURCE_DIR}/third_party/loguru/src/loguru-install/include/
)
add_dependencies(TTPJRTCommonDylibPlatform loguru)

add_library(TTPJRTCommon
    "platform.cc"
    "module_builder.cc"
    "plugin_attributes.cc"
    "pjrt_implementation/api_bindings.cc"
    "pjrt_implementation/buffer_instance.cc"
    "pjrt_implementation/client_instance.cc"
    "pjrt_implementation/device_description.cc"
    "pjrt_implementation/device_instance.cc"
    "pjrt_implementation/error_instance.cc"
    "pjrt_implementation/event_instance.cc"
    "pjrt_implementation/executable_image.cc"
    "pjrt_implementation/loaded_executable_instance.cc"
    "pjrt_implementation/utils.cc"
)
add_dependencies(TTPJRTCommon tt-mlir loguru)
target_include_directories(TTPJRTCommon PUBLIC
    ${PROJECT_SOURCE_DIR}/inc
    ${PROJECT_SOURCE_DIR}/third_party/loguru/src/loguru-install/include/
    ${PROJECT_SOURCE_DIR}/third_party/pjrt_c_api
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/include
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/include/ttmlir/Target/Common
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir/include
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir/runtime/include
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/shardy/
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/stablehlo/
    ${TTMLIR_TOOLCHAIN_DIR}/include
    ${TTMLIR_TOOLCHAIN_DIR}/src/shardy
    ${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo
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
Version
VhloOps
VhloTypes
SdyDialect
SdyRegister
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
StablehloAssemblyFormat
StablehloOps
StablehloReferenceElement
StablehloReferenceOps
StablehloReferenceTensor
StablehloRegister
StablehloBase
StablehloPasses
StablehloReferenceErrors
StablehloReferenceProcess
StablehloReferenceToken
StablehloSerialization
StablehloBroadcastUtils
StablehloPortableApi
StablehloReferenceIndex
StablehloReferenceProcessGrid
StablehloReferenceTypes
StablehloTypeInference
StablehloLinalgTransforms
StablehloReferenceAxes
StablehloReferenceNumPy
StablehloReferenceScope
StablehloReferenceValue
)

target_link_libraries(TTPJRTCommon PUBLIC
    loguru
    LLVM
    MLIR
    TTMLIR
    TTPJRTCommonDylibPlatform
    TTMLIRStatic
    TTMLIRTosaToTTIR
    TTMLIRTTIRToLinalg
    MLIRTTIRPipelines
    TTMLIRStableHLOToTTIR
    ${STABLEHLO_LIBS}
)

target_link_libraries(TTPJRTCommon PUBLIC coverage_config)

target_link_directories(TTPJRTCommon PUBLIC
    ${TTMLIR_TOOLCHAIN_DIR}/lib
    ${PROJECT_SOURCE_DIR}/third_party/loguru/src/loguru-install/lib/
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/lib
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/lib/SharedLib
)
