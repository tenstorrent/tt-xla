// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "common/api_impl.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <optional>
#include <sstream>
#include <utility>

#include "common/module_builder.h"
#include "common/status.h"

namespace tt::pjrt {

const std::string_view kMlirFormat = "mlir";

void ExecutableImage::BindApi(PJRT_Api *api) {
  api->PJRT_Executable_Destroy =
      +[](PJRT_Executable_Destroy_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_Destroy");
    ExecutableImage::Unwrap(args->executable)->DecRef();
    return nullptr;
  };
  api->PJRT_Executable_Name =
      +[](PJRT_Executable_Name_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_Name");
    const char *dummy_name = "tt_pjrt_exe";
    args->executable_name = dummy_name;
    args->executable_name_size = std::strlen(dummy_name);
    return nullptr;
  };
  api->PJRT_Executable_SizeOfGeneratedCodeInBytes =
      +[](PJRT_Executable_SizeOfGeneratedCodeInBytes_Args *args)
      -> PJRT_Error * {
    DLOG_F(LOG_DEBUG,
           "ExecutableImage::PJRT_Executable_SizeOfGeneratedCodeInBytes");
    args->size_in_bytes =
        0; // TODO:
           // ExecutableImage::Unwrap(args->executable)->binary->GetDataSize();
    return nullptr;
  };
  api->PJRT_Executable_NumOutputs =
      +[](PJRT_Executable_NumOutputs_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_NumOutputs");
    auto *exec = ExecutableImage::Unwrap(args->executable);
    args->num_outputs = exec->result_count;
    return nullptr;
  };
  api->PJRT_Executable_NumPartitions =
      +[](PJRT_Executable_NumPartitions_Args *args) -> PJRT_Error * {
    // This should be updated once iree supports partitioning.
    args->num_partitions = 1;
    return nullptr;
  };
  api->PJRT_Executable_NumReplicas =
      +[](PJRT_Executable_NumReplicas_Args *args) -> PJRT_Error * {
    // This should be updated once iree supports replicas.
    args->num_replicas = 1;
    return nullptr;
  };
  api->PJRT_Executable_Serialize =
      +[](PJRT_Executable_Serialize_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_Serialize");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Executable_DeserializeAndLoad =
      +[](PJRT_Executable_DeserializeAndLoad_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG,
           "ExecutableImage::PJRT_Executable_DeserializeAndLoad_Args");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Executable_Serialize =
      +[](PJRT_Executable_Serialize_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_Serialize_Args");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Executable_OptimizedProgram =
      +[](PJRT_Executable_OptimizedProgram_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_OptimizedProgram");
    ExecutableImage *executable = ExecutableImage::Unwrap(args->executable);
    PJRT_Program *program = args->program;
    program->format = kMlirFormat.data();
    program->format_size = kMlirFormat.size();
    size_t code_size = executable->code.size();
    if (program->code == nullptr) {
      program->code_size = code_size;
    } else {
      if (program->code_size < code_size) {
        return MakeError(tt_pjrt_status::kInvalidArgument);
      }
      std::memcpy(program->code, executable->code.c_str(),
                  executable->code.size());
    }
    return nullptr;
  };
  api->PJRT_Executable_GetCostAnalysis =
      +[](PJRT_Executable_GetCostAnalysis_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_GetCostAnalysis_Args");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Executable_OutputElementTypes =
      +[](PJRT_Executable_OutputElementTypes_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG,
           "ExecutableImage::PJRT_Executable_OutputElementTypes_Args");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Executable_OutputDimensions =
      +[](PJRT_Executable_OutputDimensions_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_OutputDimensions_Args");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Executable_OutputMemoryKinds =
      +[](PJRT_Executable_OutputMemoryKinds_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_OutputMemoryKinds");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
}

static void BindUndefineds(PJRT_Api *api) {
#define _STUB(API)                                                             \
  api->API = +[](API##_Args *args) -> decltype(api->API(args)) {               \
    DLOG_F(LOG_DEBUG, "STUB: " #API);                                          \
    return (decltype(api->API(args)))MakeError(                                \
        tt_pjrt_status::kUnimplemented);                                       \
  }

#include "stubs.inc"
}

//===----------------------------------------------------------------------===//
// Top-level API binding.
//===----------------------------------------------------------------------===//

void BindMonomorphicApi(PJRT_Api *api) {
  api->struct_size = PJRT_Api_STRUCT_SIZE;
  api->extension_start = nullptr;
  api->pjrt_api_version.major_version = PJRT_API_MAJOR;
  api->pjrt_api_version.minor_version = PJRT_API_MINOR;

  // This is a bare implementation throwing UNDEFINED errors. This way new
  // functions will not segmentation fault on invocation.
  BindUndefineds(api);
  ErrorInstance::BindApi(api);

  api->PJRT_Plugin_Initialize =
      +[](PJRT_Plugin_Initialize_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "PJRT_Plugin_Initialize");
    return nullptr;
  };

  api->PJRT_Plugin_Attributes =
      +[](PJRT_Plugin_Attributes_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "PJRT_Plugin_Attributes");
    args->num_attributes = 0;
    return nullptr;
  };

  // Bind by object types.
  BufferInstance::BindApi(api);
  ClientInstance::BindApi(api);
  DeviceDescription::BindApi(api);
  DeviceInstance::BindApi(api);
  EventInstance::BindApi(api);
  ExecutableImage::BindApi(api);
  LoadedExecutableInstance::BindApi(api);
}

} // namespace tt::pjrt
