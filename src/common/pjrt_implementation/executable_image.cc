// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "common/pjrt_implementation/executable_image.h"

#include <string>
#include <iostream>

#include "common/pjrt_implementation/utils.h"
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
    ExecutableImage *exec = ExecutableImage::Unwrap(args->executable);
    args->num_outputs = exec->result_count;
    std::cerr << "I AM HERE=" << args->num_outputs << std::endl;
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
    return ErrorInstance::MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Executable_DeserializeAndLoad =
      +[](PJRT_Executable_DeserializeAndLoad_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG,
           "ExecutableImage::PJRT_Executable_DeserializeAndLoad_Args");
    return ErrorInstance::MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Executable_Serialize =
      +[](PJRT_Executable_Serialize_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_Serialize_Args");
    return ErrorInstance::MakeError(tt_pjrt_status::kUnimplemented);
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
        return ErrorInstance::MakeError(tt_pjrt_status::kInvalidArgument);
      }
      std::memcpy(program->code, executable->code.c_str(),
                  executable->code.size());
    }
    return nullptr;
  };
  api->PJRT_Executable_GetCostAnalysis =
      +[](PJRT_Executable_GetCostAnalysis_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_GetCostAnalysis_Args");
    return ErrorInstance::MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Executable_OutputElementTypes =
      +[](PJRT_Executable_OutputElementTypes_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG,
           "ExecutableImage::PJRT_Executable_OutputElementTypes_Args");

    ExecutableImage *executable = ExecutableImage::Unwrap(args->executable);
    size_t num_outputs = 1;
    PJRT_Buffer_Type *output_types = new PJRT_Buffer_Type[num_outputs];
    output_types[0] = PJRT_Buffer_Type_F32;
    args->num_output_types = num_outputs;
    args->output_types = output_types;
    std::cerr << "num_outputs=" << args->num_output_types << std::endl;
    return nullptr;
  };
  api->PJRT_Executable_OutputDimensions =
      +[](PJRT_Executable_OutputDimensions_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_OutputDimensions_Args");
    
    ExecutableImage *executable = ExecutableImage::Unwrap(args->executable);
    int64_t *dims = new int64_t[2];
    dims[0] = 128;
    dims[1] = 128;
    size_t *dim_sizes = new size_t[1];
    dim_sizes[0] = 2;
    args->dims = dims;
    args->dim_sizes = dim_sizes;
    args->num_outputs = 1;
    std::cerr << "num_outputs=" << args->num_outputs << std::endl;
    return nullptr;
  };
  api->PJRT_Executable_OutputMemoryKinds =
      +[](PJRT_Executable_OutputMemoryKinds_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_OutputMemoryKinds");
    return ErrorInstance::MakeError(tt_pjrt_status::kUnimplemented);
  };
}

} // namespace tt::pjrt
