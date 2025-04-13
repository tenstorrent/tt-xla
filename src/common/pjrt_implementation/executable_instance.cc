// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "common/pjrt_implementation/executable_instance.h"

// c++ standard library includes
#include <cstring>
#include <string>

// tt-xla includes
#include "common/status.h"

namespace tt::pjrt {

std::unique_ptr<ExecutableInstance> ExecutableInstance::createInstance(
    std::shared_ptr<ExecutableImage> executable_image) {
  return std::make_unique<ExecutableInstance>(std::move(executable_image));
}

void ExecutableInstance::bindApi(PJRT_Api *api) {
  api->PJRT_Executable_Destroy = internal::onExecutableDestroy;
  api->PJRT_Executable_Name = internal::onExecutableName;
  api->PJRT_Executable_NumReplicas = internal::onExecutableNumReplicas;
  api->PJRT_Executable_NumPartitions = internal::onExecutableNumPartitions;
  api->PJRT_Executable_OptimizedProgram =
      internal::onExecutableOptimizedProgram;
  api->PJRT_Executable_NumOutputs = internal::onExecutableNumOutputs;
  api->PJRT_Executable_SizeOfGeneratedCodeInBytes =
      internal::onExecutableSizeOfGeneratedCodeInBytes;
  api->PJRT_Executable_OutputElementTypes =
      internal::onExecutableOutputElementTypes;
  api->PJRT_Executable_OutputDimensions =
      internal::onExecutableOutputDimensions;
}

namespace internal {

PJRT_Error *onExecutableDestroy(PJRT_Executable_Destroy_Args *args) {
  DLOG_F(LOG_DEBUG, "ExecutableInstance::PJRT_Executable_Destroy");

  delete ExecutableInstance::Unwrap(args->executable);

  return nullptr;
}

PJRT_Error *onExecutableName(PJRT_Executable_Name_Args *args) {
  DLOG_F(LOG_DEBUG, "ExecutableInstance::PJRT_Executable_Name");

  ExecutableInstance *executable_instance =
      ExecutableInstance::Unwrap(args->executable);
  const std::string &executable_name =
      executable_instance->getExecutableImage()->getExecutableName();

  args->executable_name = executable_name.data();
  args->executable_name_size = executable_name.size();

  return nullptr;
}

PJRT_Error *onExecutableNumReplicas(PJRT_Executable_NumReplicas_Args *args) {
  DLOG_F(LOG_DEBUG, "ExecutableInstance::PJRT_Executable_NumReplicas");

  ExecutableInstance *executable_instance =
      ExecutableInstance::Unwrap(args->executable);

  args->num_replicas =
      executable_instance->getExecutableImage()->getNumReplicas();

  return nullptr;
}

PJRT_Error *
onExecutableNumPartitions(PJRT_Executable_NumPartitions_Args *args) {
  DLOG_F(LOG_DEBUG, "ExecutableInstance::PJRT_Executable_NumPartitions");

  ExecutableInstance *executable_instance =
      ExecutableInstance::Unwrap(args->executable);

  args->num_partitions =
      executable_instance->getExecutableImage()->getNumPartitions();

  return nullptr;
}

PJRT_Error *
onExecutableOptimizedProgram(PJRT_Executable_OptimizedProgram_Args *args) {
  DLOG_F(LOG_DEBUG, "ExecutableInstance::PJRT_Executable_OptimizedProgram");

  ExecutableInstance *executable_instance =
      ExecutableInstance::Unwrap(args->executable);

  PJRT_Program *program = args->program;
  program->format = ModuleBuilder::c_mlir_format_name.data();
  program->format_size = ModuleBuilder::c_mlir_format_name.size();

  const std::string &optimized_mlir_code =
      executable_instance->getExecutableImage()->getOptimizedMlirCode();
  size_t code_size = optimized_mlir_code.size();

  if (program->code == nullptr) {
    program->code_size = code_size;
  } else {
    if (program->code_size < code_size) {
      DLOG_F(ERROR,
             "Not enough space allocated for optimized program: expected %zu "
             "bytes, allocated %zu bytes",
             code_size, program->code_size);
      return ErrorInstance::makeError(tt_pjrt_status::kInvalidArgument);
    }

    std::memcpy(program->code, optimized_mlir_code.data(), code_size);
  }

  return nullptr;
}

PJRT_Error *onExecutableNumOutputs(PJRT_Executable_NumOutputs_Args *args) {
  DLOG_F(LOG_DEBUG, "ExecutableInstance::PJRT_Executable_NumOutputs");

  ExecutableInstance *executable_instance =
      ExecutableInstance::Unwrap(args->executable);

  args->num_outputs =
      executable_instance->getExecutableImage()->getNumOutputs();

  return nullptr;
}

PJRT_Error *onExecutableSizeOfGeneratedCodeInBytes(
    PJRT_Executable_SizeOfGeneratedCodeInBytes_Args *args) {
  DLOG_F(LOG_DEBUG,
         "ExecutableInstance::PJRT_Executable_SizeOfGeneratedCodeInBytes");

  // In XLA they count this into on-device memory usage needed to run an
  // executable. Only their GPU client implements it (in an unclear way), other
  // clients either return 0 or -1, so it is probably not required to implement.
  // Returning -1 for now since we cannot estimate device memory usage.
  args->size_in_bytes = -1;

  return nullptr;
}

PJRT_Error *
onExecutableOutputElementTypes(PJRT_Executable_OutputElementTypes_Args *args) {
  DLOG_F(LOG_DEBUG, "ExecutableInstance::PJRT_Executable_OutputElementTypes");

  ExecutableInstance *executable_instance =
      ExecutableInstance::Unwrap(args->executable);

  args->output_types =
      executable_instance->getExecutableImage()->getOutputTypesRaw();
  args->num_output_types =
      executable_instance->getExecutableImage()->getNumOutputs();

  return nullptr;
}

PJRT_Error *
onExecutableOutputDimensions(PJRT_Executable_OutputDimensions_Args *args) {
  DLOG_F(LOG_DEBUG, "ExecutableInstance::PJRT_Executable_OutputDimensions");

  ExecutableInstance *executable_instance =
      ExecutableInstance::Unwrap(args->executable);

  args->dims =
      executable_instance->getExecutableImage()->getOutputDimensionsFlatRaw();
  args->dim_sizes =
      executable_instance->getExecutableImage()->getOutputRanksRaw();

  return nullptr;
}

} // namespace internal

} // namespace tt::pjrt
