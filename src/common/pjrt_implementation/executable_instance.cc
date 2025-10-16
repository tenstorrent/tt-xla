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

// tracy includes
#include <tracy/Tracy.hpp>

// tt-xla includes
#include "common/pjrt_implementation/client_instance.h"
#include "common/pjrt_implementation/error_instance.h"
#include "common/pjrt_implementation/module_builder/module_builder.h"
#include "common/pjrt_implementation/serialized_executable_instance.h"
#include "common/status.h"

namespace tt::pjrt {

std::unique_ptr<ExecutableInstance> ExecutableInstance::createInstance(
    std::shared_ptr<ExecutableImage> executable_image) {
  struct make_unique_enabler : public ExecutableInstance {
    make_unique_enabler(std::shared_ptr<ExecutableImage> executable_image)
        : ExecutableInstance(std::move(executable_image)) {}
  };

  return std::make_unique<make_unique_enabler>(std::move(executable_image));
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
  api->PJRT_Executable_Fingerprint = internal::onExecutableFingerprint;
  api->PJRT_Executable_OutputElementTypes =
      internal::onExecutableOutputElementTypes;
  api->PJRT_Executable_OutputDimensions =
      internal::onExecutableOutputDimensions;
  api->PJRT_Executable_OutputMemoryKinds =
      internal::onExecutableOutputMemoryKinds;
  api->PJRT_Executable_Serialize = internal::onExecutableSerialize;
}

namespace internal {

PJRT_Error *onExecutableDestroy(PJRT_Executable_Destroy_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "ExecutableInstance::PJRT_Executable_Destroy");

  delete ExecutableInstance::unwrap(args->executable);

  return nullptr;
}

PJRT_Error *onExecutableName(PJRT_Executable_Name_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "ExecutableInstance::PJRT_Executable_Name");

  ExecutableInstance *executable_instance =
      ExecutableInstance::unwrap(args->executable);
  const std::string &executable_name =
      executable_instance->getExecutableImage()->getExecutableName();

  args->executable_name = executable_name.data();
  args->executable_name_size = executable_name.size();

  return nullptr;
}

PJRT_Error *onExecutableNumReplicas(PJRT_Executable_NumReplicas_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "ExecutableInstance::PJRT_Executable_NumReplicas");

  ExecutableInstance *executable_instance =
      ExecutableInstance::unwrap(args->executable);

  args->num_replicas =
      executable_instance->getExecutableImage()->getNumReplicas();

  return nullptr;
}

PJRT_Error *
onExecutableNumPartitions(PJRT_Executable_NumPartitions_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "ExecutableInstance::PJRT_Executable_NumPartitions");

  ExecutableInstance *executable_instance =
      ExecutableInstance::unwrap(args->executable);

  args->num_partitions =
      executable_instance->getExecutableImage()->getNumPartitions();

  return nullptr;
}

PJRT_Error *
onExecutableOptimizedProgram(PJRT_Executable_OptimizedProgram_Args *args) {
  DLOG_F(LOG_DEBUG, "ExecutableInstance::PJRT_Executable_OptimizedProgram");

  ExecutableInstance *executable_instance =
      ExecutableInstance::unwrap(args->executable);

  PJRT_Program *program = args->program;
  program->format = module_builder::c_mlir_format_name.data();
  program->format_size = module_builder::c_mlir_format_name.size();

  const std::string &original_mlir_code =
      executable_instance->getExecutableImage()->getOriginalMlirCode();
  size_t code_size = original_mlir_code.size();

  if (program->code == nullptr) {
    program->code_size = code_size;
  } else {
    if (program->code_size < code_size) {
      DLOG_F(ERROR,
             "Not enough space allocated for optimized program: expected %zu "
             "bytes, allocated %zu bytes",
             code_size, program->code_size);
      return *ErrorInstance::makeError(tt_pjrt_status::kInvalidArgument)
                  .release();
    }

    std::memcpy(program->code, original_mlir_code.data(), code_size);
  }

  return nullptr;
}

PJRT_Error *onExecutableNumOutputs(PJRT_Executable_NumOutputs_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "ExecutableInstance::PJRT_Executable_NumOutputs");

  ExecutableInstance *executable_instance =
      ExecutableInstance::unwrap(args->executable);

  args->num_outputs =
      executable_instance->getExecutableImage()->getNumOutputs();

  return nullptr;
}

PJRT_Error *onExecutableSizeOfGeneratedCodeInBytes(
    PJRT_Executable_SizeOfGeneratedCodeInBytes_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG,
         "ExecutableInstance::PJRT_Executable_SizeOfGeneratedCodeInBytes");

  // In XLA they count this into on-device memory usage needed to run an
  // executable. Only their GPU client implements it (in an unclear way), other
  // clients either return 0 or -1, so it is probably not required to implement.
  // Returning -1 for now since we cannot estimate device memory usage.
  args->size_in_bytes = -1;

  return nullptr;
}

PJRT_Error *onExecutableFingerprint(PJRT_Executable_Fingerprint_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "ExecutableInstance::PJRT_Executable_Fingerprint");

  const ExecutableInstance *executable_instance =
      ExecutableInstance::unwrap(args->executable);

  const std::string &fingerprint =
      executable_instance->getExecutableImage()->getFingerprint();

  args->executable_fingerprint = fingerprint.data();
  args->executable_fingerprint_size = fingerprint.size();

  return nullptr;
}

PJRT_Error *
onExecutableOutputElementTypes(PJRT_Executable_OutputElementTypes_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "ExecutableInstance::PJRT_Executable_OutputElementTypes");

  ExecutableInstance *executable_instance =
      ExecutableInstance::unwrap(args->executable);

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
      ExecutableInstance::unwrap(args->executable);

  // Not documented as out parameter in the PJRT API but turns out that it is
  // required to be set.
  // https://github.com/openxla/xla/issues/25211
  args->num_outputs =
      executable_instance->getExecutableImage()->getNumOutputs();

  args->dims =
      executable_instance->getExecutableImage()->getOutputDimensionsFlatRaw();
  args->dim_sizes =
      executable_instance->getExecutableImage()->getOutputRanksRaw();

  return nullptr;
}

PJRT_Error *
onExecutableOutputMemoryKinds(PJRT_Executable_OutputMemoryKinds_Args *args) {
  DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_OutputMemoryKinds");

  ExecutableInstance *executable_instance =
      ExecutableInstance::unwrap(args->executable);
  args->num_outputs =
      executable_instance->getExecutableImage()->getNumOutputs();
  args->memory_kinds =
      executable_instance->getExecutableImage()->getOutputMemoryKinds().data();
  args->memory_kind_sizes = executable_instance->getExecutableImage()
                                ->getOutputMemoryKindsSizes()
                                .data();

  return nullptr;
};

PJRT_Error *onExecutableSerialize(PJRT_Executable_Serialize_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "ExecutableInstance::PJRT_Executable_Serialize");

  const ExecutableInstance *executable_instance =
      ExecutableInstance::unwrap(args->executable);

  // Make a SerializedExecutableInstance.
  const ExecutableImage *executable_image =
      executable_instance->getExecutableImage();
  std::unique_ptr<SerializedExecutableInstance> serialized_executable =
      SerializedExecutableInstance::createInstance(executable_image);

  args->serialized_bytes = reinterpret_cast<const char *>(
      serialized_executable->getSerializedPayload().data());
  args->serialized_bytes_size =
      serialized_executable->getSerializedPayload().size();
  args->serialized_executable_deleter = [](PJRT_SerializedExecutable *exec) {
    delete SerializedExecutableInstance::unwrap(exec);
  };
  args->serialized_executable = *serialized_executable.release();

  return nullptr;
}

} // namespace internal

} // namespace tt::pjrt
