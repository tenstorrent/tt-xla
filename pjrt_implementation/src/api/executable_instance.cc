// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "api/executable_instance.h"

// c++ standard library includes
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

// tt-xla includes
#include "api/client_instance.h"
#include "api/error_instance.h"
#include "api/module_builder/module_builder.h"
#include "api/serialized_executable_instance.h"
#include "utils/logging.h"
#include "utils/status.h"

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
  DLOG_F(LOG_DEBUG, "ExecutableInstance::PJRT_Executable_Destroy");

  delete ExecutableInstance::unwrap(args->executable);

  return nullptr;
}

PJRT_Error *onExecutableName(PJRT_Executable_Name_Args *args) {
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
  DLOG_F(LOG_DEBUG, "ExecutableInstance::PJRT_Executable_NumReplicas");

  ExecutableInstance *executable_instance =
      ExecutableInstance::unwrap(args->executable);

  args->num_replicas =
      executable_instance->getExecutableImage()->getNumReplicas();

  return nullptr;
}

PJRT_Error *
onExecutableNumPartitions(PJRT_Executable_NumPartitions_Args *args) {
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

  // return object
  PJRT_Program *program = args->program;
  program->format = module_builder::c_mlir_format_name.data();
  program->format_size = module_builder::c_mlir_format_name.size();

  const std::string &original_mlir_code =
      executable_instance->getExecutableImage()->getOriginalMlirCode();

  // Check if we should use cursed.mlir (only if CONVERT_SHLO_TO_SHARDY=1 and file exists)
  const char* convert_env = std::getenv("CONVERT_SHLO_TO_SHARDY");
  bool use_cursed_mlir = (convert_env != nullptr && std::string(convert_env) == "1");

  // Check if we should use sanitized MLIR code for XLA ingestion
  const char* sanitized_env = std::getenv("USE_SANITIZED_EMITHLO_IR");
  bool use_sanitized_emithlo_ir = (sanitized_env != nullptr && std::string(sanitized_env) == "1");

  // Read MLIR code from file (cached after first read)
  static std::string literal_mlir_code;
  static bool file_read = false;
  static bool file_exists = false;

  // Determine which MLIR code to use
  const std::string *checkpointed_mlir_code_ptr = &original_mlir_code;

  if (use_sanitized_emithlo_ir) {
    // Use sanitized MLIR code cleaned for XLA ingestion
    const std::string &sanitized_mlir_code =
        executable_instance->getExecutableImage()->getSanitizedMlirCode();
    checkpointed_mlir_code_ptr = &sanitized_mlir_code;
    DLOG_F(LOG_DEBUG, "USE_SANITIZED_EMITHLO_IR=1, using sanitized MLIR code");
  } else if (use_cursed_mlir) {
    // Only try to read file if we haven't read it yet
    if (!file_read) {
      const char* pjrt_dir = std::getenv("TTXLA_PJRT_DIR");
      std::string file_path;

      if (pjrt_dir) {
        file_path = std::string(pjrt_dir) + "/test_data/cursed.mlir";
      } else {
        // Default path relative to source tree
        file_path = "pjrt_implementation/test_data/cursed.mlir";
      }

      // Check if file exists before trying to read it
      std::ifstream file(file_path);
      if (file.is_open()) {
        std::stringstream buffer;
        buffer << file.rdbuf();
        literal_mlir_code = buffer.str();
        file_exists = true;
        file_read = true;
        DLOG_F(LOG_DEBUG, "Successfully read MLIR code from: %s (size=%zu bytes)",
               file_path.c_str(), literal_mlir_code.size());
      } else {
        DLOG_F(LOG_DEBUG, "cursed.mlir file not found at path: %s, using original MLIR code",
               file_path.c_str());
        file_exists = false;
        file_read = true;
      }
    }

    // Use cursed.mlir only if file exists
    if (file_exists) {
      checkpointed_mlir_code_ptr = &literal_mlir_code;
    }
  } else {
    DLOG_F(LOG_DEBUG, "CONVERT_SHLO_TO_SHARDY not set to 1, using original MLIR code");
  }

  const std::string &checkpointed_mlir_code = *checkpointed_mlir_code_ptr;

  DLOG_F(LOG_DEBUG, "Literal MLIR code (size=%zu):\n%.*s",
         checkpointed_mlir_code.size(),
         static_cast<int>(checkpointed_mlir_code.size()),
         checkpointed_mlir_code.data());

  size_t code_size = checkpointed_mlir_code.size();

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

    std::memcpy(program->code, checkpointed_mlir_code.data(), code_size);
  }

  return nullptr;
}

PJRT_Error *onExecutableNumOutputs(PJRT_Executable_NumOutputs_Args *args) {
  DLOG_F(LOG_DEBUG, "ExecutableInstance::PJRT_Executable_NumOutputs");

  ExecutableInstance *executable_instance =
      ExecutableInstance::unwrap(args->executable);

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

PJRT_Error *onExecutableFingerprint(PJRT_Executable_Fingerprint_Args *args) {
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
