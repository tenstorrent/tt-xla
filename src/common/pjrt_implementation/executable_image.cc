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

#include <memory>
#include <string>

#include "common/pjrt_implementation/utils.h"
#include "common/status.h"

namespace tt::pjrt {

const std::string_view kMlirFormat = "mlir";

ExecutableImage::ExecutableImage(const tt::runtime::Binary &binary,
                                 std::string code,
                                 const std::vector<bool> &is_output_scalar,
                                 size_t num_addressable_devices)
    : m_ref_count(1), m_binary(binary), m_code(code),
      m_arg_count(binary.getProgramInputs(0).size()),
      m_is_output_scalar(is_output_scalar),
      m_num_addressable_devices(num_addressable_devices) {

  std::vector<tt::runtime::TensorDesc> output_specs =
      m_binary.getProgramOutputs(0);
  m_result_count = output_specs.size();

  if (m_result_count != m_is_output_scalar.size()) {
    // TODO: We should throw error instead, otherwise execution will continue
    // and crash later.
    DLOG_F(ERROR,
           "Created flatbuffer binary contains different number of outputs %ld "
           "than expected %ld",
           m_result_count, m_is_output_scalar.size());
  }

  m_output_types.resize(m_result_count);
  m_output_dims.resize(m_result_count);
  for (int i = 0; i < m_result_count; i++) {
    m_output_types[i] = tt::pjrt::utils::convertElementTypeToBufferType(
        output_specs[i].dataType);

    // PJRT expects an empty shape for scalars.
    m_output_dims[i] = m_is_output_scalar[i] ? std::vector<std::uint32_t>()
                                             : output_specs[i].shape;
  }
}

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
    args->num_outputs = exec->get_result_count();
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
  api->PJRT_Executable_OptimizedProgram =
      +[](PJRT_Executable_OptimizedProgram_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_OptimizedProgram");
    ExecutableImage *executable = ExecutableImage::Unwrap(args->executable);
    PJRT_Program *program = args->program;
    program->format = kMlirFormat.data();
    program->format_size = kMlirFormat.size();
    size_t code_size = executable->get_code().size();
    if (program->code == nullptr) {
      program->code_size = code_size;
    } else {
      if (program->code_size < code_size) {
        return ErrorInstance::MakeError(tt_pjrt_status::kInvalidArgument);
      }
      std::memcpy(program->code, executable->get_code().c_str(), code_size);
    }
    return nullptr;
  };
  api->PJRT_Executable_OutputElementTypes =
      +[](PJRT_Executable_OutputElementTypes_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_OutputElementTypes");
    ExecutableImage *exec = ExecutableImage::Unwrap(args->executable);
    // There is a possibility that this method should return unique types, and
    // not a type for every output.
    args->output_types = exec->get_output_types();
    args->num_output_types = exec->num_output_types();
    return nullptr;
  };
  api->PJRT_Executable_OutputDimensions =
      +[](PJRT_Executable_OutputDimensions_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_OutputDimensions_Args");
    ExecutableImage *exec = ExecutableImage::Unwrap(args->executable);

    args->num_outputs = exec->get_result_count();
    exec->get_output_dims_concatenated(&args->dim_sizes, &args->dims);

    return nullptr;
  };
}

const std::vector<std::uint32_t> &
ExecutableImage::get_output_shape(const size_t index) const {
  assert(index < m_output_dims.size() && "Output index out of range");
  return m_output_dims[index];
}

void ExecutableImage::populateOutputDimsConcatenated() {
  size_t num_dims = 0;

  m_output_dim_sizes = std::make_unique<size_t[]>(m_result_count);
  for (size_t i = 0; i < m_result_count; i++) {
    m_output_dim_sizes[i] = m_output_dims[i].size();
    num_dims += m_output_dim_sizes[i];
  }

  m_output_dims_concatenated = std::make_unique<int64_t[]>(num_dims);
  size_t dims_index = 0;
  for (size_t i = 0; i < m_result_count; i++) {
    for (size_t j = 0; j < m_output_dim_sizes[i]; j++) {
      m_output_dims_concatenated[dims_index + j] = m_output_dims[i][j];
    }
    dims_index += m_output_dim_sizes[i];
  }
}

void ExecutableImage::get_output_dims_concatenated(const size_t **dim_sizes,
                                                   const int64_t **dims) {
  if (!areOutputDimsConcatinated()) {
    populateOutputDimsConcatenated();
  }

  *dim_sizes = m_output_dim_sizes.get();
  *dims = m_output_dims_concatenated.get();
}

} // namespace tt::pjrt
