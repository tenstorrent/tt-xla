// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

// c++ standard library includes
#include <memory>

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

// tt-xla includes
#include "api/executable_image.h"

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_EXECUTABLE_INSTANCE_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_EXECUTABLE_INSTANCE_H_

namespace tt::pjrt {

// Represents `PJRT_Executable` structure and the functionality around it. It
// knows how to take a compiled executable image and execution options and
// serialize/deserialize them so an executable can be stored and loaded as
// needed.
class ExecutableInstance {
public:
  // Creates new executable instance.
  static std::unique_ptr<ExecutableInstance>
  createInstance(std::shared_ptr<ExecutableImage> executable_image);

  // Binds PJRT API functions implementation related to PJRT_Buffer structure.
  static void bindApi(PJRT_Api *api);

  // Casts this executable instance to PJRT_Executable pointer.
  operator PJRT_Executable *() {
    return reinterpret_cast<PJRT_Executable *>(this);
  }

  // Casts the PJRT_Executable pointer to ExecutableInstance pointer.
  static ExecutableInstance *unwrap(PJRT_Executable *executable) {
    return reinterpret_cast<ExecutableInstance *>(executable);
  }

  // Casts the const PJRT_Executable pointer to const ExecutableInstance
  // pointer.
  static const ExecutableInstance *unwrap(const PJRT_Executable *executable) {
    return reinterpret_cast<const ExecutableInstance *>(executable);
  }

  // Returns pointer to the underlying executable image.
  ExecutableImage *getExecutableImage() { return m_executable_image.get(); }

  // Returns const pointer to the underlying executable image.
  const ExecutableImage *getExecutableImage() const {
    return m_executable_image.get();
  }

private:
  // Constructs executable instance from the compiled executable image.
  ExecutableInstance(std::shared_ptr<ExecutableImage> executable_image)
      : m_executable_image(std::move(executable_image)) {}

  // Executable image which is shared between executable and loaded executable
  // instances.
  std::shared_ptr<ExecutableImage> m_executable_image;
};

namespace internal {

// Implements PJRT_Executable_Destroy API function.
PJRT_Error *onExecutableDestroy(PJRT_Executable_Destroy_Args *args);

// Implements PJRT_Executable_Name API function.
PJRT_Error *onExecutableName(PJRT_Executable_Name_Args *args);

// Implements PJRT_Executable_NumReplicas API function.
PJRT_Error *onExecutableNumReplicas(PJRT_Executable_NumReplicas_Args *args);

// Implements PJRT_Executable_NumPartitions API function.
PJRT_Error *onExecutableNumPartitions(PJRT_Executable_NumPartitions_Args *args);

// Implements PJRT_Executable_OptimizedProgram API function.
PJRT_Error *
onExecutableOptimizedProgram(PJRT_Executable_OptimizedProgram_Args *args);

// Implements PJRT_Executable_NumOutputs API function.
PJRT_Error *onExecutableNumOutputs(PJRT_Executable_NumOutputs_Args *args);

// Implements PJRT_Executable_SizeOfGeneratedCodeInBytes API function.
PJRT_Error *onExecutableSizeOfGeneratedCodeInBytes(
    PJRT_Executable_SizeOfGeneratedCodeInBytes_Args *args);

// Implements PJRT_Executable_OutputElementTypes API function.
PJRT_Error *
onExecutableOutputElementTypes(PJRT_Executable_OutputElementTypes_Args *args);

// Implements PJRT_Executable_OutputDimensions API function.
PJRT_Error *
onExecutableOutputDimensions(PJRT_Executable_OutputDimensions_Args *args);

// Implements PJRT_Executable_OutputMemoryKinds API function.
PJRT_Error *
onExecutableOutputMemoryKinds(PJRT_Executable_OutputMemoryKinds_Args *args);

// Implements PJRT_Executable_Serialize API function.
PJRT_Error *onExecutableSerialize(PJRT_Executable_Serialize_Args *args);

// Implements PJRT_Executable_Fingerprint API function.
PJRT_Error *onExecutableFingerprint(PJRT_Executable_Fingerprint_Args *args);

} // namespace internal

} // namespace tt::pjrt

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_EXECUTABLE_INSTANCE_H_
