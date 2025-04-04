// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

// c++ standard library includes
// TODO_OOM: memory can be removed once the API changes
#include <memory>
#include <mutex>
#include <vector>

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

// tt-mlir includes
#include "tt/runtime/runtime.h"

// tt-xla includes
#include "common/pjrt_implementation/event_instance.h"
#include "common/status.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_BUFFER_INSTANCE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_BUFFER_INSTANCE_H_

namespace tt::pjrt {

class DeviceInstance;

// Represents PJRT_Buffer structure (device buffer) and the functionality
// around it. Wraps `tt::runtime::Tensor` underneath.
class BufferInstance {
public:
  BufferInstance(DeviceInstance &device, tt::runtime::Tensor &tensor,
                 const std::vector<std::uint32_t> &shape,
                 const std::vector<std::uint32_t> &stride,
                 std::pair<tt::target::DataType, size_t> tt_buffer_type);

  BufferInstance(DeviceInstance &device, tt::runtime::Tensor &tensor,
                 const std::vector<std::uint32_t> &shape,
                 const std::vector<std::uint32_t> &stride,
                 std::pair<tt::target::DataType, size_t> tt_buffer_type,
                 std::shared_ptr<void> host_buffer_ptr);

  // Destructor, deletes buffer data if not already deleted.
  ~BufferInstance();

  // Binds PJRT API functions implementation related to PJRT_Buffer structure.
  static void bindApi(PJRT_Api *api);

  // Casts this buffer instance to PJRT_Buffer and returns pointer to it.
  operator PJRT_Buffer *() { return reinterpret_cast<PJRT_Buffer *>(this); }

  // Casts the PJRT_Buffer pointer to BufferInstance pointer.
  static BufferInstance *unwrap(PJRT_Buffer *buffer) {
    return reinterpret_cast<BufferInstance *>(buffer);
  }

  // Returns buffer's data type.
  PJRT_Buffer_Type getDataType() const { return m_data_type; }

  // Returns raw pointer to buffer's dimensions.
  const int64_t *getRawDimensions() const { return m_dimensions.data(); }

  // Returns number of buffer's dimensions.
  size_t getNumberOfDimensions() const { return m_dimensions.size(); }

  // Returns device instance on which this buffer resides.
  const DeviceInstance *getDevice() const { return m_device; }

  // Returns the underlying runtime tensor created for this buffer.
  const tt::runtime::Tensor &getTensor() const { return m_runtime_tensor; }

  // Returns the size of the underlying runtime tensor, in bytes.
  size_t getRuntimeTensorSize() const;

  // TODO_OOM: Change this description once the runtime API is changed.
  // This method should asynchronously copy data into the buffer from the given
  // host buffer. Currently our runtime expects input buffers to be copied
  // during execution so we cannot do the copy to device buffer here and we just
  // keep a hold of the host buffer until execution, either by making a copy of
  // the host buffer when the given semantic is `ImmutableOnlyDuringCall`, or by
  // simply aliasing it (keeping the pointer to it) when the semantic is some
  // other.
  tt_pjrt_status copyFromHost();

  // TODO_OOM: Remove once the runtime API is changed.
  // Returns pointer to the host buffer from which the underlying tensor is
  // created.
  void *getHostBuffer();

  // Returns true if the buffer data was deleted, i.e. its underlying tensor was
  // deallocated.
  bool isDataDeleted();

  // Delete the buffer data
  void deleteData();

  // DELETE SECTION-------------------------------------------------------------
  bool is_on_cpu() {
    // TODO: Plumb through an indication if running on CPU and then implement
    // the hook to get an unsafe pointer (avoids a copy).
    return false;
  }

  std::vector<std::uint32_t> getDimensions() const {
    return std::vector<std::uint32_t>(dims_.begin(), dims_.end());
  }
  void setType(PJRT_Buffer_Type Type) { DataType = Type; }
  const std::vector<std::uint32_t> &get_stride() const { return stride_; }
  std::pair<tt::target::DataType, size_t> get_tt_buffer_type() const {
    return tt_buffer_type_;
  }
  // DELETE SECTION-------------------------------------------------------------

private:
  // Asynchronously copies the buffer's data into a preallocated host buffer.
  tt_pjrt_status copyToHost(void *host_buffer, size_t host_buffer_size,
                            EventInstance **out_event);

  // DELETE SECTION-------------------------------------------------------------
  // API elements that must have the same lifetime as BufferInstance.
  std::vector<int64_t> dims_;
  std::vector<std::uint32_t> stride_;
  std::pair<tt::target::DataType, size_t> tt_buffer_type_;

  std::vector<int64_t> minor_to_major_;
  std::vector<int64_t> tile_dims_;
  std::vector<size_t> tile_dim_sizes_;
  // DELETE SECTION-------------------------------------------------------------

  // TODO_OOM: Set in constructor. Check other fields too!
  // Buffer's data type.
  PJRT_Buffer_Type m_data_type;

  // Buffer's dimensions. Shouldn't be changed after construction because client
  // might depend on the raw pointer to these dimensions.
  const std::vector<int64_t> m_dimensions;

  // Device instance on which this buffer resides.
  const DeviceInstance *m_device;

  // Underlying runtime tensor created for this buffer.
  tt::runtime::Tensor m_runtime_tensor;

  // True if data in the buffer is ready (transferred from host or computed on
  // device).
  bool m_data_ready;

  // Mutex guarding buffer data state changes.
  std::mutex m_data_ready_mutex;

  // Event that is triggered when the data in the buffer becomes ready. It will
  // be created only if the buffer isn't yet ready at the moment when the client
  // requests the event with PJRT_Buffer_ReadyEvent and its ownership is
  // transferred to the client. This request happens only for output buffers.
  EventInstance *m_data_ready_event;

  // True if the buffer data was deleted, i.e. its underlying tensor was
  // deallocated.
  bool m_data_deleted;

  // Mutex guarding buffer data deletion.
  std::mutex m_data_deleted_mutex;

  // TODO(mrakita): Remove these two fields below once the runtime API changes
  // are uplifted: https://github.com/tenstorrent/tt-mlir/issues/2757

  // In case when input host buffer has a semantic `ImmutableOnlyDuringCall`
  // then we have to make a copy of the buffer and create tensor from that copy
  // instead of aliasing the host buffer directly. Unique ptr is created with a
  // deleter to free the memory on buffer destruction. In JAX this semantic is
  // used only for copying scalars and numpy arrays.
  std::unique_ptr<void> m_host_buffer_copy;

  // In case when input host buffer has a semantic other then the
  // `ImmutableOnlyDuringCall` then we can just alias the host buffer i.e.
  // create tensor directly from it.
  void *m_aliased_host_buffer;
};

namespace internal {

// Implements PJRT_Buffer_Destroy API function.
PJRT_Error *onBufferDestroy(PJRT_Buffer_Destroy_Args *args);

// Implements PJRT_Buffer_ElementType API function.
PJRT_Error *onBufferElementType(PJRT_Buffer_ElementType_Args *args);

// Implements PJRT_Buffer_Dimensions API function.
PJRT_Error *onBufferDimensions(PJRT_Buffer_Dimensions_Args *args);

// Implements PJRT_Buffer_UnpaddedDimensions API function.
PJRT_Error *
onBufferUnpaddedDimensions(PJRT_Buffer_UnpaddedDimensions_Args *args);

// Implements PJRT_Buffer_DynamicDimensionIndices API function.
PJRT_Error *
onBufferDynamicDimensionIndices(PJRT_Buffer_DynamicDimensionIndices_Args *args);

// Implements PJRT_Buffer_ToHostBuffer API function.
PJRT_Error *onBufferToHostBuffer(PJRT_Buffer_ToHostBuffer_Args *args);

// Implements PJRT_Buffer_Delete API function.
PJRT_Error *onBufferDelete(PJRT_Buffer_Delete_Args *args);

// Implements PJRT_Buffer_IsDeleted API function.
PJRT_Error *onBufferIsDeleted(PJRT_Buffer_IsDeleted_Args *args);

// Implements PJRT_Buffer_ReadyEvent API function.
PJRT_Error *onBufferReadyEvent(PJRT_Buffer_ReadyEvent_Args *args);

} // namespace internal

} // namespace tt::pjrt

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_BUFFER_INSTANCE_H_
