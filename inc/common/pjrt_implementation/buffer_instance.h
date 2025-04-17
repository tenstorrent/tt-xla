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

// Represents PJRT_Buffer structure and the functionality around it. Wraps
// `tt::runtime::Tensor` underneath. `PJRT_Buffer` was designed to represent
// device buffers, but our runtime tensor will always be the host tensor,
// created either by copying from host memory or by copying from device. We do
// the transfer to device memory at the beginning of
// `PJRT_LoadedExecutable_Execute` function, and transfer from device memory at
// the end. The reason is that we can't know the layout in which transfer needs
// to be done until compilation is done. Theoretically we could update the host
// tensor with device tensor during first execute and cache it for later
// executions which use the same input buffer, but it would cause issues if
// multiple executions are running simultaneously and they update the buffer
// with different device tensors.
class BufferInstance {
public:
  // Creates new buffer instance for input buffer.
  static std::unique_ptr<BufferInstance>
  createInputBufferInstance(PJRT_Buffer_Type data_type,
                            const std::int64_t *dims, size_t num_dims,
                            DeviceInstance *device);

  // Creates new buffer instance for output buffer.
  static std::unique_ptr<BufferInstance>
  createOutputBufferInstance(const tt::runtime::Tensor &tensor,
                             std::vector<std::uint32_t> &&dimensions,
                             DeviceInstance *device);

  // Destructor, deletes buffer data if not already deleted.
  ~BufferInstance();

  // Binds PJRT API functions implementation related to PJRT_Buffer structure.
  static void bindApi(PJRT_Api *api);

  // Casts this buffer instance to PJRT_Buffer pointer.
  operator PJRT_Buffer *() { return reinterpret_cast<PJRT_Buffer *>(this); }

  // Casts the PJRT_Buffer pointer to BufferInstance pointer.
  static BufferInstance *unwrap(PJRT_Buffer *buffer) {
    return reinterpret_cast<BufferInstance *>(buffer);
  }

  // Returns buffer's data type.
  PJRT_Buffer_Type getDataType() const { return m_data_type; }

  // Returns raw pointer to buffer's dimensions.
  const int64_t *getDimensionsRaw() const { return m_dimensions.data(); }

  // Returns number of buffer's dimensions.
  size_t getNumberOfDimensions() const { return m_dimensions.size(); }

  // Returns device instance on which this buffer resides.
  DeviceInstance *getDevice() { return m_device; }

  // Returns const device instance on which this buffer resides.
  const DeviceInstance *getDevice() const { return m_device; }

  // Returns the underlying runtime tensor created for this buffer.
  const tt::runtime::Tensor &getRuntimeTensor() const {
    return m_runtime_tensor;
  }

  // Returns the size of the underlying runtime tensor, in bytes.
  size_t getRuntimeTensorSize() const;

  // Returns true if the buffer data was deleted, i.e. its underlying tensor was
  // deallocated.
  bool isDataDeleted();

  // Deletes the buffer data.
  void deleteData();

  // This method should asynchronously copy data into device buffer from the
  // given host buffer. Currently our runtime expects all input buffers to be on
  // host and to be copied to device during execution, because it needs to read
  // from compiled flatbuffer how to do device transfers. That's why we create
  // host runtime tensor from the given host buffer.
  void copyFromHost(const void *host_buffer, PJRT_Buffer_Type data_type,
                    const std::int64_t *dims, size_t num_dims,
                    const std::int64_t *byte_strides, size_t num_byte_strides,
                    PJRT_HostBufferSemantics host_buffer_semantics,
                    EventInstance **out_done_with_host_buffer_event);

  // Asynchronously copies the buffer's data into a preallocated host buffer.
  tt_pjrt_status copyToHost(void *host_buffer, size_t host_buffer_size,
                            EventInstance **out_copy_done_event);

  // Sets that buffer data is ready (transferred from host or computed on
  // device) and marks data ready event as ready (if it is already created).
  void markAsDataReady();

  // Creates data ready event. Returns error status if data ready event was
  // already created for this buffer.
  tt_pjrt_status createDataReadyEvent(EventInstance **out_event);

private:
  // Constructor used for the input buffers.
  BufferInstance(PJRT_Buffer_Type data_type, const std::int64_t *dims,
                 size_t num_dims, DeviceInstance *device);

  // Constructor used for the output buffers.
  BufferInstance(const tt::runtime::Tensor &tensor,
                 const std::vector<std::uint32_t> &dimensions,
                 DeviceInstance *device);

  // Calculates required tensor shape.
  static std::vector<std::uint32_t> calculateShape(const std::int64_t *dims,
                                                   size_t num_dims);

  // Calculates required tensor strides.
  static std::vector<std::uint32_t>
  calculateStrides(size_t num_dims, const std::int64_t *byte_strides,
                   size_t num_byte_strides, std::uint32_t element_size);

  // Buffer's data type.
  PJRT_Buffer_Type m_data_type;

  // Buffer's dimensions. Shouldn't be changed after construction because client
  // might depend on the raw pointer to these dimensions.
  const std::vector<std::int64_t> m_dimensions;

  // Device instance on which this buffer resides.
  DeviceInstance *m_device;

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
  // transferred to the client.
  EventInstance *m_data_ready_event;

  // In case this buffer is created by copying from host and we need to notify
  // caller to free the host memory when we are done, this event will be set.
  EventInstance *m_done_with_host_buffer_event;

  // True if the buffer data was deleted, i.e. its underlying tensor was
  // deallocated.
  bool m_data_deleted;

  // Mutex guarding buffer data deletion.
  std::mutex m_data_deleted_mutex;
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

// Implements PJRT_Buffer_IsOnCpu API function.
PJRT_Error *onBufferIsOnCpu(PJRT_Buffer_IsOnCpu_Args *args);

// Implements PJRT_Buffer_Device API function.
PJRT_Error *onBufferDevice(PJRT_Buffer_Device_Args *args);

// Implements PJRT_Buffer_ReadyEvent API function.
PJRT_Error *onBufferReadyEvent(PJRT_Buffer_ReadyEvent_Args *args);

} // namespace internal

} // namespace tt::pjrt

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_BUFFER_INSTANCE_H_
