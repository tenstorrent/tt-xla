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
#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

// tt-mlir includes
#include "tt/runtime/runtime.h"

// tt-xla includes
#include "api/event_instance.h"
#include "utils/status.h"

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_BUFFER_INSTANCE_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_BUFFER_INSTANCE_H_

namespace tt::pjrt {

class ClientInstance;
class DeviceInstance;
class MemoryInstance;

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
  static std::unique_ptr<BufferInstance> createInputBufferInstance(
      PJRT_Buffer_Type data_type, const std::int64_t *dims, size_t num_dims,
      DeviceInstance *device, MemoryInstance *memory, ClientInstance *client);

  // Creates new buffer instance for output buffer.
  static std::unique_ptr<BufferInstance>
  createOutputBufferInstance(const tt::runtime::Tensor &device_tensor,
                             std::vector<std::uint32_t> &&dimensions,
                             DeviceInstance *device, MemoryInstance *memory,
                             PJRT_Buffer_Type data_type, uint32_t device_id,
                             ClientInstance *client);

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

  // Returns the underlying host runtime tensor created for this buffer.
  const std::optional<tt::runtime::Tensor> &getHostRuntimeTensor() const {
    return m_host_runtime_tensor;
  }

  // Returns the prepared runtime tensor created for this buffer on last
  // execution.
  const std::optional<tt::runtime::Tensor> &getPreparedTensor() const {
    return m_prepared_runtime_tensor;
  }

  // Sets the prepared runtime tensor created for this buffer.
  void setPreparedTensor(const tt::runtime::Tensor &tensor) {
    m_prepared_runtime_tensor = tensor;
  }

  // Sets the host runtime tensor for this buffer.
  void setHostRuntimeTensor(const tt::runtime::Tensor &tensor) {
    m_host_runtime_tensor = tensor;
  }

  // Clears the prepared runtime tensor.
  void clearPreparedTensor() { m_prepared_runtime_tensor = std::nullopt; }

  // Returns the memory instance on which this buffers resides.
  MemoryInstance *getMemory() { return m_memory; }

  // Returns the unique identifier for this buffer instance.
  uint64_t getUID() const { return m_uid; }

  // Returns the size of the tensor in the data type that the host expects.
  // This is since some PJRT_Buffer_Type's do not have a supported equivalent in
  // runtime/ttnn. And so, the true data type of the runtime tensor may be
  // different than what the host expects, and will be casted to the hosts
  // expected data type when copying to host, possibly leading to a different
  // size. This function will calculate the converted runtime tensor size to be
  // tensor_volume * expected_host_data_type_element_size
  static size_t getConvertedRuntimeTensorSize(const tt::runtime::Tensor &tensor,
                                              PJRT_Buffer_Type data_type);

  // Returns a string representation of the buffer's shape in the format
  // [d1,d2,d3,...].
  std::string toShapeStr() const;

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

  // Asynchronously copies this buffer's data into a preallocated host buffer.
  tt_pjrt_status copyToHost(void *host_buffer, size_t host_buffer_size,
                            EventInstance **out_copy_done_event);

  // Copies this buffer's data to the device and its memory specified in the
  // arguments.
  tt_pjrt_status copyToDeviceMemory(DeviceInstance *dst_device,
                                    MemoryInstance *dst_memory,
                                    BufferInstance **dst_buffer);

  // Sets that buffer data is ready (transferred from host or computed on
  // device) and marks data ready event as ready (if it is already created).
  void markAsDataReady();

  // Creates data ready event. Returns error status if data ready event was
  // already created for this buffer.
  tt_pjrt_status createDataReadyEvent(EventInstance **out_event);

  // Returns buffer's device id relative to mesh on which a output shard resides
  std::optional<uint32_t> getDeviceId() const { return m_device_id; }

private:
  // Constructor used for the input buffers.
  BufferInstance(PJRT_Buffer_Type data_type, const std::int64_t *dims,
                 size_t num_dims, DeviceInstance *device,
                 MemoryInstance *memory, ClientInstance *client);

  // Constructor used for the output buffers.
  BufferInstance(
      const std::vector<std::uint32_t> &dimensions, DeviceInstance *device,
      MemoryInstance *memory, PJRT_Buffer_Type data_type,
      ClientInstance *client,
      const std::optional<tt::runtime::Tensor> &host_tensor = std::nullopt,
      const std::optional<tt::runtime::Tensor> &device_tensor = std::nullopt,
      std::optional<uint32_t> device_id = std::nullopt);

  // Copies the tensor inside the src_buffer to the tensor of this buffer.
  // Currently only used for device to device transfer in copy construction
  // of new buffer instance.
  void copyFromBuffer(const BufferInstance *src_buffer);

  // Calculates required tensor shape.
  static std::vector<std::uint32_t> calculateShape(const std::int64_t *dims,
                                                   size_t num_dims);

  // Calculates required tensor strides.
  static std::vector<std::uint32_t>
  calculateStrides(size_t num_dims, const std::int64_t *byte_strides,
                   size_t num_byte_strides, std::uint32_t element_size);

  // Gets next UID for buffer instances, used in buffer instance constructor
  // to assign unique identifier to each buffer instance.
  static uint64_t nextUID() {
    static std::atomic<uint64_t> uid{0};
    return uid.fetch_add(1, std::memory_order_relaxed);
  }

  // Unique identifier for this buffer instance.
  const uint64_t m_uid;

  // Buffer's data type.
  PJRT_Buffer_Type m_data_type;

  // Buffer's dimensions. Shouldn't be changed after construction because client
  // might depend on the raw pointer to these dimensions.
  const std::vector<std::int64_t> m_dimensions;

  // Device instance on which this buffer resides.
  DeviceInstance *m_device;

  // Client instance that owns this buffer, used for buffer tracking.
  // Can be nullptr if buffer is created without client tracking.
  ClientInstance *m_client;

  // Device index relative to mesh on which a output shard resides
  const std::optional<uint32_t> m_device_id;

  // Memory on which this buffer resides, Can be nullptr if buffer is created
  // via `PJRT_Client_BufferFromHostBuffer_Args` and memory was not specified.
  MemoryInstance *m_memory;

  // Underlying host runtime tensor created for this buffer. Unlike the prepared
  // tensor this one is guaranteed to be on host and is exactly the shard that
  // the buffer instance represents.
  std::optional<tt::runtime::Tensor> m_host_runtime_tensor;

  // Prepared runtime tensor created for this buffer on last execution. If this
  // buffer is used in multiple programs, it will be the tensor prepared for the
  // last program executed. If this buffer instance is a part of a multi-device
  // tensor, this field contains the full tensor.
  std::optional<tt::runtime::Tensor> m_prepared_runtime_tensor;

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

  // Thread for copying data to host.
  std::unique_ptr<std::thread> m_copy_to_host_thread;

  // Mutex guarding thread spawning for copying data to host.
  // Prevents multiple threads from concurrently copying into the same
  // BufferInstance.
  std::mutex m_copy_to_host_thread_mutex;

  // Mutex guarding internal copy to host operation on the same mesh device
  // Metal+Program Cache is not thread safe when untilizing on device, so
  //  even different bufferInstances may not be concurrently copied to host.
  static std::mutex s_copy_to_host_internal_mutex;
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

// Implements PJRT_Buffer_CopyToDevice API function.
PJRT_Error *onBufferCopyToDevice(PJRT_Buffer_CopyToDevice_Args *args);

// Implements PJRT_Buffer_CopyToMemory API function.
PJRT_Error *onBufferCopyToMemory(PJRT_Buffer_CopyToMemory_Args *args);

// Implements PJRT_Buffer_IsOnCpu API function.
PJRT_Error *onBufferIsOnCpu(PJRT_Buffer_IsOnCpu_Args *args);

// Implements PJRT_Buffer_Device API function.
PJRT_Error *onBufferDevice(PJRT_Buffer_Device_Args *args);

// Implements PJRT_Buffer_Memory API function.
PJRT_Error *onBufferMemory(PJRT_Buffer_Memory_Args *args);

// Implements PJRT_Buffer_ReadyEvent API function.
PJRT_Error *onBufferReadyEvent(PJRT_Buffer_ReadyEvent_Args *args);

} // namespace internal

} // namespace tt::pjrt

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_BUFFER_INSTANCE_H_
