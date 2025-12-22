// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

// PJRT test headers
#include "unit_test_utils.h"

// PJRT implementation headers
#include "api/buffer_instance.h"

namespace tt::pjrt::tests {

// Literals used in tests.
static constexpr std::int64_t DEFAULT_DIMS[] = {2, 3, 4};
static constexpr size_t DEFAULT_NUM_DIMS = 3;
static constexpr PJRT_Buffer_Type DEFAULT_DATA_TYPE = PJRT_Buffer_Type_F32;

// Specialization of PJRTComponentUnitTestsBase ensures that BufferInstance
// unit tests will be scoped to their own test suite.
class BufferInstanceUnitTests : public PJRTComponentUnitTests {
protected:
  // Runs before every test.
  void SetUp() override {
    PJRTComponentUnitTests::SetUp();
    m_buffer = createInputBuffer();
  }

  // Helper that creates an input buffer with default arguments.
  std::unique_ptr<BufferInstance> createInputBuffer() {
    return BufferInstance::createInputBufferInstance(
        DEFAULT_DATA_TYPE, DEFAULT_DIMS, DEFAULT_NUM_DIMS, m_device.get(),
        m_default_memory.get());
  }

  // Buffer that is unique per-test.
  std::unique_ptr<BufferInstance> m_buffer;
};

// Tests successful creation of input buffer instances with valid parameters.
TEST_F(BufferInstanceUnitTests, createInputBufferInstance_successCase) {
  ASSERT_NE(m_buffer, nullptr);
  EXPECT_EQ(m_buffer->getDataType(), DEFAULT_DATA_TYPE);
  EXPECT_EQ(m_buffer->getNumberOfDimensions(), DEFAULT_NUM_DIMS);
  EXPECT_EQ(m_buffer->getDevice(), m_device.get());
  EXPECT_EQ(m_buffer->getMemory(), m_default_memory.get());
  EXPECT_FALSE(m_buffer->getHostRuntimeTensor().has_value());
  EXPECT_FALSE(m_buffer->isDataDeleted());
  EXPECT_FALSE(m_buffer->toShapeStr().empty());
}

// Tests casting BufferInstance to raw PJRT_Buffer pointer.
TEST_F(BufferInstanceUnitTests, castToPJRTBuffer) {
  PJRT_Buffer *pjrt_buffer = *m_buffer;
  EXPECT_NE(pjrt_buffer, nullptr);
  EXPECT_EQ(static_cast<void *>(m_buffer.get()),
            static_cast<void *>(pjrt_buffer));
}

// Tests "unwrapping" raw PJRT_Buffer pointer back to BufferInstance.
// Verifies the unwrapped instance matches the original.
TEST_F(BufferInstanceUnitTests, unwrapPJRTBuffer) {
  PJRT_Buffer *pjrt_buffer = *m_buffer;
  const BufferInstance *unwrapped = BufferInstance::unwrap(pjrt_buffer);
  ASSERT_NE(unwrapped, nullptr);
  EXPECT_EQ(unwrapped, m_buffer.get());
}

// Tests that dimensions are correctly stored and retrievable.
TEST_F(BufferInstanceUnitTests, getDimensionsRaw) {
  const int64_t *dims = m_buffer->getDimensionsRaw();
  ASSERT_NE(dims, nullptr);
  for (size_t i = 0; i < DEFAULT_NUM_DIMS; ++i) {
    EXPECT_EQ(dims[i], DEFAULT_DIMS[i]);
  }
}

// Tests that unique ID is assigned to each buffer instance.
TEST_F(BufferInstanceUnitTests, getUID_unique) {
  std::unique_ptr<BufferInstance> another_buffer = createInputBuffer();
  EXPECT_NE(m_buffer->getUID(), another_buffer->getUID());
}

// Tests deleting buffer data.
TEST_F(BufferInstanceUnitTests, deleteData) {
  EXPECT_FALSE(m_buffer->isDataDeleted());
  m_buffer->deleteData();
  EXPECT_TRUE(m_buffer->isDataDeleted());
}

// Tests that deleting data multiple times is safe.
TEST_F(BufferInstanceUnitTests, deleteData_multipleCallsSafe) {
  m_buffer->deleteData();
  m_buffer->deleteData(); // should not crash
  EXPECT_TRUE(m_buffer->isDataDeleted());
}

// Tests PJRT API for getting buffer element type.
TEST_F(BufferInstanceUnitTests, API_PJRT_Buffer_ElementType) {
  PJRT_Buffer_ElementType_Args args;
  args.struct_size = PJRT_Buffer_ElementType_Args_STRUCT_SIZE;
  args.buffer = *m_buffer;
  args.type = PJRT_Buffer_Type_INVALID; // intentionally different

  PJRT_Error *error = internal::onBufferElementType(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_EQ(args.type, DEFAULT_DATA_TYPE);
}

// Tests PJRT API for getting buffer dimensions.
TEST_F(BufferInstanceUnitTests, API_PJRT_Buffer_Dimensions) {
  PJRT_Buffer_Dimensions_Args args;
  args.struct_size = PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
  args.buffer = *m_buffer;
  args.dims = nullptr;
  args.num_dims = 0;

  PJRT_Error *error = internal::onBufferDimensions(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_EQ(args.num_dims, DEFAULT_NUM_DIMS);
  ASSERT_NE(args.dims, nullptr);
  for (size_t i = 0; i < DEFAULT_NUM_DIMS; ++i) {
    EXPECT_EQ(args.dims[i], DEFAULT_DIMS[i]);
  }
}

// Tests PJRT API for getting unpadded dimensions.
TEST_F(BufferInstanceUnitTests, API_PJRT_Buffer_UnpaddedDimensions) {
  PJRT_Buffer_UnpaddedDimensions_Args args;
  args.struct_size = PJRT_Buffer_UnpaddedDimensions_Args_STRUCT_SIZE;
  args.buffer = *m_buffer;
  args.unpadded_dims = nullptr;
  args.num_dims = 0;

  PJRT_Error *error = internal::onBufferUnpaddedDimensions(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_EQ(args.num_dims, DEFAULT_NUM_DIMS);
  ASSERT_NE(args.unpadded_dims, nullptr);
  // DEVNOTE: Dynamic dimensions with padding are not supported.
  for (size_t i = 0; i < DEFAULT_NUM_DIMS; ++i) {
    EXPECT_EQ(args.unpadded_dims[i], DEFAULT_DIMS[i]);
  }
}

// Tests PJRT API for getting dynamic dimension indices.
TEST_F(BufferInstanceUnitTests, API_PJRT_Buffer_DynamicDimensionIndices) {
  PJRT_Buffer_DynamicDimensionIndices_Args args;
  args.struct_size = PJRT_Buffer_DynamicDimensionIndices_Args_STRUCT_SIZE;
  args.buffer = *m_buffer;

  PJRT_Error *error = internal::onBufferDynamicDimensionIndices(&args);
  ASSERT_EQ(error, nullptr);
  // DEVNOTE: Dynamic dimensions are not supported.
  EXPECT_EQ(args.num_dynamic_dims, 0);
  EXPECT_EQ(args.dynamic_dim_indices, nullptr);
}

// Tests PJRT API for data deletion.
TEST_F(BufferInstanceUnitTests, API_PJRT_Buffer_Delete) {
  EXPECT_FALSE(m_buffer->isDataDeleted());

  PJRT_Buffer_Delete_Args args;
  args.struct_size = PJRT_Buffer_Delete_Args_STRUCT_SIZE;
  args.buffer = *m_buffer;

  PJRT_Error *error = internal::onBufferDelete(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_TRUE(m_buffer->isDataDeleted());
}

// Tests PJRT API for checking if buffer data is deleted.
TEST_F(BufferInstanceUnitTests, API_PJRT_Buffer_IsDeleted) {
  PJRT_Buffer_IsDeleted_Args args;
  args.struct_size = PJRT_Buffer_IsDeleted_Args_STRUCT_SIZE;
  args.buffer = *m_buffer;
  args.is_deleted = true; // intentionally different

  PJRT_Error *error = internal::onBufferIsDeleted(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_FALSE(args.is_deleted);

  m_buffer->deleteData();

  error = internal::onBufferIsDeleted(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_TRUE(args.is_deleted);
}

// Tests PJRT API for checking if buffer is on CPU.
TEST_F(BufferInstanceUnitTests, API_PJRT_Buffer_IsOnCpu) {
  PJRT_Buffer_IsOnCpu_Args args;
  args.struct_size = PJRT_Buffer_IsOnCpu_Args_STRUCT_SIZE;
  args.buffer = *m_buffer;
  args.is_on_cpu = true; // intentionally different

  PJRT_Error *error = internal::onBufferIsOnCpu(&args);
  ASSERT_EQ(error, nullptr);
  // Currently all our inputs are transferred to device for computation.
  EXPECT_FALSE(args.is_on_cpu);
}

// Tests PJRT API for getting buffer's owning device.
TEST_F(BufferInstanceUnitTests, API_PJRT_Buffer_Device) {
  PJRT_Buffer_Device_Args args;
  args.struct_size = PJRT_Buffer_Device_Args_STRUCT_SIZE;
  args.buffer = *m_buffer;
  args.device = nullptr;

  PJRT_Error *error = internal::onBufferDevice(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_NE(args.device, nullptr);
  EXPECT_EQ(DeviceInstance::unwrap(args.device), m_device.get());
}

// Tests PJRT API for getting buffer's owning memory.
TEST_F(BufferInstanceUnitTests, API_PJRT_Buffer_Memory) {
  PJRT_Buffer_Memory_Args args;
  args.struct_size = PJRT_Buffer_Memory_Args_STRUCT_SIZE;
  args.buffer = *m_buffer;
  args.memory = nullptr;

  PJRT_Error *error = internal::onBufferMemory(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_NE(args.memory, nullptr);
  EXPECT_EQ(MemoryInstance::unwrap(args.memory), m_default_memory.get());
}

// Tests PJRT API for destroying buffer.
TEST_F(BufferInstanceUnitTests, API_PJRT_Buffer_Destroy) {
  // Release m_buffer since onBufferDestroy will delete it.
  BufferInstance *buffer_ptr = m_buffer.release();

  PJRT_Buffer_Destroy_Args args;
  args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
  args.buffer = *buffer_ptr;

  // Should not crash and should properly delete the buffer.
  PJRT_Error *error = internal::onBufferDestroy(&args);
  ASSERT_EQ(error, nullptr);
}

} // namespace tt::pjrt::tests
