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

// Specialization of PJRTComponentUnitTestsBase ensures that BufferInstance
// unit tests will be scoped to their own test suite.
class BufferInstanceUnitTests : public PJRTComponentUnitTests {
protected:
  // Helper dimensions for buffer creation.
  static constexpr std::int64_t DEFAULT_DIMS[] = {2, 3, 4};
  static constexpr size_t DEFAULT_NUM_DIMS = 3;
  static constexpr PJRT_Buffer_Type DEFAULT_DATA_TYPE = PJRT_Buffer_Type_F32;

  // Helper to create a default input buffer for testing.
  std::unique_ptr<BufferInstance> createDefaultInputBuffer() {
    return BufferInstance::createInputBufferInstance(
        DEFAULT_DATA_TYPE, DEFAULT_DIMS, DEFAULT_NUM_DIMS, m_device.get(),
        m_default_memory.get());
  }
};

// Tests successful creation of input buffer instances with valid parameters.
TEST_F(BufferInstanceUnitTests, createInputBufferInstance_successCase) {
  auto buffer = createDefaultInputBuffer();
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(buffer->getDataType(), DEFAULT_DATA_TYPE);
  EXPECT_EQ(buffer->getNumberOfDimensions(), DEFAULT_NUM_DIMS);
  EXPECT_EQ(buffer->getDevice(), m_device.get());
  EXPECT_EQ(buffer->getMemory(), m_default_memory.get());
  EXPECT_FALSE(buffer->isDataDeleted());
}

// Tests casting BufferInstance to raw PJRT_Buffer pointer.
TEST_F(BufferInstanceUnitTests, castToPJRTBuffer) {
  auto buffer = createDefaultInputBuffer();
  PJRT_Buffer *pjrt_buffer = *buffer;
  EXPECT_NE(pjrt_buffer, nullptr);
  EXPECT_EQ(static_cast<void *>(buffer.get()),
            static_cast<void *>(pjrt_buffer));
}

// Tests "unwrapping" raw PJRT_Buffer pointer back to BufferInstance.
// Verifies the unwrapped instance matches the original.
TEST_F(BufferInstanceUnitTests, unwrapPJRTBuffer) {
  auto buffer = createDefaultInputBuffer();
  PJRT_Buffer *pjrt_buffer = *buffer;
  BufferInstance *unwrapped = BufferInstance::unwrap(pjrt_buffer);
  ASSERT_NE(unwrapped, nullptr);
  EXPECT_EQ(unwrapped, buffer.get());
}

// Tests that dimensions are correctly stored and retrievable.
TEST_F(BufferInstanceUnitTests, getDimensionsRaw) {
  auto buffer = createDefaultInputBuffer();
  const int64_t *dims = buffer->getDimensionsRaw();
  ASSERT_NE(dims, nullptr);
  for (size_t i = 0; i < DEFAULT_NUM_DIMS; ++i) {
    EXPECT_EQ(dims[i], DEFAULT_DIMS[i]);
  }
}

// Tests that unique ID is assigned to each buffer instance.
TEST_F(BufferInstanceUnitTests, getUID_unique) {
  auto buffer1 = createDefaultInputBuffer();
  auto buffer2 = createDefaultInputBuffer();
  EXPECT_NE(buffer1->getUID(), buffer2->getUID());
}

// Tests deleting buffer data.
TEST_F(BufferInstanceUnitTests, deleteData) {
  auto buffer = createDefaultInputBuffer();
  EXPECT_FALSE(buffer->isDataDeleted());
  buffer->deleteData();
  EXPECT_TRUE(buffer->isDataDeleted());
}

// Tests that deleting data multiple times is safe.
TEST_F(BufferInstanceUnitTests, deleteData_multipleCallsSafe) {
  auto buffer = createDefaultInputBuffer();
  buffer->deleteData();
  buffer->deleteData(); // should not crash
  EXPECT_TRUE(buffer->isDataDeleted());
}

// Tests toShapeStr returns a non-empty string.
TEST_F(BufferInstanceUnitTests, toShapeStr) {
  auto buffer = createDefaultInputBuffer();
  std::string shape_str = buffer->toShapeStr();
  EXPECT_FALSE(shape_str.empty());
}

// Tests PJRT API for getting buffer element type.
TEST_F(BufferInstanceUnitTests, API_PJRT_Buffer_ElementType) {
  auto buffer = createDefaultInputBuffer();

  PJRT_Buffer_ElementType_Args args;
  args.struct_size = PJRT_Buffer_ElementType_Args_STRUCT_SIZE;
  args.buffer = *buffer;
  args.type = PJRT_Buffer_Type_INVALID; // intentionally different

  PJRT_Error *error = internal::onBufferElementType(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_EQ(args.type, DEFAULT_DATA_TYPE);
}

// Tests PJRT API for getting buffer dimensions.
TEST_F(BufferInstanceUnitTests, API_PJRT_Buffer_Dimensions) {
  auto buffer = createDefaultInputBuffer();

  PJRT_Buffer_Dimensions_Args args;
  args.struct_size = PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
  args.buffer = *buffer;
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
  auto buffer = createDefaultInputBuffer();

  PJRT_Buffer_UnpaddedDimensions_Args args;
  args.struct_size = PJRT_Buffer_UnpaddedDimensions_Args_STRUCT_SIZE;
  args.buffer = *buffer;
  args.unpadded_dims = nullptr;
  args.num_dims = 0;

  PJRT_Error *error = internal::onBufferUnpaddedDimensions(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_EQ(args.num_dims, DEFAULT_NUM_DIMS);
  ASSERT_NE(args.unpadded_dims, nullptr);
  // Since we don't support dynamic dimensions with padding yet,
  // unpadded_dims should match dims.
  for (size_t i = 0; i < DEFAULT_NUM_DIMS; ++i) {
    EXPECT_EQ(args.unpadded_dims[i], DEFAULT_DIMS[i]);
  }
}

// Tests PJRT API for getting dynamic dimension indices.
TEST_F(BufferInstanceUnitTests, API_PJRT_Buffer_DynamicDimensionIndices) {
  auto buffer = createDefaultInputBuffer();

  PJRT_Buffer_DynamicDimensionIndices_Args args;
  args.struct_size = PJRT_Buffer_DynamicDimensionIndices_Args_STRUCT_SIZE;
  args.buffer = *buffer;

  PJRT_Error *error = internal::onBufferDynamicDimensionIndices(&args);
  ASSERT_EQ(error, nullptr);
  // We don't support dynamic dimensions yet.
  EXPECT_EQ(args.num_dynamic_dims, 0);
  EXPECT_EQ(args.dynamic_dim_indices, nullptr);
}

// Tests PJRT API for deleting buffer.
TEST_F(BufferInstanceUnitTests, API_PJRT_Buffer_Delete) {
  auto buffer = createDefaultInputBuffer();
  EXPECT_FALSE(buffer->isDataDeleted());

  PJRT_Buffer_Delete_Args args;
  args.struct_size = PJRT_Buffer_Delete_Args_STRUCT_SIZE;
  args.buffer = *buffer;

  PJRT_Error *error = internal::onBufferDelete(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_TRUE(buffer->isDataDeleted());
}

// Tests PJRT API for checking if buffer is deleted.
TEST_F(BufferInstanceUnitTests, API_PJRT_Buffer_IsDeleted) {
  auto buffer = createDefaultInputBuffer();

  PJRT_Buffer_IsDeleted_Args args;
  args.struct_size = PJRT_Buffer_IsDeleted_Args_STRUCT_SIZE;
  args.buffer = *buffer;
  args.is_deleted = true; // intentionally different

  PJRT_Error *error = internal::onBufferIsDeleted(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_FALSE(args.is_deleted);

  buffer->deleteData();
  error = internal::onBufferIsDeleted(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_TRUE(args.is_deleted);
}

// Tests PJRT API for checking if buffer is on CPU.
TEST_F(BufferInstanceUnitTests, API_PJRT_Buffer_IsOnCpu) {
  auto buffer = createDefaultInputBuffer();

  PJRT_Buffer_IsOnCpu_Args args;
  args.struct_size = PJRT_Buffer_IsOnCpu_Args_STRUCT_SIZE;
  args.buffer = *buffer;
  args.is_on_cpu = true; // intentionally different

  PJRT_Error *error = internal::onBufferIsOnCpu(&args);
  ASSERT_EQ(error, nullptr);
  // Currently all our inputs are transferred to device where computation runs.
  EXPECT_FALSE(args.is_on_cpu);
}

// Tests PJRT API for getting buffer device.
TEST_F(BufferInstanceUnitTests, API_PJRT_Buffer_Device) {
  auto buffer = createDefaultInputBuffer();

  PJRT_Buffer_Device_Args args;
  args.struct_size = PJRT_Buffer_Device_Args_STRUCT_SIZE;
  args.buffer = *buffer;
  args.device = nullptr;

  PJRT_Error *error = internal::onBufferDevice(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_NE(args.device, nullptr);
  EXPECT_EQ(DeviceInstance::unwrap(args.device), m_device.get());
}

// Tests PJRT API for getting buffer memory.
TEST_F(BufferInstanceUnitTests, API_PJRT_Buffer_Memory) {
  auto buffer = createDefaultInputBuffer();

  PJRT_Buffer_Memory_Args args;
  args.struct_size = PJRT_Buffer_Memory_Args_STRUCT_SIZE;
  args.buffer = *buffer;
  args.memory = nullptr;

  PJRT_Error *error = internal::onBufferMemory(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_NE(args.memory, nullptr);
  EXPECT_EQ(MemoryInstance::unwrap(args.memory), m_default_memory.get());
}

// Tests PJRT API for destroying buffer.
TEST_F(BufferInstanceUnitTests, API_PJRT_Buffer_Destroy) {
  // Create buffer on heap since destroy will delete it.
  auto buffer = createDefaultInputBuffer();
  BufferInstance *buffer_ptr = buffer.release();

  PJRT_Buffer_Destroy_Args args;
  args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
  args.buffer = *buffer_ptr;

  // Should not crash and should properly delete the buffer.
  PJRT_Error *error = internal::onBufferDestroy(&args);
  EXPECT_EQ(error, nullptr);
}

} // namespace tt::pjrt::tests
