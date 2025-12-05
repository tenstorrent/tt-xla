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

namespace tt::pjrt::tests {

// Specialization of PJRTComponentUnitTestsBase ensures that MemoryInstance
// unit tests will be scoped to their own test suite.
class MemoryInstanceUnitTests : public PJRTComponentUnitTests {};

// Tests successful creation of memory instances with valid parameters.
// Verifies that all provided values are correctly persisted.
// As part of SetUp(), m_default_memory has already been created.
TEST_F(MemoryInstanceUnitTests, createInstance_successCase) {
  ASSERT_NE(m_default_memory, nullptr);
  EXPECT_EQ(m_default_memory->getId(), DEFAULT_MEMORY_ID);
  EXPECT_TRUE(m_default_memory->isHostMemory());
  EXPECT_EQ(m_default_memory->getMemoryKind(),
            MemoryInstance::c_host_memory_kind_name);
  EXPECT_FALSE(m_default_memory->getDebugString().empty());
  EXPECT_EQ(m_default_memory->getAddressableByDevices().size(), 1);
  EXPECT_EQ(m_default_memory->getAddressableByDevices()[0], m_device.get());
}

// Tests casting MemoryInstance to raw PJRT_Memory pointer.
TEST_F(MemoryInstanceUnitTests, castToPJRTMemory) {
  PJRT_Memory *pjrt_mem = *m_default_memory;
  EXPECT_NE(pjrt_mem, nullptr);
  EXPECT_EQ(static_cast<void *>(m_default_memory.get()),
            static_cast<void *>(pjrt_mem));
}

// Tests "unwrapping" raw PJRT_Memory pointer back to MemoryInstance.
// Verifies the unwrapped instance matches the original.
TEST_F(MemoryInstanceUnitTests, unwrapPJRTMemory) {
  PJRT_Memory *pjrt_mem = *m_default_memory;
  MemoryInstance *unwrapped = MemoryInstance::unwrap(pjrt_mem);
  ASSERT_NE(unwrapped, nullptr);
  EXPECT_EQ(unwrapped, m_default_memory.get());
}

// Tests PJRT API for getting memory ID.
TEST_F(MemoryInstanceUnitTests, API_PJRT_Memory_Id) {
  PJRT_Memory_Id_Args args;
  args.struct_size = PJRT_Memory_Id_Args_STRUCT_SIZE;
  args.memory = *m_default_memory;
  args.id = DEFAULT_MEMORY_ID + 1; // intentionally set to a different value

  PJRT_Error *error = internal::onMemoryId(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_EQ(args.id, DEFAULT_MEMORY_ID);
}

// Tests PJRT API for getting memory kind string.
TEST_F(MemoryInstanceUnitTests, API_PJRT_Memory_Kind) {
  PJRT_Memory_Kind_Args args;
  args.struct_size = PJRT_Memory_Kind_Args_STRUCT_SIZE;
  args.memory = *m_default_memory;
  args.kind = nullptr;
  args.kind_size = 0;

  PJRT_Error *error = internal::onMemoryKind(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_NE(args.kind, nullptr);
  EXPECT_GT(args.kind_size, 0);
  EXPECT_EQ(std::string(args.kind, args.kind_size),
            MemoryInstance::c_host_memory_kind_name);
}

// Tests PJRT API for getting memory kind ID.
TEST_F(MemoryInstanceUnitTests, API_PJRT_Memory_Kind_Id) {
  PJRT_Memory_Kind_Id_Args args;
  args.struct_size = PJRT_Memory_Kind_Id_Args_STRUCT_SIZE;
  args.memory = *m_default_memory;
  args.kind_id = MEMORY_KIND_DEVICE; // intentionally different

  PJRT_Error *error = internal::onMemoryKindId(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_EQ(args.kind_id, MEMORY_KIND_HOST);
}

// Tests PJRT API for getting debug string.
TEST_F(MemoryInstanceUnitTests, API_PJRT_Memory_DebugString) {
  PJRT_Memory_DebugString_Args args;
  args.struct_size = PJRT_Memory_DebugString_Args_STRUCT_SIZE;
  args.memory = *m_default_memory;
  args.debug_string = nullptr;
  args.debug_string_size = 0;

  PJRT_Error *error = internal::onMemoryDebugString(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_NE(args.debug_string, nullptr);
  EXPECT_GT(args.debug_string_size, 0);
  EXPECT_EQ(std::string(args.debug_string, args.debug_string_size),
            m_default_memory->getDebugString());
}

// Tests PJRT API for getting the toString output.
TEST_F(MemoryInstanceUnitTests, API_PJRT_Memory_ToString) {
  PJRT_Memory_ToString_Args args;
  args.struct_size = PJRT_Memory_ToString_Args_STRUCT_SIZE;
  args.memory = *m_default_memory;
  args.to_string = nullptr;
  args.to_string_size = 0;

  PJRT_Error *error = internal::onMemoryToString(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_NE(args.to_string, nullptr);
  EXPECT_GT(args.to_string_size, 0);
  EXPECT_EQ(std::string(args.to_string, args.to_string_size),
            m_default_memory->getDebugString());
}

// Tests PJRT API for getting addressable devices.
TEST_F(MemoryInstanceUnitTests, API_PJRT_Memory_AddressableByDevices) {
  PJRT_Memory_AddressableByDevices_Args args;
  args.struct_size = PJRT_Memory_AddressableByDevices_Args_STRUCT_SIZE;
  args.memory = *m_default_memory;
  args.devices = nullptr;
  args.num_devices = 0;

  PJRT_Error *error = internal::onMemoryAddressableByDevices(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_EQ(args.num_devices, 1);
  EXPECT_NE(args.devices, nullptr);
  EXPECT_EQ(DeviceInstance::unwrap(args.devices[0]), m_device.get());
}

} // namespace tt::pjrt::tests
