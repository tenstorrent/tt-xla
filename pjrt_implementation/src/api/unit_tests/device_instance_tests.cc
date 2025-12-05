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

// Specialization of PJRTComponentUnitTestsBase ensures that DeviceInstance
// unit tests will be scoped to their own test suite.
class DeviceInstanceUnitTests : public PJRTComponentUnitTests {};

// Tests successful creation of device instances with valid parameters.
// Verifies that all provided values are correctly persisted.
// As part of SetUp(), m_device has already been created.
TEST_F(DeviceInstanceUnitTests, createInstance_successCase) {
  ASSERT_NE(m_device, nullptr);
  EXPECT_EQ(m_device->getGlobalDeviceId(), TT_DEVICE_HOST);
  EXPECT_TRUE(m_device->isAddressable());
  EXPECT_EQ(m_device->getLocalDeviceId(), LOCAL_HARDWARE_ID_UNDEFINED);
  EXPECT_EQ(m_device->getDefaultMemory(), nullptr);
  EXPECT_TRUE(m_device->getAddressableMemories().empty());
}

// Tests casting DeviceInstance to raw PJRT_Device pointer.
TEST_F(DeviceInstanceUnitTests, castToPJRTDevice) {
  PJRT_Device *pjrt_dvc = *m_device;
  EXPECT_NE(pjrt_dvc, nullptr);
  EXPECT_EQ(static_cast<void *>(m_device.get()), static_cast<void *>(pjrt_dvc));
}

// Tests "unwrapping" raw PJRT_Device pointer back to DeviceInstance.
// Verifies the unwrapped instance matches the original.
TEST_F(DeviceInstanceUnitTests, unwrapPJRTDevice) {
  PJRT_Device *pjrt_dvc = *m_device;
  const DeviceInstance *unwrapped = DeviceInstance::unwrap(pjrt_dvc);
  ASSERT_NE(unwrapped, nullptr);
  EXPECT_EQ(unwrapped, m_device.get());
}

// Tests the assignment of default memory.
// TODO(acicovic): Should the test also validate getAddressableMemories()?
TEST_F(DeviceInstanceUnitTests, setDefaultMemory) {
  m_device->setDefaultMemory(m_default_memory.get());
  EXPECT_EQ(m_device->getDefaultMemory(), m_default_memory.get());
}

// Tests adding a memory to the addressable memories list.
// DEVNOTE: Uses m_default_memory for this because it is convenient.
TEST_F(DeviceInstanceUnitTests, addAddressableMemory) {
  m_device->addAddressableMemory(m_default_memory.get());
  const std::vector<MemoryInstance *> &v = m_device->getAddressableMemories();
  ASSERT_EQ(v.size(), 1);
  EXPECT_EQ(v[0], m_default_memory.get());
}

// Tests PJRT API for getting device description.
TEST_F(DeviceInstanceUnitTests, API_PJRT_Device_GetDescription) {
  PJRT_Device_GetDescription_Args args;
  args.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
  args.device = *m_device;
  args.device_description = nullptr;

  PJRT_Error *error = internal::onDeviceGetDescription(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_NE(args.device_description, nullptr);
  DeviceDescription *desc = DeviceDescription::unwrap(args.device_description);
  EXPECT_EQ(desc->getDeviceId(), TT_DEVICE_HOST);
}

// Tests PJRT API for checking if device is addressable.
TEST_F(DeviceInstanceUnitTests, API_PJRT_Device_IsAddressable) {
  PJRT_Device_IsAddressable_Args args;
  args.struct_size = PJRT_Device_IsAddressable_Args_STRUCT_SIZE;
  args.device = *m_device;
  args.is_addressable = false; // intentionally different

  PJRT_Error *error = internal::onDeviceIsAddressable(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_TRUE(args.is_addressable);
}

// Tests PJRT API for getting local hardware ID.
TEST_F(DeviceInstanceUnitTests, API_PJRT_Device_LocalHardwareId) {
  PJRT_Device_LocalHardwareId_Args args;
  args.struct_size = PJRT_Device_LocalHardwareId_Args_STRUCT_SIZE;
  args.device = *m_device;
  args.local_hardware_id = LOCAL_HARDWARE_ID_UNDEFINED + 1; // intentional

  PJRT_Error *error = internal::onDeviceLocalHardwareId(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_EQ(args.local_hardware_id, LOCAL_HARDWARE_ID_UNDEFINED);
}

// Tests PJRT API for getting addressable memories.
TEST_F(DeviceInstanceUnitTests, API_PJRT_Device_AddressableMemories) {
  m_device->addAddressableMemory(
      MemoryInstance::createInstance(m_device_as_vector,
                                     /*id=*/10,
                                     /*is_host_memory=*/true)
          .get());

  m_device->addAddressableMemory(
      MemoryInstance::createInstance(m_device_as_vector,
                                     /*id=*/11,
                                     /*is_host_memory=*/true)
          .get());

  PJRT_Device_AddressableMemories_Args args;
  args.struct_size = PJRT_Device_AddressableMemories_Args_STRUCT_SIZE;
  args.device = *m_device;
  args.memories = nullptr;
  args.num_memories = 0;

  PJRT_Error *error = internal::onDeviceAddressableMemories(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_EQ(args.num_memories, 2);
  EXPECT_NE(args.memories, nullptr);
}

// Tests PJRT API for getting default memory.
TEST_F(DeviceInstanceUnitTests, API_PJRT_Device_DefaultMemory) {
  m_device->setDefaultMemory(m_default_memory.get());

  PJRT_Device_DefaultMemory_Args args;
  args.struct_size = PJRT_Device_DefaultMemory_Args_STRUCT_SIZE;
  args.device = *m_device;
  args.memory = nullptr;

  PJRT_Error *error = internal::onDeviceDefaultMemory(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_NE(args.memory, nullptr);
  EXPECT_EQ(MemoryInstance::unwrap(args.memory), m_default_memory.get());
}

} // namespace tt::pjrt::tests
