// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

// C++ standard library headers
#include <string>
#include <vector>

// GTest headers
#include "gtest/gtest.h"

// PJRT implementation headers
#include "api/device_instance.h"
#include "api/executable_image.h"
#include "api/executable_instance.h"
#include "api/loaded_executable_instance.h"
#include "api/memory_instance.h"
#include "utils/status.h"

namespace tt::pjrt::tests {

// Mock implementation of LoadedExecutableInstance for testing.
// Since LoadedExecutableInstance is abstract, we need a concrete
// implementation.
class MockLoadedExecutableInstance : public LoadedExecutableInstance {
public:
  MockLoadedExecutableInstance(
      std::shared_ptr<ExecutableImage> executable_image,
      const std::vector<DeviceInstance *> &addressable_devices)
      : LoadedExecutableInstance(std::move(executable_image),
                                 addressable_devices,
                                 /*client_instance=*/nullptr) {}

  tt_pjrt_status execute(PJRT_LoadedExecutable_Execute_Args *args) override {
    // Mock implementation - just return success.
    return tt_pjrt_status::kSuccess;
  }

protected:
  std::optional<tt::runtime::Tensor>
  prepareInputTensor(const std::vector<BufferInstance *> &arg_buffers,
                     tt::runtime::Device device, size_t num_devices,
                     std::uint32_t program_index, size_t arg_index) override {
    // Mock implementation - return nullopt.
    return std::nullopt;
  }
};

// Test fixture for LoadedExecutableInstance unit tests.
class LoadedExecutableInstanceUnitTests : public ::testing::Test {
protected:
  // Common test values.
  static constexpr size_t NUM_INPUTS = 2;
  static constexpr size_t NUM_OUTPUTS = 1;
  static constexpr size_t NUM_REPLICAS = 1;
  static constexpr size_t NUM_PARTITIONS = 1;
  static constexpr size_t NUM_DEVICES = 1;
  static constexpr int TT_DEVICE_HOST = 0;
  static constexpr int LOCAL_HARDWARE_ID_UNDEFINED = -1;
  static constexpr tt::target::Arch TT_ARCH_WH = tt::target::Arch::Wormhole_b0;

  void SetUp() override {
    // Create a device for testing.
    m_device = DeviceInstance::createInstance(
        TT_DEVICE_HOST,
        /*is_addressable=*/true, LOCAL_HARDWARE_ID_UNDEFINED, TT_ARCH_WH);
    m_addressable_devices.push_back(m_device.get());

    // Create output dimensions for one output with shape [2, 3].
    std::vector<std::vector<std::uint32_t>> output_dimensions = {{2, 3}};
    std::vector<size_t> output_ranks = {2};
    std::vector<std::int64_t> output_dimensions_flat = {2, 3};
    std::vector<PJRT_Buffer_Type> output_types = {PJRT_Buffer_Type_F32};
    std::vector<std::uint32_t> mesh_shape = {1, 1};

    // Create empty sharding info for inputs and outputs.
    std::vector<mlir::tt::sharding_utils::MeshSharding> input_sharding(
        NUM_INPUTS);
    std::vector<mlir::tt::sharding_utils::MeshSharding> output_sharding(
        NUM_OUTPUTS);

    // Create output memory kinds.
    std::vector<const char *> output_memory_kinds = {
        MemoryInstance::c_device_memory_kind_name.data()};
    std::vector<size_t> output_memory_kinds_sizes = {
        MemoryInstance::c_device_memory_kind_name.size()};

    CompileOptions compile_options;

    // Create SOExecutableImage (doesn't require flatbuffer binary).
    m_executable_image = SOExecutableImage::createInstance(
        "original_mlir_code", "ttir_mlir_code", "ttnn_mlir_code",
        "test_executable", NUM_INPUTS, NUM_OUTPUTS,
        std::move(output_dimensions), std::move(output_ranks),
        std::move(output_dimensions_flat), NUM_PARTITIONS, NUM_REPLICAS,
        NUM_DEVICES, mesh_shape, input_sharding, output_sharding, output_types,
        std::move(output_memory_kinds), std::move(output_memory_kinds_sizes),
        std::move(compile_options));

    m_loaded_executable = std::make_unique<MockLoadedExecutableInstance>(
        m_executable_image, m_addressable_devices);
  }

  std::unique_ptr<DeviceInstance> m_device;
  std::vector<DeviceInstance *> m_addressable_devices;
  std::shared_ptr<SOExecutableImage> m_executable_image;
  std::unique_ptr<MockLoadedExecutableInstance> m_loaded_executable;
};

// Tests casting LoadedExecutableInstance to raw PJRT_LoadedExecutable pointer.
TEST_F(LoadedExecutableInstanceUnitTests, castToPJRTLoadedExecutable) {
  PJRT_LoadedExecutable *pjrt_exec = *m_loaded_executable;
  EXPECT_NE(pjrt_exec, nullptr);
  EXPECT_EQ(static_cast<void *>(m_loaded_executable.get()),
            static_cast<void *>(pjrt_exec));
}

// Tests "unwrapping" raw PJRT_LoadedExecutable pointer back to
// LoadedExecutableInstance. Verifies the unwrapped instance matches the
// original.
TEST_F(LoadedExecutableInstanceUnitTests, unwrapPJRTLoadedExecutable) {
  PJRT_LoadedExecutable *pjrt_exec = *m_loaded_executable;
  LoadedExecutableInstance *unwrapped =
      LoadedExecutableInstance::unwrap(pjrt_exec);
  ASSERT_NE(unwrapped, nullptr);
  EXPECT_EQ(unwrapped, m_loaded_executable.get());
}

// Tests getting addressable devices.
TEST_F(LoadedExecutableInstanceUnitTests, getAddressableDevices) {
  const std::vector<DeviceInstance *> &devices =
      m_loaded_executable->getAddressableDevices();
  EXPECT_EQ(devices.size(), 1);
  EXPECT_EQ(devices[0], m_device.get());
}

// Tests getting shared executable image.
TEST_F(LoadedExecutableInstanceUnitTests, getSharedExecutableImage) {
  std::shared_ptr<ExecutableImage> image =
      m_loaded_executable->getSharedExecutableImage();
  EXPECT_NE(image, nullptr);
  EXPECT_EQ(image->getExecutableName(), "test_executable");
}

// Tests isDeleted returns false initially.
TEST_F(LoadedExecutableInstanceUnitTests, isDeleted_initiallyFalse) {
  EXPECT_FALSE(m_loaded_executable->isDeleted());
}

// Tests releaseResources marks the executable as deleted.
TEST_F(LoadedExecutableInstanceUnitTests, releaseResources) {
  EXPECT_FALSE(m_loaded_executable->isDeleted());
  m_loaded_executable->releaseResources();
  EXPECT_TRUE(m_loaded_executable->isDeleted());
}

// Tests that releaseResources can be called multiple times safely.
TEST_F(LoadedExecutableInstanceUnitTests, releaseResources_multipleCallsSafe) {
  m_loaded_executable->releaseResources();
  m_loaded_executable->releaseResources(); // should not crash
  EXPECT_TRUE(m_loaded_executable->isDeleted());
}

// Tests PJRT API for getting addressable devices.
TEST_F(LoadedExecutableInstanceUnitTests,
       API_PJRT_LoadedExecutable_AddressableDevices) {
  PJRT_LoadedExecutable_AddressableDevices_Args args;
  args.struct_size = PJRT_LoadedExecutable_AddressableDevices_Args_STRUCT_SIZE;
  args.executable = *m_loaded_executable;
  args.addressable_devices = nullptr;
  args.num_addressable_devices = 0;

  PJRT_Error *error = internal::onLoadedExecutableAddressableDevices(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_EQ(args.num_addressable_devices, 1);
  ASSERT_NE(args.addressable_devices, nullptr);
  EXPECT_EQ(DeviceInstance::unwrap(args.addressable_devices[0]),
            m_device.get());
}

// Tests PJRT API for checking if loaded executable is deleted.
TEST_F(LoadedExecutableInstanceUnitTests, API_PJRT_LoadedExecutable_IsDeleted) {
  PJRT_LoadedExecutable_IsDeleted_Args args;
  args.struct_size = PJRT_LoadedExecutable_IsDeleted_Args_STRUCT_SIZE;
  args.executable = *m_loaded_executable;
  args.is_deleted = true; // intentionally different

  PJRT_Error *error = internal::onLoadedExecutableIsDeleted(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_FALSE(args.is_deleted);

  m_loaded_executable->releaseResources();
  error = internal::onLoadedExecutableIsDeleted(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_TRUE(args.is_deleted);
}

// Tests PJRT API for deleting loaded executable.
TEST_F(LoadedExecutableInstanceUnitTests, API_PJRT_LoadedExecutable_Delete) {
  EXPECT_FALSE(m_loaded_executable->isDeleted());

  PJRT_LoadedExecutable_Delete_Args args;
  args.struct_size = PJRT_LoadedExecutable_Delete_Args_STRUCT_SIZE;
  args.executable = *m_loaded_executable;

  PJRT_Error *error = internal::onLoadedExecutableDelete(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_TRUE(m_loaded_executable->isDeleted());
}

// Tests PJRT API for getting executable from loaded executable.
TEST_F(LoadedExecutableInstanceUnitTests,
       API_PJRT_LoadedExecutable_GetExecutable) {
  PJRT_LoadedExecutable_GetExecutable_Args args;
  args.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
  args.loaded_executable = *m_loaded_executable;
  args.executable = nullptr;

  PJRT_Error *error = internal::onLoadedExecutableGetExecutable(&args);
  ASSERT_EQ(error, nullptr);
  ASSERT_NE(args.executable, nullptr);

  // Verify the returned executable has the same image.
  ExecutableInstance *exec_instance =
      ExecutableInstance::unwrap(args.executable);
  EXPECT_EQ(exec_instance->getExecutableImage()->getExecutableName(),
            "test_executable");

  // Clean up - the API caller is responsible for destroying the executable.
  delete exec_instance;
}

// Tests PJRT API for destroying loaded executable.
TEST_F(LoadedExecutableInstanceUnitTests, API_PJRT_LoadedExecutable_Destroy) {
  // Create a new loaded executable on heap since destroy will delete it.
  auto loaded_exec = std::make_unique<MockLoadedExecutableInstance>(
      m_executable_image, m_addressable_devices);
  LoadedExecutableInstance *loaded_exec_ptr = loaded_exec.release();

  PJRT_LoadedExecutable_Destroy_Args args;
  args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
  args.executable = *loaded_exec_ptr;

  // Should not crash and should properly delete the loaded executable.
  PJRT_Error *error = internal::onLoadedExecutableDestroy(&args);
  EXPECT_EQ(error, nullptr);
}

} // namespace tt::pjrt::tests
