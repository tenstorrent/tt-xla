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
#include <vector>

// GTest headers
#include "gtest/gtest.h"

// PJRT implementation headers
#include "api/device_instance.h"
#include "api/memory_instance.h"

#ifndef TT_XLA_PJRT_IMPLEMENTATION_SRC_API_UNIT_TESTS_INC_UNIT_TEST_UTILS_H_
#define TT_XLA_PJRT_IMPLEMENTATION_SRC_API_UNIT_TESTS_INC_UNIT_TEST_UTILS_H_

namespace tt::pjrt::tests {

// Common literals.
static constexpr int TT_DEVICE_HOST = 0;
static constexpr int LOCAL_HARDWARE_ID_UNDEFINED = -1;
static constexpr tt::target::Arch TT_ARCH_WH = tt::target::Arch::Wormhole_b0;
static constexpr int DEFAULT_MEMORY_ID = 1;
static constexpr int MEMORY_KIND_HOST = 0;
static constexpr int MEMORY_KIND_DEVICE = 1;

// Test fixure for PJRT unit tests. It runs before every test with this fixure,
// ensuring that a DeviceInstance is created and ready to be used in the test.
// For the purpose of tests, this will be a host device (TT_DEVICE_HOST ID).
// It also creates a default MemoryInstance and other helpers.
// There is no need to override the TearDown method because there is nothing
// to clean up.
class PJRTComponentUnitTests : public ::testing::Test {
protected:
  // Runs before every test.
  void SetUp() override {
    m_device = DeviceInstance::createInstance(
        TT_DEVICE_HOST,
        /*is_addressable=*/true, LOCAL_HARDWARE_ID_UNDEFINED, TT_ARCH_WH);

    m_device_as_vector.push_back(m_device.get());

    // DEVNOTE: Intentionally not set to device during setup.
    m_default_memory =
        MemoryInstance::createInstance(m_device_as_vector, DEFAULT_MEMORY_ID,
                                       /*is_host_memory=*/true);
  }

  // Device that is unique per-test.
  std::unique_ptr<DeviceInstance> m_device;

  // A single element device vector.
  std::vector<DeviceInstance *> m_device_as_vector;

  // Default memory space for the device.
  std::unique_ptr<MemoryInstance> m_default_memory;
};

} // namespace tt::pjrt::tests

#endif // TT_XLA_PJRT_IMPLEMENTATION_SRC_API_UNIT_TESTS_INC_UNIT_TEST_UTILS_H_
