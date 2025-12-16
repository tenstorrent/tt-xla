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
#include "api/executable_image.h"
#include "api/executable_instance.h"
#include "api/memory_instance.h"

namespace tt::pjrt::tests {

// Test fixture for ExecutableInstance unit tests.
// Uses SOExecutableImage which doesn't require a flatbuffer binary.
class ExecutableInstanceUnitTests : public ::testing::Test {
protected:
  // Common test values.
  static constexpr size_t NUM_INPUTS = 2;
  static constexpr size_t NUM_OUTPUTS = 1;
  static constexpr size_t NUM_REPLICAS = 1;
  static constexpr size_t NUM_PARTITIONS = 1;
  static constexpr size_t NUM_DEVICES = 1;

  void SetUp() override {
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

    m_executable = ExecutableInstance::createInstance(m_executable_image);
  }

  std::shared_ptr<SOExecutableImage> m_executable_image;
  std::unique_ptr<ExecutableInstance> m_executable;
};

// Tests successful creation of executable instances.
TEST_F(ExecutableInstanceUnitTests, createInstance_successCase) {
  ASSERT_NE(m_executable, nullptr);
  EXPECT_NE(m_executable->getExecutableImage(), nullptr);
}

// Tests casting ExecutableInstance to raw PJRT_Executable pointer.
TEST_F(ExecutableInstanceUnitTests, castToPJRTExecutable) {
  PJRT_Executable *pjrt_exec = *m_executable;
  EXPECT_NE(pjrt_exec, nullptr);
  EXPECT_EQ(static_cast<void *>(m_executable.get()),
            static_cast<void *>(pjrt_exec));
}

// Tests "unwrapping" raw PJRT_Executable pointer back to ExecutableInstance.
// Verifies the unwrapped instance matches the original.
TEST_F(ExecutableInstanceUnitTests, unwrapPJRTExecutable) {
  PJRT_Executable *pjrt_exec = *m_executable;
  ExecutableInstance *unwrapped = ExecutableInstance::unwrap(pjrt_exec);
  ASSERT_NE(unwrapped, nullptr);
  EXPECT_EQ(unwrapped, m_executable.get());
}

// Tests that getExecutableImage returns the correct image.
TEST_F(ExecutableInstanceUnitTests, getExecutableImage) {
  ExecutableImage *image = m_executable->getExecutableImage();
  ASSERT_NE(image, nullptr);
  EXPECT_EQ(image->getExecutableName(), "test_executable");
  EXPECT_EQ(image->getNumInputs(), NUM_INPUTS);
  EXPECT_EQ(image->getNumOutputs(), NUM_OUTPUTS);
  EXPECT_EQ(image->getNumReplicas(), NUM_REPLICAS);
  EXPECT_EQ(image->getNumPartitions(), NUM_PARTITIONS);
}

// Tests PJRT API for getting executable name.
TEST_F(ExecutableInstanceUnitTests, API_PJRT_Executable_Name) {
  PJRT_Executable_Name_Args args;
  args.struct_size = PJRT_Executable_Name_Args_STRUCT_SIZE;
  args.executable = *m_executable;
  args.executable_name = nullptr;
  args.executable_name_size = 0;

  PJRT_Error *error = internal::onExecutableName(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_NE(args.executable_name, nullptr);
  EXPECT_GT(args.executable_name_size, 0);
  EXPECT_EQ(std::string(args.executable_name, args.executable_name_size),
            "test_executable");
}

// Tests PJRT API for getting number of replicas.
TEST_F(ExecutableInstanceUnitTests, API_PJRT_Executable_NumReplicas) {
  PJRT_Executable_NumReplicas_Args args;
  args.struct_size = PJRT_Executable_NumReplicas_Args_STRUCT_SIZE;
  args.executable = *m_executable;
  args.num_replicas = 0; // intentionally different

  PJRT_Error *error = internal::onExecutableNumReplicas(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_EQ(args.num_replicas, NUM_REPLICAS);
}

// Tests PJRT API for getting number of partitions.
TEST_F(ExecutableInstanceUnitTests, API_PJRT_Executable_NumPartitions) {
  PJRT_Executable_NumPartitions_Args args;
  args.struct_size = PJRT_Executable_NumPartitions_Args_STRUCT_SIZE;
  args.executable = *m_executable;
  args.num_partitions = 0; // intentionally different

  PJRT_Error *error = internal::onExecutableNumPartitions(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_EQ(args.num_partitions, NUM_PARTITIONS);
}

// Tests PJRT API for getting number of outputs.
TEST_F(ExecutableInstanceUnitTests, API_PJRT_Executable_NumOutputs) {
  PJRT_Executable_NumOutputs_Args args;
  args.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
  args.executable = *m_executable;
  args.num_outputs = 0; // intentionally different

  PJRT_Error *error = internal::onExecutableNumOutputs(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_EQ(args.num_outputs, NUM_OUTPUTS);
}

// Tests PJRT API for getting size of generated code in bytes.
TEST_F(ExecutableInstanceUnitTests,
       API_PJRT_Executable_SizeOfGeneratedCodeInBytes) {
  PJRT_Executable_SizeOfGeneratedCodeInBytes_Args args;
  args.struct_size =
      PJRT_Executable_SizeOfGeneratedCodeInBytes_Args_STRUCT_SIZE;
  args.executable = *m_executable;
  args.size_in_bytes = 0;

  PJRT_Error *error = internal::onExecutableSizeOfGeneratedCodeInBytes(&args);
  ASSERT_EQ(error, nullptr);
  // Implementation returns -1 as we cannot estimate device memory usage.
  EXPECT_EQ(args.size_in_bytes, -1);
}

// Tests PJRT API for getting executable fingerprint.
TEST_F(ExecutableInstanceUnitTests, API_PJRT_Executable_Fingerprint) {
  PJRT_Executable_Fingerprint_Args args;
  args.struct_size = PJRT_Executable_Fingerprint_Args_STRUCT_SIZE;
  args.executable = *m_executable;
  args.executable_fingerprint = nullptr;
  args.executable_fingerprint_size = 0;

  PJRT_Error *error = internal::onExecutableFingerprint(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_NE(args.executable_fingerprint, nullptr);
  EXPECT_GT(args.executable_fingerprint_size, 0);
}

// Tests PJRT API for getting output element types.
TEST_F(ExecutableInstanceUnitTests, API_PJRT_Executable_OutputElementTypes) {
  PJRT_Executable_OutputElementTypes_Args args;
  args.struct_size = PJRT_Executable_OutputElementTypes_Args_STRUCT_SIZE;
  args.executable = *m_executable;
  args.output_types = nullptr;
  args.num_output_types = 0;

  PJRT_Error *error = internal::onExecutableOutputElementTypes(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_EQ(args.num_output_types, NUM_OUTPUTS);
  ASSERT_NE(args.output_types, nullptr);
  EXPECT_EQ(args.output_types[0], PJRT_Buffer_Type_F32);
}

// Tests PJRT API for getting output dimensions.
TEST_F(ExecutableInstanceUnitTests, API_PJRT_Executable_OutputDimensions) {
  PJRT_Executable_OutputDimensions_Args args;
  args.struct_size = PJRT_Executable_OutputDimensions_Args_STRUCT_SIZE;
  args.executable = *m_executable;
  args.dims = nullptr;
  args.dim_sizes = nullptr;
  args.num_outputs = 0;

  PJRT_Error *error = internal::onExecutableOutputDimensions(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_EQ(args.num_outputs, NUM_OUTPUTS);
  ASSERT_NE(args.dims, nullptr);
  ASSERT_NE(args.dim_sizes, nullptr);
  // First output has rank 2 with shape [2, 3].
  EXPECT_EQ(args.dim_sizes[0], 2);
  EXPECT_EQ(args.dims[0], 2);
  EXPECT_EQ(args.dims[1], 3);
}

// Tests PJRT API for getting output memory kinds.
TEST_F(ExecutableInstanceUnitTests, API_PJRT_Executable_OutputMemoryKinds) {
  PJRT_Executable_OutputMemoryKinds_Args args;
  args.struct_size = PJRT_Executable_OutputMemoryKinds_Args_STRUCT_SIZE;
  args.executable = *m_executable;
  args.memory_kinds = nullptr;
  args.memory_kind_sizes = nullptr;
  args.num_outputs = 0;

  PJRT_Error *error = internal::onExecutableOutputMemoryKinds(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_EQ(args.num_outputs, NUM_OUTPUTS);
  ASSERT_NE(args.memory_kinds, nullptr);
  ASSERT_NE(args.memory_kind_sizes, nullptr);
  EXPECT_EQ(std::string(args.memory_kinds[0], args.memory_kind_sizes[0]),
            MemoryInstance::c_device_memory_kind_name);
}

// Tests PJRT API for getting optimized program.
TEST_F(ExecutableInstanceUnitTests, API_PJRT_Executable_OptimizedProgram) {
  // First call with nullptr to get the required size.
  PJRT_Program program;
  program.code = nullptr;
  program.code_size = 0;

  PJRT_Executable_OptimizedProgram_Args args;
  args.struct_size = PJRT_Executable_OptimizedProgram_Args_STRUCT_SIZE;
  args.executable = *m_executable;
  args.program = &program;

  PJRT_Error *error = internal::onExecutableOptimizedProgram(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_GT(program.code_size, 0);

  // Second call with allocated buffer to get the code.
  std::vector<char> code_buffer(program.code_size);
  program.code = code_buffer.data();

  error = internal::onExecutableOptimizedProgram(&args);
  ASSERT_EQ(error, nullptr);
  EXPECT_EQ(std::string(code_buffer.data(), program.code_size),
            "original_mlir_code");
}

// Tests PJRT API for destroying executable.
TEST_F(ExecutableInstanceUnitTests, API_PJRT_Executable_Destroy) {
  // Create executable on heap since destroy will delete it.
  auto executable =
      ExecutableInstance::createInstance(m_executable_image).release();

  PJRT_Executable_Destroy_Args args;
  args.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
  args.executable = *executable;

  // Should not crash and should properly delete the executable.
  PJRT_Error *error = internal::onExecutableDestroy(&args);
  EXPECT_EQ(error, nullptr);
}

} // namespace tt::pjrt::tests
