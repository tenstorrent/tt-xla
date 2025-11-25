// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "utils/data_type_utils.h"
#include "utils/logging.h"
#include "utils/utils.h"

#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Test data_type_utils functions
class DataTypeUtilsTest : public ::testing::Test {};

TEST_F(DataTypeUtilsTest, GetPJRTBufferTypeString_CommonTypes) {
  using namespace tt::pjrt::data_type_utils;

  EXPECT_EQ(getPJRTBufferTypeString(PJRT_Buffer_Type_INVALID),
            "PJRT_Buffer_Type_INVALID");
  EXPECT_EQ(getPJRTBufferTypeString(PJRT_Buffer_Type_PRED),
            "PJRT_Buffer_Type_PRED");
  EXPECT_EQ(getPJRTBufferTypeString(PJRT_Buffer_Type_S8), "PJRT_Buffer_Type_S8");
  EXPECT_EQ(getPJRTBufferTypeString(PJRT_Buffer_Type_S16),
            "PJRT_Buffer_Type_S16");
  EXPECT_EQ(getPJRTBufferTypeString(PJRT_Buffer_Type_S32),
            "PJRT_Buffer_Type_S32");
  EXPECT_EQ(getPJRTBufferTypeString(PJRT_Buffer_Type_S64),
            "PJRT_Buffer_Type_S64");
  EXPECT_EQ(getPJRTBufferTypeString(PJRT_Buffer_Type_U8), "PJRT_Buffer_Type_U8");
  EXPECT_EQ(getPJRTBufferTypeString(PJRT_Buffer_Type_U16),
            "PJRT_Buffer_Type_U16");
  EXPECT_EQ(getPJRTBufferTypeString(PJRT_Buffer_Type_U32),
            "PJRT_Buffer_Type_U32");
  EXPECT_EQ(getPJRTBufferTypeString(PJRT_Buffer_Type_U64),
            "PJRT_Buffer_Type_U64");
  EXPECT_EQ(getPJRTBufferTypeString(PJRT_Buffer_Type_F16),
            "PJRT_Buffer_Type_F16");
  EXPECT_EQ(getPJRTBufferTypeString(PJRT_Buffer_Type_F32),
            "PJRT_Buffer_Type_F32");
  EXPECT_EQ(getPJRTBufferTypeString(PJRT_Buffer_Type_F64),
            "PJRT_Buffer_Type_F64");
  EXPECT_EQ(getPJRTBufferTypeString(PJRT_Buffer_Type_BF16),
            "PJRT_Buffer_Type_BF16");
}

TEST_F(DataTypeUtilsTest, GetPJRTBufferTypeString_UnknownType) {
  using namespace tt::pjrt::data_type_utils;

  // Test with an invalid enum value (assuming max enum value + 1)
  // This tests the default case
  PJRT_Buffer_Type unknown_type = static_cast<PJRT_Buffer_Type>(9999);
  EXPECT_EQ(getPJRTBufferTypeString(unknown_type), "UNKNOWN_TYPE");
}

TEST_F(DataTypeUtilsTest, ConvertRuntimeToPJRTDataType_SupportedTypes) {
  using namespace tt::pjrt::data_type_utils;

  EXPECT_EQ(convertRuntimeToPJRTDataType(tt::target::DataType::UInt8),
            PJRT_Buffer_Type_U8);
  EXPECT_EQ(convertRuntimeToPJRTDataType(tt::target::DataType::UInt16),
            PJRT_Buffer_Type_U16);
  EXPECT_EQ(convertRuntimeToPJRTDataType(tt::target::DataType::UInt32),
            PJRT_Buffer_Type_U32);
  EXPECT_EQ(convertRuntimeToPJRTDataType(tt::target::DataType::Int32),
            PJRT_Buffer_Type_S32);
  EXPECT_EQ(convertRuntimeToPJRTDataType(tt::target::DataType::Float16),
            PJRT_Buffer_Type_F16);
  EXPECT_EQ(convertRuntimeToPJRTDataType(tt::target::DataType::Float32),
            PJRT_Buffer_Type_F32);
  EXPECT_EQ(convertRuntimeToPJRTDataType(tt::target::DataType::BFloat16),
            PJRT_Buffer_Type_BF16);
}

TEST_F(DataTypeUtilsTest, ConvertRuntimeToPJRTDataType_UnsupportedType) {
  using namespace tt::pjrt::data_type_utils;

  // Test with an unsupported type
  // Based on the code, only UInt8, UInt16, UInt32, Int32, Float16, Float32, BFloat16 are supported
  // Int8, Int16, Int64, UInt64, Float64, Bool should throw
  EXPECT_THROW(convertRuntimeToPJRTDataType(tt::target::DataType::Int8),
               std::runtime_error);
  EXPECT_THROW(convertRuntimeToPJRTDataType(tt::target::DataType::Int16),
               std::runtime_error);
  EXPECT_THROW(convertRuntimeToPJRTDataType(tt::target::DataType::Int64),
               std::runtime_error);
  EXPECT_THROW(convertRuntimeToPJRTDataType(tt::target::DataType::UInt64),
               std::runtime_error);
  EXPECT_THROW(convertRuntimeToPJRTDataType(tt::target::DataType::Float64),
               std::runtime_error);
  EXPECT_THROW(convertRuntimeToPJRTDataType(tt::target::DataType::Bool),
               std::runtime_error);
}

TEST_F(DataTypeUtilsTest, ConvertPJRTToRuntimeDataType_SupportedTypes) {
  using namespace tt::pjrt::data_type_utils;

  EXPECT_EQ(convertPJRTToRuntimeDataType(PJRT_Buffer_Type_U8),
            tt::target::DataType::UInt8);
  EXPECT_EQ(convertPJRTToRuntimeDataType(PJRT_Buffer_Type_U16),
            tt::target::DataType::UInt16);
  EXPECT_EQ(convertPJRTToRuntimeDataType(PJRT_Buffer_Type_U32),
            tt::target::DataType::UInt32);
  EXPECT_EQ(convertPJRTToRuntimeDataType(PJRT_Buffer_Type_S32),
            tt::target::DataType::Int32);
  EXPECT_EQ(convertPJRTToRuntimeDataType(PJRT_Buffer_Type_F16),
            tt::target::DataType::Float16);
  EXPECT_EQ(convertPJRTToRuntimeDataType(PJRT_Buffer_Type_F32),
            tt::target::DataType::Float32);
  EXPECT_EQ(convertPJRTToRuntimeDataType(PJRT_Buffer_Type_BF16),
            tt::target::DataType::BFloat16);
  EXPECT_EQ(convertPJRTToRuntimeDataType(PJRT_Buffer_Type_F64),
            tt::target::DataType::Float64);
  EXPECT_EQ(convertPJRTToRuntimeDataType(PJRT_Buffer_Type_S64),
            tt::target::DataType::Int64);
  EXPECT_EQ(convertPJRTToRuntimeDataType(PJRT_Buffer_Type_S16),
            tt::target::DataType::Int16);
  EXPECT_EQ(convertPJRTToRuntimeDataType(PJRT_Buffer_Type_S8),
            tt::target::DataType::Int8);
  EXPECT_EQ(convertPJRTToRuntimeDataType(PJRT_Buffer_Type_U64),
            tt::target::DataType::UInt64);
  EXPECT_EQ(convertPJRTToRuntimeDataType(PJRT_Buffer_Type_PRED),
            tt::target::DataType::Bool);
}

TEST_F(DataTypeUtilsTest, ConvertPJRTToRuntimeDataType_UnsupportedType) {
  using namespace tt::pjrt::data_type_utils;

  // Test with an unsupported type
  PJRT_Buffer_Type unsupported = PJRT_Buffer_Type_INVALID;
  EXPECT_THROW(convertPJRTToRuntimeDataType(unsupported), std::runtime_error);
}

TEST_F(DataTypeUtilsTest, RoundTripConversion) {
  using namespace tt::pjrt::data_type_utils;

  // Test round-trip conversions for types that support both directions
  std::vector<PJRT_Buffer_Type> supported_types = {
      PJRT_Buffer_Type_U8,  PJRT_Buffer_Type_U16, PJRT_Buffer_Type_U32,
      PJRT_Buffer_Type_S32, PJRT_Buffer_Type_F16, PJRT_Buffer_Type_F32,
      PJRT_Buffer_Type_BF16};

  for (PJRT_Buffer_Type pjrt_type : supported_types) {
    tt::target::DataType runtime_type = convertPJRTToRuntimeDataType(pjrt_type);
    PJRT_Buffer_Type converted_back =
        convertRuntimeToPJRTDataType(runtime_type);
    EXPECT_EQ(converted_back, pjrt_type)
        << "Round-trip conversion failed for type: "
        << getPJRTBufferTypeString(pjrt_type);
  }
}

// Test utils::to_string function
class UtilsToStringTest : public ::testing::Test {};

TEST_F(UtilsToStringTest, ToString_EmptyVector) {
  using namespace tt::pjrt::utils;

  std::vector<int> empty_vec;
  EXPECT_EQ(to_string(empty_vec), "[]");
}

TEST_F(UtilsToStringTest, ToString_SingleElement) {
  using namespace tt::pjrt::utils;

  std::vector<int> vec = {42};
  EXPECT_EQ(to_string(vec), "[42]");
}

TEST_F(UtilsToStringTest, ToString_MultipleElements) {
  using namespace tt::pjrt::utils;

  std::vector<int> vec = {1, 2, 3, 4, 5};
  EXPECT_EQ(to_string(vec), "[1, 2, 3, 4, 5]");
}

TEST_F(UtilsToStringTest, ToString_StringVector) {
  using namespace tt::pjrt::utils;

  std::vector<std::string> vec = {"hello", "world", "test"};
  EXPECT_EQ(to_string(vec), "[hello, world, test]");
}

TEST_F(UtilsToStringTest, ToString_FloatVector) {
  using namespace tt::pjrt::utils;

  std::vector<float> vec = {1.5f, 2.5f, 3.5f};
  std::string result = to_string(vec);
  // Check that it contains the expected format (brackets and commas)
  EXPECT_TRUE(result.front() == '[' && result.back() == ']');
  // Should contain commas between elements
  EXPECT_TRUE(result.find(',') != std::string::npos);
}

// Test utils::invoke function
class UtilsInvokeTest : public ::testing::Test {};

TEST_F(UtilsInvokeTest, Invoke_SuccessfulFunction) {
  using namespace tt::pjrt::utils;

  auto add = [](int a, int b) { return a + b; };
  auto result = invoke(add, 5, 3);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, 8);
}

TEST_F(UtilsInvokeTest, Invoke_VoidFunction) {
  using namespace tt::pjrt::utils;

  bool called = false;
  auto void_fn = [&called]() { called = true; };
  auto result = invoke(void_fn);
  ASSERT_TRUE(result.has_value());
  EXPECT_TRUE(called);
}

TEST_F(UtilsInvokeTest, Invoke_ExceptionHandling) {
  using namespace tt::pjrt::utils;

  auto throwing_fn = []() -> int {
    throw std::runtime_error("Test exception");
    return 42;
  };
  auto result = invoke(throwing_fn);
  EXPECT_FALSE(result.has_value());
}

TEST_F(UtilsInvokeTest, Invoke_UnknownException) {
  using namespace tt::pjrt::utils;

  auto throwing_fn = []() -> int {
    throw "Unknown exception type";
    return 42;
  };
  auto result = invoke(throwing_fn);
  EXPECT_FALSE(result.has_value());
}

TEST_F(UtilsInvokeTest, Invoke_WithReturnValue) {
  using namespace tt::pjrt::utils;

  auto multiply = [](int a, int b) { return a * b; };
  auto result = invoke(multiply, 6, 7);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, 42);
}

// Test logging functions
class LoggingTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Save original environment
    const char *original_bringup = std::getenv("ENABLE_BRINGUP_STAGE_LOGGING");
    if (original_bringup) {
      original_bringup_value = original_bringup;
    }
    // Clean up test file if it exists
    std::remove("._bringup_stage.txt");
  }

  void TearDown() override {
    // Restore original environment
    if (original_bringup_value.empty()) {
      unsetenv("ENABLE_BRINGUP_STAGE_LOGGING");
    } else {
      setenv("ENABLE_BRINGUP_STAGE_LOGGING", original_bringup_value.c_str(), 1);
    }
    // Clean up test file
    std::remove("._bringup_stage.txt");
  }

  std::string original_bringup_value;
};

TEST_F(LoggingTest, IsBringupStageLoggingEnabled_NotSet) {
  unsetenv("ENABLE_BRINGUP_STAGE_LOGGING");
  EXPECT_FALSE(tt::pjrt::isBringupStageLoggingEnabled());
}

TEST_F(LoggingTest, IsBringupStageLoggingEnabled_SetToZero) {
  setenv("ENABLE_BRINGUP_STAGE_LOGGING", "0", 1);
  EXPECT_FALSE(tt::pjrt::isBringupStageLoggingEnabled());
}

TEST_F(LoggingTest, IsBringupStageLoggingEnabled_SetToOne) {
  setenv("ENABLE_BRINGUP_STAGE_LOGGING", "1", 1);
  EXPECT_TRUE(tt::pjrt::isBringupStageLoggingEnabled());
}

TEST_F(LoggingTest, LogBringupStage_Disabled) {
  unsetenv("ENABLE_BRINGUP_STAGE_LOGGING");
  tt::pjrt::logBringupStage("test_stage");
  // File should not be created
  std::ifstream file("._bringup_stage.txt");
  EXPECT_FALSE(file.good());
}

TEST_F(LoggingTest, LogBringupStage_Enabled) {
  setenv("ENABLE_BRINGUP_STAGE_LOGGING", "1", 1);
  tt::pjrt::logBringupStage("test_stage_name");
  // File should be created with the stage name
  std::ifstream file("._bringup_stage.txt");
  EXPECT_TRUE(file.good());
  std::string content;
  std::getline(file, content);
  EXPECT_EQ(content, "test_stage_name");
}

} // namespace
