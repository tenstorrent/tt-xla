// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

// GTest includes
#include <gtest/gtest.h>

// PJRT implementation includes
#include "utils/data_type_utils.h"
#include "utils/logging.h"
#include "utils/utils.h"

// C++ standard library includes
#include <cstdlib>
#include <filesystem>
#include <vector>

namespace tt::pjrt::tests {

// ---------- Fixture classes --------------------------------------------------

class PJRTUtilLoggingTests : public ::testing::Test {
public:
  static const char *TMP_LOG_FILE_NAME;
  static const char *ENV_VAR_LOG_FILE_PATH;
  static const char *ENV_VAR_LOGGER_LEVEL;
  static const char *ENV_VAR_BRINGUP_LOGGING;

protected:
  void SetUp() override {
    setenv(ENV_VAR_BRINGUP_LOGGING, "1", /*overwrite=*/1);
    setenv(ENV_VAR_LOG_FILE_PATH, TMP_LOG_FILE_NAME, /*overwrite=*/1);
    std::filesystem::remove(TMP_LOG_FILE_NAME);
  }

  void TearDown() override {
    std::filesystem::remove(TMP_LOG_FILE_NAME);
  }
};

const char *PJRTUtilLoggingTests::TMP_LOG_FILE_NAME = "tmp_test.log";

const char *PJRTUtilLoggingTests::ENV_VAR_LOG_FILE_PATH = "TT_XLA_LOGGER_FILE";

const char *PJRTUtilLoggingTests::ENV_VAR_LOGGER_LEVEL = "LOGGER_LEVEL";

const char *PJRTUtilLoggingTests::ENV_VAR_BRINGUP_LOGGING =
  "ENABLE_BRINGUP_STAGE_LOGGING";


// ---------- Tests ------------------------------------------------------------

TEST(PJRTUtilUnitTests, Test_getPJRTBufferTypeString_defaultCase) {
  PJRT_Buffer_Type invalid_type = static_cast<PJRT_Buffer_Type>(~0U);
  EXPECT_EQ(
    tt::pjrt::data_type_utils::getPJRTBufferTypeString(invalid_type),
    "UNKNOWN_TYPE");
}

TEST(PJRTUtilUnitTests, Test_convertRuntimeToPJRTDataType_failCase) {
  const std::vector<tt::target::DataType> unsupported_types = {
    tt::target::DataType::Bool,
    tt::target::DataType::Int8,
    tt::target::DataType::Int16,
    tt::target::DataType::Int64,
    tt::target::DataType::UInt64,
    tt::target::DataType::Float64,
  };
  for (tt::target::DataType unsupported : unsupported_types) {
    EXPECT_THROW(
      tt::pjrt::data_type_utils::convertRuntimeToPJRTDataType(unsupported),
      std::runtime_error);
  }
}

TEST(PJRTUtilUnitTests, Test_convertPJRTToRuntimeDataType_failCase) {
  const std::vector<PJRT_Buffer_Type> unsupported_types = {
    PJRT_Buffer_Type_INVALID,
    PJRT_Buffer_Type_F8E5M2,
    PJRT_Buffer_Type_F8E4M3FN,
    PJRT_Buffer_Type_F8E4M3B11FNUZ,
  };
  for (PJRT_Buffer_Type unsupported : unsupported_types) {
    EXPECT_THROW(
      tt::pjrt::data_type_utils::convertPJRTToRuntimeDataType(unsupported),
      std::runtime_error);
  }
}

TEST(PJRTUtilUnitTests, Test_convertDataTypeBetweenPJRTAndRuntime_successCase) {
  const std::vector<tt::target::DataType> supported_types_in_both = {
    tt::target::DataType::UInt8,
    tt::target::DataType::UInt16,
    tt::target::DataType::UInt32,
    tt::target::DataType::Int32,
    tt::target::DataType::Float16,
    tt::target::DataType::Float32,
    tt::target::DataType::BFloat16,
  };
  for (tt::target::DataType runtime_type : supported_types_in_both) {
    EXPECT_EQ(
      runtime_type,
      tt::pjrt::data_type_utils::convertPJRTToRuntimeDataType(
        tt::pjrt::data_type_utils::convertRuntimeToPJRTDataType(runtime_type)));
  }
}

TEST_F(PJRTUtilLoggingTests, Test_isBringupStageLoggingEnabled) {
  EXPECT_TRUE(tt::pjrt::isBringupStageLoggingEnabled());
  setenv(ENV_VAR_BRINGUP_LOGGING, "1", /*overwrite=*/1);
  EXPECT_TRUE(tt::pjrt::isBringupStageLoggingEnabled());
  setenv(ENV_VAR_BRINGUP_LOGGING, "0", /*overwrite=*/1);
  EXPECT_FALSE(tt::pjrt::isBringupStageLoggingEnabled());
}

TEST_F(PJRTUtilLoggingTests, Test_initializeLogging) {
  unsetenv(ENV_VAR_LOGGER_LEVEL);
  tt::pjrt::initializeLogging();
  ASSERT_FALSE(std::filesystem::exists(TMP_LOG_FILE_NAME));

  setenv(ENV_VAR_LOGGER_LEVEL, "INFO", /*overwrite=*/1);
  tt::pjrt::initializeLogging();
  EXPECT_EQ(loguru::g_stderr_verbosity, loguru::NamedVerbosity::Verbosity_INFO);
  EXPECT_TRUE(std::filesystem::exists(TMP_LOG_FILE_NAME));
}

} // namespace tt::pjrt::tests
