// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

// GTest headers
#include "gtest/gtest.h"

// PJRT implementation headers
#include "api/error_instance.h"
#include "utils/status.h"

namespace tt::pjrt::tests {

// Tests successful creation of error instances.
TEST(ErrorInstanceUnitTests, makeError_successCase) {
  auto error = ErrorInstance::makeError(tt_pjrt_status::kUnknown);
  ASSERT_NE(error, nullptr);
  EXPECT_EQ(error->getStatus(), tt_pjrt_status::kUnknown);
  EXPECT_FALSE(error->getMessage().empty());
  EXPECT_EQ(
    error->getMessage(),
    "Error code: " + std::to_string(static_cast<int>(tt_pjrt_status::kUnknown)));
}

// Tests the invalid case of error instance creation.
TEST(ErrorInstanceUnitTests, makeError_failCase) {
  auto error = ErrorInstance::makeError(tt_pjrt_status::kSuccess);
  EXPECT_EQ(error, nullptr);
}

// Tests casting ErrorInstance to raw PJRT_Error pointer.
TEST(ErrorInstanceUnitTests, castToPJRTError) {
  auto error = ErrorInstance::makeError(tt_pjrt_status::kOutOfRange);
  PJRT_Error *pjrt_error = *error;
  EXPECT_NE(pjrt_error, nullptr);
  EXPECT_EQ(static_cast<void *>(error.get()), static_cast<void *>(pjrt_error));
}

// Tests "unwrapping" raw PJRT_Error pointer back to ErrorInstance .
// Verifies the unwrapped instance matches the original.
TEST(ErrorInstanceUnitTests, unwrapPJRTError) {
  auto error = ErrorInstance::makeError(tt_pjrt_status::kAborted);
  PJRT_Error *pjrt_error = *error;
  const ErrorInstance *unwrapped = ErrorInstance::unwrap(pjrt_error);
  ASSERT_NE(unwrapped, nullptr);
  EXPECT_EQ(unwrapped, error.get());
  EXPECT_EQ(unwrapped->getStatus(), tt_pjrt_status::kAborted);
  EXPECT_EQ(unwrapped->getMessage(), error->getMessage());
}

// Tests that different status codes produce different messages.
TEST(ErrorInstanceUnitTests, getMessage_differentStatusDifferentMessage) {
  EXPECT_NE(
    ErrorInstance::makeError(tt_pjrt_status::kInvalidArgument)->getMessage(),
    ErrorInstance::makeError(tt_pjrt_status::kNotFound)->getMessage());
}

// Tests PJRT API for getting the error code.
// Verifies correct PJRT error code is returned.
TEST(ErrorInstanceUnitTests, API_PJRT_Error_GetCode) {
  auto error = ErrorInstance::makeError(tt_pjrt_status::kDeadlineExceeded);

  PJRT_Error_GetCode_Args args;
  args.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
  args.error = *error;
  args.code = PJRT_Error_Code_FAILED_PRECONDITION; // intentionally different

  PJRT_Error *result = internal::onErrorGetCode(&args);
  ASSERT_EQ(result, nullptr);
  EXPECT_EQ(args.code, PJRT_Error_Code_DEADLINE_EXCEEDED);
}

// Tests PJRT API for getting the error message.
// Verifies message is non-empty and matches error's message.
TEST(ErrorInstanceUnitTests, API_PJRT_Error_Message) {
  auto error = ErrorInstance::makeError(tt_pjrt_status::kPermissionDenied);

  PJRT_Error_Message_Args args;
  args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  args.error = *error;
  args.message = nullptr;
  args.message_size = 0;

  internal::onErrorMessage(&args);
  EXPECT_TRUE(args.message_size > 0);
  EXPECT_EQ(args.message, error->getMessage());
}

} // namespace tt::pjrt::tests
