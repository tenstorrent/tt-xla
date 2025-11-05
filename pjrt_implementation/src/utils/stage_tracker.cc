// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "utils/stage_tracker.h"
#include <cstdlib>
#include <cstring>

namespace tt::pjrt::utils {

StageTracker &StageTracker::getInstance() {
  static StageTracker instance;
  return instance;
}

void StageTracker::setStage(const char *stage_name) {
  if (stage_name == nullptr) {
    return;
  }

  // Check environment variable to enable stage tracking
  const char *enable_logging = std::getenv("ENABLE_BRINGUP_STAGE_LOGGING");
  if (enable_logging == nullptr || strcmp(enable_logging, "1") != 0) {
    return;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  current_stage_ = stage_name;
}

const char *StageTracker::getCurrentStageString() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return current_stage_.c_str();
}

} // namespace tt::pjrt::utils
