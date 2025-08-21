# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""
# code apapted from :
# https://github.com/ServiceNow/N-BEATS.git

This source code is provided for the purposes of scientific reproducibility
under the following limited license from Element AI Inc. The code is an
implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
expansion analysis for interpretable time series forecasting,
https://arxiv.org/abs/1905.10437). The copyright to the source code is licensed
under the Creative Commons - Attribution-NonCommercial 4.0 International license
(CC BY-NC 4.0): https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial
use (whether for the benefit of third parties or internally in production)
requires an explicit license. The subject-matter of the N-BEATS model and
associated materials are the property of Element AI Inc. and may be subject to
patent protection. No license to patents is granted hereunder (whether express
or implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""

"""
Common settings
"""

import os

STORAGE = "STORAGE"
DATASETS_PATH = os.path.join(STORAGE, "datasets")
EXPERIMENTS_PATH = os.path.join(STORAGE, "experiments")
TESTS_STORAGE_PATH = os.path.join(STORAGE, "test")
