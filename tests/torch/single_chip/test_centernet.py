# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
from loguru import logger

from third_party.tt_forge_models.centernet.pytorch import ModelLoader


@pytest.mark.push
def test_centernet():

    loader = ModelLoader()
    model = loader.load_model()

    logger.info("model={}", model)
