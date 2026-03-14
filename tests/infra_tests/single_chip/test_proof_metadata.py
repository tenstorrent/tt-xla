# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from ttxla_tools.serialization import save_proof_metadata_to_disk


def test_save_proof_metadata_to_disk(tmp_path):
    output_prefix = str(tmp_path / "proof_case")

    metadata_path = Path(save_proof_metadata_to_disk(output_prefix, test_name="proof_case"))

    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text())
    assert metadata["test_name"] == "proof_case"
    assert metadata["output_prefix"] == output_prefix
    assert metadata["tt_xla_commit"]
    assert metadata["tt_mlir_version"]
    assert "timestamp_utc" in metadata
    assert metadata["tt_forge_models_commit"] is not None
    assert isinstance(metadata["tt_xla_dirty"], bool)
