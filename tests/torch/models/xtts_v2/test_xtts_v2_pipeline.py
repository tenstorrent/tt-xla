# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""XTTS-v2 (coqui/XTTS-v2) nightly end-to-end text-to-speech pipeline test.

Runs the full ``Xtts.inference`` path with every learned nn.Module on TT and
asserts the output artifact is valid (a finite, non-empty 24 kHz waveform). xfail
(non-strict) pending the conditioning group_norm (#5483 / tt-mlir #8935) and
HiFi-GAN squeeze (#5375 / #5388) fixes. Needs coqui-tts, weights, and a TT device.
"""

import os
from pathlib import Path

import pytest
from infra import RunMode
from utils import BringupStatus, Category, ModelGroup

pytestmark = [
    pytest.mark.nightly,
    pytest.mark.model_test,
    pytest.mark.single_device,
    pytest.mark.large,
]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_LOADER_PATH = str(
    _REPO_ROOT / "third_party" / "tt_forge_models" / "xtts_v2" / "pytorch" / "loader.py"
)

# Small cap so the single-compile decode loop stays test-sized (each token is
# ~1024 output samples @ 24 kHz); the property under test is that the chain runs
# end-to-end and emits a valid waveform, not audio length.
MAX_AUDIO_TOKENS = 32
OUTPUT_SAMPLE_RATE = 24000


@pytest.mark.xfail(
    reason=(
        "e2e on TT blocked by conditioning ttnn.group_norm tile-alignment "
        "(#5483 / tt-mlir #8935) and HiFi-GAN squeeze->prims::view_of "
        "functionalization (#5375 / #5388); flips to xpass once both land."
    ),
    strict=False,
)
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="XTTS_v2_Pipeline",
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
def test_xtts_v2_pipeline():
    """Run the full XTTS-v2 pipeline (all learned modules on TT) and check output."""
    import torch

    torch_xla = pytest.importorskip("torch_xla")
    import torch_xla.runtime as xr

    # Install the loader's own requirements (coqui-tts + torchaudio) for the run,
    # exactly as tests/runner/test_models.py does for component tests.
    from tests.runner.requirements import RequirementsManager

    RequirementsManager.capture_golden_state()
    with RequirementsManager.for_loader(_LOADER_PATH, framework="torch"):
        # Probe torchaudio inside the manager (after install); TTS is imported
        # lazily by the loader (its import chain needs the isin_mps_friendly shim
        # the loader installs first).
        pytest.importorskip("torchaudio", reason="torchaudio not installed")

        os.environ.setdefault("COQUI_TOS_AGREED", "1")

        xr.set_device_type("TT")
        if xr.global_runtime_device_count() < 1:
            pytest.skip("No TT device available")

        from third_party.tt_forge_models.xtts_v2.pytorch.pipeline import (
            run_xtts_pipeline,
        )

        output_path = "xtts_v2_pipeline_output.wav"
        output_file = Path(output_path)
        if output_file.exists():
            output_file.unlink()

        try:
            wav = run_xtts_pipeline(
                output_path=output_path,
                max_audio_tokens=MAX_AUDIO_TOKENS,
                seed=0,
            )
        except Exception as exc:  # deps missing / CPML weights ungated -> skip
            if "COQUI" in str(exc) or "download" in str(exc).lower():
                pytest.skip(f"Could not build/download XTTS-v2: {exc}")
            raise

        # Output artifact validity (mirrors the SDXL-Lightning e2e test).
        assert output_file.exists(), f"Output WAV {output_path} was not created"
        assert (
            wav.ndim == 3 and wav.shape[0] == 1
        ), f"Unexpected waveform shape {tuple(wav.shape)}"
        assert wav.shape[-1] > 0, "Pipeline produced an empty waveform"
        assert torch.isfinite(wav).all(), "Waveform contains non-finite samples"
