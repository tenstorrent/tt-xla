# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""XTTS-v2 (coqui/XTTS-v2) nightly end-to-end text-to-speech pipeline test.

Runs the full ``Xtts.inference`` path with every learned nn.Module on TT and, on
the real text/speaker inputs the pipeline actually uses, asserts three things:

* the output artifact is a finite, non-empty 24 kHz waveform,
* every autoregressive decode step matches a CPU replay of the same token
  sequence (per-step PCC), and
* the decode loop compiles a bounded number of graphs (prefill + decode) rather
  than recompiling per step.

The pipeline itself is reused from tt-forge-models; only the decode loop is
wrapped here, because the PCC comparison and the compile accounting have to
interleave with it. Needs coqui-tts, weights, and a TT device.
"""

import os
from pathlib import Path

import pytest
from infra import RunMode
from loguru import logger
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

# Cap on decode steps so the test stays bounded (each token is ~1024 output
# samples @ 24 kHz); the property under test is that the chain runs end-to-end
# and emits a valid waveform, not audio length.
#
# Kept well under the ~163 steps a full utterance takes because the CPU replay
# below costs roughly O(cap^2) -- it runs one CPU forward per step over a static
# cache sized prefix_len + cap. Measured warm-cache wall clock: 32 -> 79s,
# 64 -> 374s. 64 still exercises the reuse property (compile count goes flat and
# stays flat well before the tail).
MAX_AUDIO_TOKENS = 64
OUTPUT_SAMPLE_RATE = 24000

# Per-step correlation floor between the TT decode logits and the CPU replay.
PCC_THRESHOLD = 0.99


def _make_pcc_pipeline_cls():
    """Build the PCC/compile-checking ``XTTSPipeline`` subclass.

    Deferred into a function because importing the pipeline pulls in ``TTS``,
    which is only installed inside the ``RequirementsManager`` context.
    """
    import torch
    import torch_xla.debug.metrics as met
    from infra.evaluators import PccConfig, TorchComparisonEvaluator
    from infra.evaluators.evaluation_config import ComparisonConfig

    from third_party.tt_forge_models.xtts_v2.pytorch.pipeline import XTTSPipeline
    from third_party.tt_forge_models.xtts_v2.pytorch.src.model import GptCachedStep

    _evaluator = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
    _pcc_config = PccConfig()

    def pcc(device_out, golden_out) -> float:
        return float(_evaluator._compare_pcc(device_out, golden_out, _pcc_config))

    def compile_count() -> int:
        data = met.metric_data("CompileTime")
        return data[0] if data else 0

    class PccXTTSPipeline(XTTSPipeline):
        """``XTTSPipeline`` that also measures the decode loop.

        Records the TT logits of every decode step via a forward hook (so the
        upstream loop runs untouched), counts the graphs the loop compiled, then
        replays the exact token sequence TT produced through an uncompiled CPU
        copy of the same step module for a per-step golden.
        """

        def __init__(self, config):
            super().__init__(config)
            self.step_pccs = []
            self.compile_curve = []

        def _generate_codes_tt(self, gpt_cond_latent, text_tokens):
            tt_logits = []

            def record(_module, args, output):
                tt_logits.append(output[:, -1, :].detach().to("cpu").float())
                # Cumulative graph compilations observed after each decode call;
                # the curve is what shows the loop settling onto one reused graph.
                self.compile_curve.append(compile_count())
                shapes = tuple(
                    tuple(a.shape) if hasattr(a, "shape") else type(a).__name__
                    for a in args
                )
                print(
                    f"[DIAG] step={len(self.compile_curve) - 1} "
                    f"compiles={self.compile_curve[-1]} shapes={shapes}"
                )

            hook = self.decode_step.register_forward_hook(record)
            try:
                codes = super()._generate_codes_tt(gpt_cond_latent, text_tokens)
            finally:
                hook.remove()

            # The parent moves decode_step back to CPU before returning, so the
            # replay below runs on CPU weights.
            self.step_pccs = self._replay_on_cpu(
                gpt_cond_latent, text_tokens, codes, tt_logits
            )
            return codes

        def _replay_on_cpu(self, gpt_cond_latent, text_tokens, codes, tt_logits):
            """Re-run TT's own token sequence on CPU, returning per-step PCC.

            Uses a fresh, *uncompiled* ``GptCachedStep``: it shares weights with
            the pipeline (no second model in memory) but has no ``tt`` backend
            attached, so it really executes on CPU.
            """
            gpt = self.xtts.gpt
            cpu_step = GptCachedStep(self.xtts).eval()

            with torch.no_grad():
                gpt.compute_embeddings(gpt_cond_latent, text_tokens)
            prefix_emb = gpt.gpt_inference.cached_prefix_emb.clone()
            prefix_len = prefix_emb.shape[1]

            # TT produced one logits row per decode call: the prefill, then one
            # per generated token. Feed the same tokens in the same order.
            token_seq = [self.start_audio_token] + codes.squeeze(0).tolist()
            n_steps = min(len(tt_logits), len(token_seq))

            max_cache_len = prefix_len + n_steps
            cache = self._make_static_cache(max_cache_len, "cpu")

            def mask(valid):
                m = torch.zeros((1, max_cache_len), dtype=torch.long)
                m[:, :valid] = 1
                return m

            pccs = []
            with torch.no_grad():
                # Prefill: [prefix, START] -> logits for the first audio token.
                ids = torch.tensor([[token_seq[0]]], dtype=torch.long)
                pos = torch.tensor([0], dtype=torch.long)
                out = cpu_step(ids, pos, mask(prefix_len + 1), cache, prefix_emb)
                pccs.append(pcc(tt_logits[0], out[:, -1, :].float()))

                cur = prefix_len + 1
                for step in range(1, n_steps):
                    tok = torch.tensor([[token_seq[step]]], dtype=torch.long)
                    pos = torch.tensor([step], dtype=torch.long)
                    out = cpu_step(tok, pos, mask(cur + 1), cache, None)
                    pccs.append(pcc(tt_logits[step], out[:, -1, :].float()))
                    cur += 1
            for i, value in enumerate(pccs):
                logger.info(
                    "[PCC] decode step {}/{}: pcc={:.6f}", i, len(pccs) - 1, value
                )
            return pccs

    return PccXTTSPipeline


@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="XTTS_v2_Pipeline",
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_xtts_v2_pipeline():
    """Full XTTS-v2 chain on TT: waveform validity + per-step decode PCC."""
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
            XTTSConfig,
            save_wav,
        )

        output_path = "xtts_v2_pipeline_output.wav"
        output_file = Path(output_path)
        if output_file.exists():
            output_file.unlink()

        # Mirrors run_xtts_pipeline(), but with the measuring subclass so the
        # decode loop can be checked while the real pipeline drives it.
        torch_xla.set_custom_compile_options({"optimization_level": 0})
        pipeline_cls = _make_pcc_pipeline_cls()
        pipeline = pipeline_cls(XTTSConfig(max_audio_tokens=MAX_AUDIO_TOKENS, seed=0))

        try:
            pipeline.setup()
            wav = pipeline.run()
        except Exception as exc:  # deps missing / CPML weights ungated -> skip
            if "COQUI" in str(exc) or "download" in str(exc).lower():
                pytest.skip(f"Could not build/download XTTS-v2: {exc}")
            raise
        save_wav(wav, output_path)

        # 1) Output artifact validity (mirrors the SDXL-Lightning e2e test).
        assert output_file.exists(), f"Output WAV {output_path} was not created"
        assert (
            wav.ndim == 3 and wav.shape[0] == 1
        ), f"Unexpected waveform shape {tuple(wav.shape)}"
        assert wav.shape[-1] > 0, "Pipeline produced an empty waveform"
        assert torch.isfinite(wav).all(), "Waveform contains non-finite samples"

        # 2) Per-step decode correctness against a CPU replay of the same tokens.
        pccs = pipeline.step_pccs
        assert pccs, "no decode steps were recorded"
        worst = min(pccs)
        assert worst >= PCC_THRESHOLD, (
            f"decode-step logits diverged from CPU: worst PCC={worst:.5f} < "
            f"{PCC_THRESHOLD} over {len(pccs)} steps; per-step PCC={pccs}"
        )

        # 3) The decode loop settles onto one reused graph: after the warm-up
        # steps the cumulative compile count stops growing, so the cost does not
        # scale with the number of tokens. (Absolute count is not asserted -- the
        # loop legitimately builds several graphs while warming up.)
        curve = pipeline.compile_curve
        assert len(curve) >= 4, f"too few decode steps to judge reuse: {curve}"
        settled_from = len(curve) // 2
        tail_growth = curve[-1] - curve[settled_from]
        logger.info(
            "[decode] steps={} worst_pcc={:.6f} cumulative_compiles={} tail_growth={}",
            len(pccs),
            worst,
            curve,
            tail_growth,
        )
        assert tail_growth == 0, (
            "decode loop kept compiling instead of reusing its graph: "
            f"{tail_growth} new compilations over the last "
            f"{len(curve) - settled_from} of {len(curve)} steps; "
            f"cumulative compile counts per step={curve}"
        )
