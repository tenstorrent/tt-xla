# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Krea Realtime Video — nightly e2e pipeline PCC test on Tenstorrent.

The pipeline (transformer on the mesh, VAE/encoder on CPU) lives in
tt_forge_models. This file owns the CPU verification: it loads CPU twins and,
via the pipeline's ``on_forward`` hook, runs each twin on the same inputs the
TT forward saw and asserts PCC >= ``PCC_THRESHOLD`` (compounded: the twin caches
evolve alongside the TT caches, so there is no per-forward gather of the sharded
TT cache to host).
"""

import pytest
from infra import ComparisonConfig, RunMode
from infra.evaluators import PccConfig, TorchComparisonEvaluator
from loguru import logger
from utils import BringupStatus, Category, ModelGroup

from third_party.tt_forge_models.krea_realtime_video.pytorch.pipeline import (
    NUM_INFERENCE_STEPS,
    PROMPT,
    SEED,
    KreaRealtimePipeline,
    init_crossattn_cache,
    init_kv_cache,
)
from third_party.tt_forge_models.krea_realtime_video.pytorch.src.model_utils import (
    DTYPE,
    KREA_REPO_ID,
    NUM_FRAMES_PER_BLOCK,
    SEQ_LENGTH,
    WAN_REPO_ID,
    load_text_encoder,
    load_transformer,
)

NUM_BLOCKS = 1  # TODO: >1 needs ttnn::sort stable=True support (flex_attention's create_block_mask): https://github.com/tenstorrent/tt-xla/issues/6041
PCC_THRESHOLD = 0.95

_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()


def _pcc(device_out, golden_out) -> float:
    return float(_PCC_EVALUATOR._compare_pcc(device_out, golden_out, _PCC_CONFIG))


class _TwinValidator:
    """CPU twin + per-forward PCC, driven by the pipeline's on_forward hook.

    Loaded AFTER the pipeline's setup() so the twin transformer inherits the
    class-level static-kv patch and the sinusoidal-embedding global.
    """

    def __init__(self):
        self.pccs = []

    def setup(self, num_heads, head_dim, num_tf_blocks):
        self.twin_text_encoder = load_text_encoder(WAN_REPO_ID, DTYPE)
        self.twin_transformer = load_transformer(KREA_REPO_ID, DTYPE)
        for blk in self.twin_transformer.blocks:
            blk.self_attn.local_attn_size = -1
            blk.self_attn.num_frame_per_block = NUM_FRAMES_PER_BLOCK
        # One cache entry per transformer layer (not per video block).
        self._kv = init_kv_cache(num_tf_blocks, num_heads, head_dim)
        self._ca = init_crossattn_cache(num_tf_blocks, num_heads, head_dim)

    def _record(self, label, tt_out, golden):
        p = _pcc(tt_out, golden)
        self.pccs.append((label, p))
        logger.info(f"[PCC] {label}: {p:.6f}")
        assert p >= PCC_THRESHOLD, f"{label} PCC {p:.6f} < {PCC_THRESHOLD}"

    def on_forward(self, kind, label, inputs, tt_out):
        if kind == "encoder":
            golden = self.twin_text_encoder(
                inputs["input_ids"], inputs["mask"]
            ).last_hidden_state
            self._record(label, tt_out, golden)
            self.twin_text_encoder = None  # used once
        elif kind == "transformer":
            golden = self.twin_transformer(
                x=inputs["x"],
                t=inputs["t"],
                context=inputs["context"],
                kv_cache=self._kv,
                seq_len=SEQ_LENGTH,
                crossattn_cache=self._ca,
                current_start=inputs["current_start"],
                cache_start=None,
            )
            self._record(label, tt_out, golden)
        elif kind == "block_start" and inputs["block_idx"] > 0:
            for e in self._kv:
                e["k"].zero_()
                e["v"].zero_()
                e["global_end_index"] = 0
                e["local_end_index"] = 0
        # vae_encode / recompute run on CPU -> no PCC.


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.llmbox
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="KreaRealtimeVideo_Pipeline",
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_krea_realtime_pipeline():
    """e2e Krea on TT (bf16), per-forward PCC vs an inline CPU twin."""
    validator = _TwinValidator()
    pipe = KreaRealtimePipeline(on_forward=validator.on_forward)
    pipe.setup()
    validator.setup(pipe._num_heads, pipe._head_dim, pipe._num_tf_blocks)

    pipe.generate(PROMPT, NUM_BLOCKS, NUM_INFERENCE_STEPS, SEED)

    worst_label, worst = min(validator.pccs, key=lambda lp: lp[1])
    logger.info(f"[PCC] worst: {worst_label} = {worst:.6f} (all >= {PCC_THRESHOLD})")


if __name__ == "__main__":
    test_krea_realtime_pipeline()
