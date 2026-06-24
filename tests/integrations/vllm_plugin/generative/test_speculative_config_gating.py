# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CompilationMode, CUDAGraphMode
from vllm_tt.platform import TTPlatform


class _SpecConfig:
    def __init__(self, method: str, use_ngram_gpu: bool = False):
        self.method = method
        self._use_ngram_gpu = use_ngram_gpu

    def use_ngram_gpu(self) -> bool:
        return self._use_ngram_gpu


def _make_vllm_config(speculative_config=None, async_scheduling: bool = False):
    return SimpleNamespace(
        additional_config={},
        speculative_config=speculative_config,
        model_config=SimpleNamespace(
            runner_type="generate",
            dtype=torch.bfloat16,
            use_mla=False,
            max_model_len=128,
        ),
        scheduler_config=SimpleNamespace(
            scheduler_cls="",
            is_multimodal_model=False,
            disable_chunked_mm_input=True,
            async_scheduling=async_scheduling,
            enable_chunked_prefill=True,
            chunked_prefill_enabled=True,
            max_num_batched_tokens=256,
            DEFAULT_MAX_NUM_BATCHED_TOKENS=2048,
        ),
        cache_config=SimpleNamespace(block_size=32),
        compilation_config=SimpleNamespace(
            mode=CompilationMode.DYNAMO_TRACE_ONCE,
            cudagraph_mode=CUDAGraphMode.NONE,
            backend="",
        ),
        parallel_config=SimpleNamespace(worker_cls="auto"),
    )


@pytest.fixture(autouse=True)
def _mock_tt_page_size(monkeypatch):
    monkeypatch.setattr(
        "vllm_tt.attention.TTAttentionBackend.get_page_size",
        lambda *_args, **_kwargs: 32,
    )


def test_tt_allows_ngram_speculative_decode_sync_only():
    cfg = _make_vllm_config(speculative_config=_SpecConfig(method="ngram"))

    TTPlatform.check_and_update_config(cfg)

    assert cfg.scheduler_config.scheduler_cls == "vllm_tt.scheduler.AscendScheduler"
    assert cfg.parallel_config.worker_cls == "vllm_tt.worker.TTWorker"


@pytest.mark.parametrize(
    "spec_cfg, async_scheduling, error_match",
    [
        (_SpecConfig(method="eagle"), False, "method='ngram'"),
        (_SpecConfig(method="ngram"), True, "synchronous scheduling"),
        (_SpecConfig(method="ngram", use_ngram_gpu=True), False, "ngram_gpu"),
    ],
)
def test_tt_rejects_unsupported_speculative_decode_configs(
    spec_cfg, async_scheduling, error_match
):
    cfg = _make_vllm_config(
        speculative_config=spec_cfg, async_scheduling=async_scheduling
    )

    with pytest.raises(NotImplementedError, match=error_match):
        TTPlatform.check_and_update_config(cfg)
